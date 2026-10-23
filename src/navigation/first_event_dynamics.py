"""Optional causal first-event models; unqualified and outside deployed agents."""

import torch
from torch import nn
from torch.nn import functional as F


def hazard_nll(logits, events, valid, reduction="mean"):
    """Proper discrete censored survival NLL, summed through each observed prefix."""
    if (
        logits.shape != events.shape
        or logits.shape != valid.shape
        or logits.ndim != 2
        or logits.shape[1] != 80
        or valid.dtype != torch.bool
        or not torch.isfinite(logits).all()
    ):
        raise ValueError("Finite batchx80hazards and Boolean prefix masks required")
    targets = torch.where(valid, events, 0.0)
    if (
        not torch.isfinite(targets).all()
        or not ((targets == 0) | (targets == 1)).all()
        or (targets.sum(1) > 1).any()
        or (valid[:, 1:] & ~valid[:, :-1]).any()
        or (targets[:, :-1].bool() & valid[:, 1:]).any()
    ):
        raise ValueError("Valid prefix must end at first event or censoring")
    loss = (F.binary_cross_entropy_with_logits(logits, targets, reduction="none") * valid).sum(1)
    if reduction == "none":
        return loss
    if reduction != "mean":
        raise ValueError("Mean over observed sequences or unreduced sequence NLL only")
    observed = valid.any(1)
    return (loss * observed).sum() / observed.sum().clamp_min(1)


class FirstEventDynamics(nn.Module):
    """Equal RGB-object history and spatial-latent inputs for both learned arms.

    The future-control GRU is causal. JEPA uses deterministic latent prediction;
    RSSM adds a recurrent Gaussian prior with a training-only posterior regularizer.
    Hazard gradients pass through the predicted spatial-token decoder. No future
    image, event metadata, role, town, actor, teacher or rule state enters forward.
    """

    def __init__(self, kind="jepa", hidden=96, token_dim=384, stochastic=32):
        super().__init__()
        if kind not in ("jepa", "rssm"):
            raise ValueError("DeclaredJEPAorRSSMkind required")
        self.kind, self.token_dim, self.hidden = kind, token_dim, hidden
        self.visual = nn.Sequential(nn.Linear(token_dim, 32), nn.LayerNorm(32), nn.SiLU())
        self.visual_summary = nn.Linear(48 * 32, hidden)
        self.object = nn.Sequential(nn.Linear(20, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU())
        self.view = nn.Embedding(3, 32)
        self.register_buffer("views", torch.arange(3).repeat_interleave(100))
        self.object_scores = nn.Linear(32, 4)
        self.null = nn.Parameter(torch.zeros(1, 1, 32))
        self.object_pool = nn.Linear(4 * 32, hidden)
        self.history = nn.GRU(hidden + 3, hidden, batch_first=True)
        self.condition = nn.Sequential(nn.Linear(41, hidden), nn.SiLU())
        if kind == "jepa":
            self.transition = nn.GRU(3, hidden, batch_first=True)
            combined = hidden
        else:
            self.transition = nn.GRUCell(3 + stochastic, hidden)
            self.prior = nn.Linear(hidden, 2 * stochastic)
            self.posterior = nn.Linear(hidden + 48 * 32, 2 * stochastic)
            self.stochastic = stochastic
            combined = hidden + stochastic
        self.spatial = nn.Linear(combined, 48 * 32)
        self.decode = nn.Linear(32, token_dim)
        self.cost_grid = nn.Sequential(nn.Linear(token_dim, 8), nn.SiLU())
        self.hazard = nn.Sequential(nn.Linear(48 * 8, 64), nn.SiLU(), nn.Linear(64, 1))

    @staticmethod
    def gaussian(parameters):
        mean, raw = parameters.chunk(2, -1)
        return mean, F.softplus(raw) + 0.1

    def forward(self, tokens, controls, conditioning, objects, presence, past_ego):
        batch = len(tokens)
        if (
            tokens.shape != (batch, 48, self.token_dim)
            or controls.shape != (batch, 80, 3)
            or conditioning.shape != (batch, 41)
            or objects.shape != (batch, 8, 3, 100, 20)
            or presence.shape != (batch, 8, 3, 100)
            or presence.dtype != torch.bool
            or past_ego.shape != (batch, 8, 3)
            or not all(
                torch.isfinite(x).all() for x in (tokens, controls, conditioning, objects, past_ego)
            )
        ):
            raise ValueError(
                "Declared spatial grid, actions, own state and shared RGB history required"
            )
        visual = self.visual(tokens)
        rows = self.object(objects.reshape(batch * 8, 300, 20)) + self.view(self.views)[None]
        rows = torch.cat([rows, self.null.expand(batch * 8, -1, -1)], 1)
        known = torch.cat([presence.reshape(batch * 8, 300), presence.new_ones(batch * 8, 1)], 1)
        weights = self.object_scores(rows).masked_fill(~known[..., None], -torch.inf).softmax(1)
        pooled = torch.einsum("bnh,bnd->bhd", weights, rows).flatten(1)
        history = self.object_pool(pooled).reshape(batch, 8, self.hidden)
        _, memory = self.history(torch.cat([history, past_ego], -1))
        state = torch.tanh(
            self.visual_summary(visual.flatten(1)) + memory[0] + self.condition(conditioning)
        )
        extra = {}
        if self.kind == "jepa":
            combined, _ = self.transition(controls, state[None])
        else:
            latent = state.new_zeros(batch, self.stochastic)
            states, latents, means, stds = [], [], [], []
            for tick in range(80):
                state = self.transition(torch.cat([controls[:, tick], latent], -1), state)
                mean, std = self.gaussian(self.prior(state))
                latent = mean  # Deterministic prior-only inference, no posterior leakage.
                states.append(state)
                latents.append(latent)
                means.append(mean)
                stds.append(std)
            states, latents = torch.stack(states, 1), torch.stack(latents, 1)
            combined = torch.cat([states, latents], -1)
            extra = dict(
                states=states, prior_mean=torch.stack(means, 1), prior_std=torch.stack(stds, 1)
            )
        predicted = self.spatial(combined).reshape(batch, 80, 48, 32) + visual[:, None]
        future = tokens[:, None] + self.decode(predicted)
        logits = self.hazard(self.cost_grid(future).flatten(2))[..., 0].float()
        cumulative = -torch.expm1(F.logsigmoid(-logits).cumsum(1))
        return dict(future_tokens=future[:, 9::10], logits=logits, cumulative=cumulative, **extra)

    def posterior_kl(self, prediction, targets, valid):
        if self.kind != "rssm":
            return targets.new_zeros(())
        safe = torch.where(valid[..., None, None], targets.detach(), 0.0)
        encoded = self.visual(safe).flatten(2)
        q_mean, q_std = self.gaussian(
            self.posterior(torch.cat([prediction["states"][:, 9::10], encoded], -1))
        )
        p_mean, p_std = prediction["prior_mean"][:, 9::10], prediction["prior_std"][:, 9::10]
        kl = (
            torch.log(p_std / q_std)
            + (q_std.square() + (q_mean - p_mean).square()) / (2 * p_std.square())
            - 0.5
        ).sum(-1)
        return (kl.clamp_min(1.0) * valid).sum() / valid.sum().clamp_min(1)
