"""Small continuous RSSM comparison arm with ordered controls and prior-only inference.

Future encoder targets enter only posterior_kl during training. The deployment
forward method has no future observation/label input. This is untrained code,
not a planner improvement or a substitute for reactive branch calibration.
"""

import torch
from torch import nn
from torch.nn import functional as F


class RSSMDynamics(nn.Module):
    def __init__(
        self,
        token_dim=384,
        spatial_tokens=48,
        hidden=128,
        stochastic=32,
        steps=8,
        controls_per_step=10,
    ):
        super().__init__()
        self.spatial_tokens, self.steps, self.controls_per_step = (
            spatial_tokens,
            steps,
            controls_per_step,
        )
        self.hidden, self.stochastic = hidden, stochastic
        self.context = nn.Sequential(nn.Linear(token_dim, 64), nn.LayerNorm(64), nn.SiLU())
        self.summarize = nn.Linear(spatial_tokens * 64, hidden)
        self.condition = nn.Sequential(nn.Linear(35, hidden), nn.SiLU())
        self.controls = nn.GRU(3, hidden, batch_first=True)
        self.transition = nn.GRUCell(hidden + stochastic, hidden)
        self.prior = nn.Linear(hidden, 2 * stochastic)
        self.posterior = nn.Linear(hidden * 2, 2 * stochastic)
        self.spatial_state = nn.Linear(hidden + stochastic, 64)
        self.position = nn.Parameter(torch.randn(spatial_tokens, 64) * 0.02)
        self.decode = nn.Linear(64, token_dim)
        self.cost = nn.Linear(hidden + stochastic, 3)

    def check(self, past_tokens, ordered_controls):
        if (
            past_tokens.ndim != 3
            or past_tokens.shape[1] != self.spatial_tokens
            or ordered_controls.shape
            != (past_tokens.shape[0], self.steps * self.controls_per_step, 3)
            or not torch.isfinite(past_tokens).all()
            or not torch.isfinite(ordered_controls).all()
        ):
            raise ValueError("Past spatial grid and complete ordered control horizon required")

    @staticmethod
    def gaussian(parameters):
        mean, raw = parameters.chunk(2, -1)
        std = F.softplus(raw) + 0.1
        return mean, std

    def forward(self, past_tokens, ordered_controls, sample=False, conditioning=None):
        self.check(past_tokens, ordered_controls)
        spatial = self.context(past_tokens)
        state = torch.tanh(self.summarize(spatial.flatten(1)))
        if conditioning is not None:
            if conditioning.shape != (len(state), 35) or not torch.isfinite(conditioning).all():
                raise ValueError("Declared16-point navigation and3 ego-motion values required")
            state = state + self.condition(conditioning)
        latent = state.new_zeros(len(state), self.stochastic)
        actions, _ = self.controls(ordered_controls)
        future, costs, states, means, stds = [], [], [], [], []
        for horizon in range(self.steps):
            through = (horizon + 1) * self.controls_per_step - 1
            state = self.transition(torch.cat([actions[:, through], latent], -1), state)
            mean, std = self.gaussian(self.prior(state))
            latent = mean + std * torch.randn_like(std) if sample else mean
            combined = torch.cat([state, latent], -1)
            features = spatial + self.position[None] + self.spatial_state(combined)[:, None]
            future.append(self.decode(features))
            costs.append(self.cost(combined))
            states.append(state)
            means.append(mean)
            stds.append(std)
        return dict(
            future_tokens=torch.stack(future, 1),
            outcome_logits=torch.stack(costs, 1),
            deterministic_states=torch.stack(states, 1),
            prior_mean=torch.stack(means, 1),
            prior_std=torch.stack(stds, 1),
        )

    def posterior_kl(self, prediction, future_tokens, valid):
        """Training-only diagonal Gaussian posterior; masked censored tails stay unused."""
        if future_tokens.shape[:3] != (
            len(prediction["prior_mean"]),
            self.steps,
            self.spatial_tokens,
        ):
            raise ValueError("Explicit future encoder supervision shape required")
        if valid.shape != future_tokens.shape[:2]:
            raise ValueError("Per-horizon validity required")
        supervised = torch.where(valid[..., None, None] > 0, future_tokens.detach(), 0.0)
        target = self.context(supervised).flatten(2)
        summary = torch.tanh(self.summarize(target))
        q_mean, q_std = self.gaussian(
            self.posterior(torch.cat([prediction["deterministic_states"], summary], -1))
        )
        p_mean, p_std = prediction["prior_mean"], prediction["prior_std"]
        kl = (
            torch.log(p_std / q_std)
            + (q_std.square() + (q_mean - p_mean).square()) / (2 * p_std.square())
            - 0.5
        )
        # Free nats keep this posterior regularizer from overwhelming latent prediction.
        per_horizon = kl.sum(-1).clamp_min(1.0)
        return (per_horizon * valid).sum() / valid.sum().clamp_min(1)


def prediction_objective(prediction, future_tokens, latent_valid, outcomes, cost_valid):
    """Factual executed controls only; censored outcomes cannot become safe labels."""
    predicted = prediction["future_tokens"]
    if predicted.shape != future_tokens.shape or latent_valid.shape != predicted.shape[:2]:
        raise ValueError("Future target and horizon mask mismatch")
    logits = prediction["outcome_logits"]
    if outcomes.shape != logits.shape or cost_valid.shape not in (
        logits.shape[:2],
        logits.shape,
    ):
        raise ValueError("Collision/red/progress targets and separate cost mask required")
    if (
        not torch.isfinite(latent_valid).all()
        or not torch.isfinite(cost_valid).all()
        or ((latent_valid != 0) & (latent_valid != 1)).any()
        or ((cost_valid != 0) & (cost_valid != 1)).any()
    ):
        raise ValueError("Finite binary factual supervision masks required")
    supervised = torch.where(latent_valid[..., None, None] > 0, future_tokens.detach(), 0.0)
    component_valid = (
        cost_valid[..., None].expand_as(logits)
        if cost_valid.shape == logits.shape[:2]
        else cost_valid
    )
    labeled_costs = torch.where(component_valid > 0, outcomes, 0.0)
    if (
        not torch.isfinite(supervised).all()
        or not torch.isfinite(labeled_costs).all()
        or (labeled_costs[..., :2] < 0).any()
        or (labeled_costs[..., :2] > 1).any()
    ):
        raise ValueError(
            "Finite valid future targets and probability-valued hazard labels required"
        )
    latent_error = F.smooth_l1_loss(predicted.float(), supervised.float(), reduction="none").mean(
        (-1, -2)
    )
    hazard = F.binary_cross_entropy_with_logits(
        logits[..., :2].float(), labeled_costs[..., :2].float(), reduction="none"
    )
    progress = F.smooth_l1_loss(
        logits[..., 2].float() / 20, labeled_costs[..., 2].float() / 20, reduction="none"
    )
    latent = (latent_error * latent_valid).sum() / latent_valid.sum().clamp_min(1)
    hazard_mask = component_valid[..., :2]
    progress_mask = component_valid[..., 2]
    hazard_loss = (hazard * hazard_mask).sum() / hazard_mask.sum().clamp_min(1)
    progress_loss = (progress * progress_mask).sum() / progress_mask.sum().clamp_min(1)
    cost = hazard_loss + progress_loss
    return latent + cost, dict(latent=latent, cost=cost)
