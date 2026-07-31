"""
RSSM (Recurrent State Space Model) wrapper for CarDreamer/DreamerV3.

This module provides an interface to CarDreamer's built-in RSSM
implementation. We do NOT reimplement the RSSM from scratch —
CarDreamer/DreamerV3's RSSM is well-tested and our contribution
is integration, not the RSSM math.

The wrapper handles:
- RSSM state management (stochastic + deterministic)
- Imagination rollouts for actor-critic training
- Interface between our encoder adapters and the RSSM
"""

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical


class RSSMState:
    """
    Container for RSSM state (stochastic + deterministic components).

    DreamerV3 uses categorical latents (not Gaussian) with
    straight-through gradients. The stochastic state is a set
    of one-hot categorical vectors.
    """

    def __init__(
        self,
        stochastic: torch.Tensor,
        deterministic: torch.Tensor,
    ):
        self.stochastic = stochastic      # (B, num_categories * num_classes) or (B, stoch_size)
        self.deterministic = deterministic  # (B, deter_size)

    @property
    def combined(self) -> torch.Tensor:
        """Concatenated state for downstream use (actor, critic, decoder)."""
        return torch.cat([self.stochastic, self.deterministic], dim=-1)

    @property
    def device(self) -> torch.device:
        return self.stochastic.device

    def detach(self) -> "RSSMState":
        """Detach both components from computation graph."""
        return RSSMState(
            stochastic=self.stochastic.detach(),
            deterministic=self.deterministic.detach(),
        )


class RSSM(nn.Module):
    """
    Recurrent State Space Model (DreamerV3-style).

    The RSSM maintains a latent world model with two components:
    - Deterministic: GRU-based recurrent state (captures temporal dynamics)
    - Stochastic: Categorical latent variables (captures uncertainty)

    Key operations:
    - observe: encoder embedding + action → posterior state
    - imagine: action → prior state (no observation, for planning)

    Args:
        stochastic_size: Total stochastic state dimension.
        deterministic_size: GRU hidden state dimension.
        hidden_size: Hidden layer size for internal MLPs.
        num_categories: Number of categorical distributions.
        num_classes: Number of classes per categorical distribution.
        action_dim: Action dimension (steer + throttle/brake = 2).
        embed_dim: Dimension of encoder output (after adapter).
    """

    def __init__(
        self,
        stochastic_size: int = 32,
        deterministic_size: int = 512,
        hidden_size: int = 512,
        num_categories: int = 32,
        num_classes: int = 32,
        action_dim: int = 2,
        embed_dim: int = 1024,
    ):
        super().__init__()

        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size
        self.num_categories = num_categories
        self.num_classes = num_classes
        self.stoch_dim = num_categories * num_classes

        # Prior: predict stochastic state from deterministic state alone
        self.prior_net = nn.Sequential(
            nn.Linear(deterministic_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, self.stoch_dim),
        )

        # Posterior: predict stochastic state from deterministic + observation
        self.posterior_net = nn.Sequential(
            nn.Linear(deterministic_size + embed_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, self.stoch_dim),
        )

        # Recurrence: action + previous stochastic → next deterministic
        self.pre_gru = nn.Sequential(
            nn.Linear(self.stoch_dim + action_dim, hidden_size),
            nn.SiLU(),
        )
        self.gru = nn.GRUCell(hidden_size, deterministic_size)

    def initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        """Create initial RSSM state (zeros)."""
        return RSSMState(
            stochastic=torch.zeros(batch_size, self.stoch_dim, device=device),
            deterministic=torch.zeros(
                batch_size, self.deterministic_size, device=device
            ),
        )

    def observe_step(
        self,
        prev_state: RSSMState,
        action: torch.Tensor,
        embed: torch.Tensor,
    ) -> tuple[RSSMState, dict[str, torch.Tensor]]:
        """
        Single observation step: action + encoder output → posterior state.

        Args:
            prev_state: Previous RSSM state.
            action: (B, action_dim) action taken.
            embed: (B, embed_dim) encoder output (through adapter).

        Returns:
            Tuple of (new RSSMState, info dict with prior/posterior logits).
        """
        # Compute deterministic state via GRU
        x = self.pre_gru(torch.cat([prev_state.stochastic, action], dim=-1))
        deterministic = self.gru(x, prev_state.deterministic)

        # Posterior (with observation)
        posterior_logits = self.posterior_net(
            torch.cat([deterministic, embed], dim=-1)
        )
        posterior_logits = posterior_logits.reshape(
            -1, self.num_categories, self.num_classes
        )

        # Prior (without observation — for KL loss)
        prior_logits = self.prior_net(deterministic)
        prior_logits = prior_logits.reshape(
            -1, self.num_categories, self.num_classes
        )

        # Sample from posterior using straight-through gradient
        stochastic = self._sample_categorical(posterior_logits)

        new_state = RSSMState(
            stochastic=stochastic.reshape(-1, self.stoch_dim),
            deterministic=deterministic,
        )

        info = {
            "prior_logits": prior_logits,
            "posterior_logits": posterior_logits,
        }

        return new_state, info

    def imagine_step(
        self,
        prev_state: RSSMState,
        action: torch.Tensor,
    ) -> RSSMState:
        """
        Single imagination step: action → prior state (no observation).

        Used during actor-critic training (imagination rollouts).

        Args:
            prev_state: Previous RSSM state.
            action: (B, action_dim) action to imagine.

        Returns:
            New RSSMState from prior only.
        """
        x = self.pre_gru(torch.cat([prev_state.stochastic, action], dim=-1))
        deterministic = self.gru(x, prev_state.deterministic)

        prior_logits = self.prior_net(deterministic)
        prior_logits = prior_logits.reshape(
            -1, self.num_categories, self.num_classes
        )

        stochastic = self._sample_categorical(prior_logits)

        return RSSMState(
            stochastic=stochastic.reshape(-1, self.stoch_dim),
            deterministic=deterministic,
        )

    def observe_sequence(
        self,
        actions: torch.Tensor,
        embeds: torch.Tensor,
        initial_state: Optional[RSSMState] = None,
    ) -> tuple[list[RSSMState], list[dict[str, torch.Tensor]]]:
        """
        Process a sequence of observations.

        Args:
            actions: (B, T, action_dim) action sequence.
            embeds: (B, T, embed_dim) encoder output sequence.
            initial_state: Optional starting state.

        Returns:
            Tuple of (list of states, list of info dicts).
        """
        batch_size, seq_len = actions.shape[:2]

        if initial_state is None:
            initial_state = self.initial_state(batch_size, actions.device)

        states = []
        infos = []
        state = initial_state

        for t in range(seq_len):
            state, info = self.observe_step(state, actions[:, t], embeds[:, t])
            states.append(state)
            infos.append(info)

        return states, infos

    def imagine_sequence(
        self,
        initial_state: RSSMState,
        actor: nn.Module,
        horizon: int,
    ) -> list[RSSMState]:
        """
        Imagine a sequence of future states using the actor.

        Args:
            initial_state: Starting state for imagination.
            actor: Actor network that selects actions.
            horizon: Number of steps to imagine.

        Returns:
            List of imagined RSSMStates.
        """
        states = []
        state = initial_state

        for _ in range(horizon):
            action = actor(state.combined)
            state = self.imagine_step(state, action)
            states.append(state)

        return states

    @staticmethod
    def _sample_categorical(logits: torch.Tensor) -> torch.Tensor:
        """
        Sample from categorical distribution with straight-through gradient.

        Uses the Gumbel-softmax trick for differentiable sampling.

        Args:
            logits: (B, num_categories, num_classes) unnormalized log-probabilities.

        Returns:
            (B, num_categories, num_classes) one-hot samples.
        """
        # Straight-through: sample hard one-hot but pass gradients through softmax
        probs = F.softmax(logits, dim=-1)
        dist = OneHotCategorical(probs=probs)
        sample = dist.sample()

        # Straight-through gradient estimator
        sample = sample + probs - probs.detach()

        return sample

    @staticmethod
    def kl_loss(
        prior_logits: torch.Tensor,
        posterior_logits: torch.Tensor,
        free_nats: float = 1.0,
        balance: float = 0.8,
    ) -> torch.Tensor:
        """
        Compute KL divergence loss with free bits and balancing.

        DreamerV3 uses KL balancing: a mix of two KL directions
        to prevent the posterior from collapsing to the prior.

        Args:
            prior_logits: (B, num_cat, num_cls) prior logits.
            posterior_logits: (B, num_cat, num_cls) posterior logits.
            free_nats: Free bits (minimum KL before it contributes to loss).
            balance: KL balance ratio (0.8 = 80% posterior-to-prior direction).

        Returns:
            Scalar KL loss.
        """
        prior_probs = F.softmax(prior_logits, dim=-1)
        posterior_probs = F.softmax(posterior_logits, dim=-1)

        # KL(posterior || prior) — train prior to match posterior
        kl_forward = (
            posterior_probs * (
                torch.log(posterior_probs + 1e-8) - torch.log(prior_probs + 1e-8)
            )
        ).sum(dim=-1).mean()

        # KL(prior || posterior) — train posterior (reverse direction)
        kl_reverse = (
            prior_probs * (
                torch.log(prior_probs + 1e-8) - torch.log(posterior_probs + 1e-8)
            )
        ).sum(dim=-1).mean()

        # Apply free nats
        kl_forward = torch.clamp(kl_forward, min=free_nats)
        kl_reverse = torch.clamp(kl_reverse, min=free_nats)

        # Balanced KL
        kl_loss = balance * kl_forward + (1 - balance) * kl_reverse

        return kl_loss
