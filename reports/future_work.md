# Phase 4: Future Work — Joint JEPA+RSSM Co-training

**Status:** Designed but explicitly out of scope for the current implementation.
This document records the design rationale for future execution.

## Motivation

Phases 1–3 treat the encoder and world model as separate components:
- Phase 2 pretrains the encoder independently (JEPA on static video)
- Phase 3 freezes the encoder and only trains the RSSM/actor-critic

This means the encoder's representations are optimized for **video prediction**
(the JEPA objective) but not specifically for **world modeling and planning**
(the RSSM/actor-critic objective). Phase 4 would close this gap.

## Proposed Approach

### Joint Loss Function

Train the encoder end-to-end with both objectives simultaneously:

```
L_total = α * L_JEPA + β * L_RSSM + γ * L_actor_critic
```

Where:
- `L_JEPA`: Self-supervised masked prediction loss (same as Phase 2)
- `L_RSSM`: World model loss (reconstruction + KL + reward prediction)
- `L_actor_critic`: Policy optimization (imagined returns)
- `α, β, γ`: Weighting coefficients (key hyperparameters to tune)

### Key Design Decisions

1. **Gradient conflict**: JEPA loss encourages the encoder to produce representations
   that are predictable from context. RSSM loss encourages representations useful for
   reconstruction. These objectives may conflict — requires careful loss balancing.

2. **EMA stability**: The JEPA target encoder uses EMA. When the encoder is also being
   updated by RSSM gradients, the EMA update rate needs to account for two gradient
   sources. May need to increase momentum (slower EMA) to maintain stability.

3. **Curriculum**: Start with JEPA-dominant training (high α, low β, zero γ) and
   gradually shift to RSSM-dominant as the encoder stabilizes. This prevents the
   RL signal from destabilizing early encoder learning.

4. **Compute**: This doubles the training compute per step (both losses computed
   on every batch). On the target hardware (single A4000), this likely requires
   halving the batch size or using gradient accumulation.

### Expected Outcome

If successful, the jointly-trained encoder should:
- Outperform both frozen-JEPA (Phase 3 Arm 2) and V-JEPA2 (Arm 3) on driving
  tasks because the representations are specifically tuned for world modeling
- Show faster convergence than the CNN baseline (Arm 1) because the JEPA
  objective provides a strong initialization signal

### Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Gradient conflict causes training instability | High | High | Gradient scaling, separate LR per component |
| Representation collapse under dual objectives | Medium | High | Collapse monitoring (from Phase 2), EMA momentum tuning |
| Compute budget insufficient on single GPU | High | Medium | Gradient accumulation, reduced batch size |
| No improvement over frozen-JEPA baseline | Medium | Low | Still a valid negative result if well-analyzed |

## Implementation Notes

If implementing Phase 4:
1. Start from the Phase 3 `custom_jepa` arm as the baseline
2. Unfreeze the encoder and add JEPA loss to the Phase 1 training loop
3. The `src/jepa/` modules are already structured for reuse — just instantiate
   the target encoder, predictor, and masking alongside the RSSM
4. Log both loss terms separately to diagnose gradient conflicts
5. Compare against all three Phase 3 arms to demonstrate improvement

## Timeline Estimate

With Phase 3 complete, Phase 4 would take approximately 2–3 weeks:
- Week 1: Integration and loss balancing experiments
- Week 2: Full training runs with different α/β/γ schedules
- Week 3: Analysis, comparison, and writeup

---

# CarDreamer Integration — Future Refactor

**Status:** Deferred. The current codebase uses a custom DreamerV3-style
implementation (RSSM, actor-critic, decoder, replay buffer). Migrating to
[CarDreamer](https://github.com/ucd-dare/CarDreamer) is a future refactor.

## Current State

The project reimplements the following DreamerV3 components from scratch:

| Component | Our file | CarDreamer equivalent |
|-----------|----------|----------------------|
| RSSM | `src/dreamer/rssm_wrapper.py` | `dreamerv3.models.RSSM` |
| Actor-Critic | `scripts/train_phase1.py` | `dreamerv3.agent.ActorCritic` |
| Image Decoder | `scripts/train_phase1.py` | `dreamerv3.models.Decoder` |
| Replay Buffer | inline in training loop | `dreamerv3.replay.ReplayBuffer` |
| Training Loop | `scripts/train_phase1.py` | `dreamerv3.train.train()` |

This was a pragmatic decision: integrating CarDreamer requires CARLA to be
available during development, and the project needed to be developed on
machines without CARLA. However, the custom implementation increases
debugging and algorithmic risk compared to using CarDreamer's battle-tested
code.

## Why Switch to CarDreamer

1. **Tested RSSM**: CarDreamer's DreamerV3 implementation has been validated
   on multiple CARLA tasks. Our custom RSSM may have subtle bugs in KL
   balancing, categorical sampling, or imagination rollouts that are hard
   to diagnose.

2. **Task library**: CarDreamer provides pre-built CARLA tasks (lane following,
   right turn, intersection crossing) with proper route planning, traffic,
   and success criteria — all things we currently approximate.

3. **Replay buffer**: CarDreamer's replay buffer handles episode boundaries,
   sequence sampling with configurable train ratio, and prioritized replay.
   Our inline list-based buffer is naive.

4. **Reduced maintenance**: By depending on CarDreamer, we only maintain the
   encoder swap logic and JEPA pretraining — not the entire RL stack.

## Migration Plan

1. Add CarDreamer as a Git submodule or pip dependency
2. Replace `src/dreamer/rssm_wrapper.py` with imports from CarDreamer
3. Replace `scripts/train_phase1.py` with CarDreamer's training loop, modified
   to accept our `EncoderAdapter` as a drop-in encoder replacement
4. Remove custom Actor, Critic, ImageDecoder, RewardPredictor, ContinuePredictor
5. Keep `src/dreamer/encoder_adapter.py` — this is our contribution layer
6. Update Phase 3 to use CarDreamer's training function with the `--arm` flag
   controlling only the encoder

## Estimated Effort

~1 week if CarDreamer's API is stable. The main risk is API compatibility
between CarDreamer's expected encoder interface and our adapter pattern.
