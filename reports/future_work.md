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
