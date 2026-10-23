# Spatial risk-path repair, before any second-generation fit

The six completed factual models remain immutable and unqualified. The v1
JEPA risk path initializes its recurrent state from the mean of all48tokens;
its spatial attention only feeds the latent decoder, never its collision cost.
Permuting complete view/token assignments is mathematically invisible to risk,
apart from floating-point summation. This is an architectural limitation, not
proof that it caused every validation miss. Encoder tokens contain positional
information internally; the diagnosed invariance concerns assignment of those
already encoded tokens to camera/grid slots.

Synthetic input tests of all selected trained checkpoints are retained at
outputs/navigation_dynamics_spatial_diagnosis_20261020. They are not physical
camera interventions, calibration or driving outcomes. RSSM already uses an
ordered flatten and does not have this exact invariance.

Implement a separate untrained spatial JEPA v2, retaining front/left/right and
row/column identity with explicit embeddings. Four attention queries read the
ordered scene into the action recurrence. Predict each future spatial grid
causally from only controls through that horizon. A separate attention readout
uses that predicted hidden grid to estimate collision/red/progress costs;
collision gradients must reach the spatial decoder. View identities are fixed
rig metadata, not simulator actor, map-signal or privileged labels. This is a
spatial feature model, not a geometric3D reconstruction claim.

Do not overwrite/relabel v1 fits or promote v2 from construction tests. No v2
fit is launched by this protocol. Before fitting, register new factual branch
coverage, identical TRAIN/VAL windows and sampling/selector/compute budgets
for JEPA/RSSM comparisons, including parameter and measured latency differences.
Retain zero-red-positive limitation until genuine independent red outcomes
exist; mask/freeze unsupported costs. Evaluate missed hazards, always-safe
baselines and same-candidate constant-velocity ranking before planner promotion.
The old encoder-label study, native candidate selection and scoring are unchanged.
