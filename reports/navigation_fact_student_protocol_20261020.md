# Structured camera-fact student — before fitting

Engineering countermeasure separate from the corrected M4 representation study.
The spatial driving heads learn speed/stop imitation but have no direct semantic
supervision for the signal state, stop-line distance or hazard that caused a
teacher decision. The old speed-conditioned head could use being stationary
as a shortcut; removing that neural input improved movement without establishing
rule reliability. Test whether explicit facts help the remaining decision task.

Use only the corrected55TRAIN episodes/15712timestamps and unchanged13validation
episodes. Add their existing collection-side light/presence/distance/hazard labels
to the same fresh visual-decision architecture, with learned auxiliary heads on
the recurrent state. No teacher/map/actor value enters forward inference. These
are privileged routing-fact training targets, not visibility-certified RGB labels;
out-of-view cases require memory and can remain unidentifiable. M3 tests, rather
than a classifier accuracy alone, assess actual causal rule behavior.

Four predeclared engineering candidates: frozen ImageNet ResNet18 and trainable
CNN, each with plain-driving versus added-fact supervision, seed42,2000updates,
same full-episode sampling/five scheduled validations/old balanced-stop selector.
All four supervise from episode start (burn-in0). Earlier diagnosis found that
273/392 stationary-go labels were excluded by burn-in8, although deployment also
starts its recurrent state cold. Keep the paired plain variants to distinguish
startup supervision from semantic supervision. The corrected67-cell M4 matrix
retains its original burn-in8 and remains unchanged; disclose this M2 training
change, rather than claiming an isolated auxiliary-head effect against it.
Initialize fresh, no old contaminated or
full-label driver inherited. Keep the exact parent driving loss and add light
cross-entropy weight1, presence BCE.5, present-distance smooth-L1(normalized40m)
weight1 and hazard BCE.5. Unknown light class4 is masked; no validation-derived
class weight, threshold search or best-seed driving selection. Neural ego speed
remains omitted; physical speed still reaches the controller. Publish both
candidate outcomes, auxiliary errors and the unchanged corrected M4 full-label
baselines. These extra semantic labels are explicitly counted15712, never hidden
inside the low-label encoder comparisons.

First evaluate with all engineered rule/emergency overrides off. A separately
declared learned-perception rule layer may enforce predicted red/stop behavior,
with requested/executed controls and interventions reported. Never claim rule
understanding from that layer. Native pilot/long driving/300development evidence,
every recorder contact and unchanged final qualification gates remain mandatory.
Do not modify or restart the active corrected67-fit matrix to include these models.
This is a prospective countermeasure, not a trained or integrated policy yet.
