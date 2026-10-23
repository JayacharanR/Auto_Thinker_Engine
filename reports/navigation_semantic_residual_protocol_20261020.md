# Structured stop correction — before fitting or student rule outcomes

The frozen CNN-facts primary stops indefinitely in TRAINengineering route101.
Its closed native trace predictsRed.770 but applicable-light presence.260,
distance44.05m, and every learned hazard probability below.043; it requests
zero speed/brake.662. The available video shows open road ahead. This is a
post-hoc diagnosis, not certified visibility or proof that every false hold
has the same cause. ResNet-facts also drove through an applicable held red
despite an earlier strong red prediction. Neither model is safe or promoted.

Preserve all current camera pilots and the prospectively selected CNN-facts
72-case M3study. No current checkpoint, controller, source or outcome changes.
Register this bounded M2engineering countermeasure before its first fit and
before any student M3outcome is observed. Keep the M4matrix unchanged.

Freeze the selected CNN-facts parent (SHA
e3a688576d007158ee9d61527de83c4eed839b2bd6ddb0d1bc34e5f88181ea48).
Cache its FP32 whole-episode predictions on exactly the corrected55TRAIN/
13VAL episodes,15712/2822timestamps. Verify the original1312scored validation
positions/560stops/582controller-go positions reproduce the selected parent.
No new annotations, geography, heldout_dev or Town05. Auxiliary supervision
already used15712distinct TRAINtimestamps; disclose reuse in this engineering
experiment rather than counting it as a low-label encoder comparison.

Train two identical small causal residuals from zero output initialization:
predicted-facts versus TRAINtimestamp-mean constant facts. Both also receive
the same frozen9parent stop logits. Input facts are5light probabilities,
2presence probabilities,2distances/80 clipped to[0,1],4hazard probabilities,
3presence×colored-state products and presence×(1-normalized stop distance).
These17derived camera predictions never directly command control. A32unit
GRU predicts9bounded logit corrections,4*tanh(delta). Steering and raw cruise
speeds remain the identical frozen parent; no raw-image/route/ego-speed bypass
is added to the residual, no engineered rule or emergency override is enabled.

Fresh seed42,1000updates/arm, identical family-balanced whole-episode draws,
AdamW learningrate.0003, gradientclip1, full causal history/burn-in0. Loss is
the parent's2×current-stop BCE plus masked future-stop BCE only. Validate at
0/250/500/750/1000; checkpoint optimizer/RNG/sampler every100. The entire cache
and paired fits have a1800s cap and live approved storage guard. No blind
longer continuation if selected0. Small prediction/cache/checkpoint artifacts
must stay below.1GB; original RGB and failed/native results remain retained.

Prospective engineering selector: require at least2percentage-point stop-recall
gain; false holds <=parent+1point; go-speedMAE <=parent+.2m/s; stop recall for
each family with>=30stopped validation positions >=parent−2points; same old
composite score improves by>.005. Otherwise retain0. Select the lowest old
composite among admitted scheduled validations. Facts merit a new native pilot
only if admitted and its score beats the matched constant-facts control by.01
without lower stop recall. This does not change any driving metric or gate.

Queue extraction/fitting only after the existing M3followthrough terminates,
under the global navigation lock; no competing GPU/native work. Retain all
raw fits, five validations, exact draws, parent identity and rejected candidates.
Deployment needs a new explicit architecture/source generation, causal worker
adapter and unchanged bounded native pilot. Passing this selector alone is
neither rule understanding nor qualification for the300encounters/M6.
