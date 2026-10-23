# RGB housing-context verifier: registered before mining or fitting

Previous bulb and hard-background studies remain failed and unchanged.
The frozen bulb proposal model finds56/72known lights,44withcorrectstate, but
37,241backgroundpeaks. Penalizing hard background loses17correctlights. Test
whether wider surrounding RGB can distinguish a housing from local bright edges.
This hypothesis is not established by the earlier experiments.

Freeze proposal checkpoint SHA27ddcc86e6be56a2edc7033e5ac9b47100f0c305d86e34893ced3cb20000b193,
threshold.5,8-connected one-peak rule and all proposals (no runtime top-Kcap).
Keep the audited876TRAINimages/sevenlocations and original144VALimages/72labels.
TRAINproposal inference uses batch3, sameBF16model. Reuse ALLoriginal frozen
validation proposals/pixel coordinates and verify their source/checkpoint hashes;
never mine training examples from validation or change a validation label.

Verifier input is a32x32native RGB patch plus a96x96RGBcontext patch averaged
to32x32, both centered on the RGB-proposed point. Concatenate6channels, edge-pad
at image boundaries. No absolute coordinates, map, teacher, phase-pair, prior
state or source identity enters the network. Both patches come from the SAME
current RGB image. This is an RGB preprocessing change within allowed inputs.

TRAIN positives: every frozen proposal inside a supported housing, labelled
with that housing's phase, plus up to4deterministically spaced audited active-bulb
pixels per housing/phase. Deduplicate identicalimage/point/label; ambiguous
overlapping labels are unknown. TRAIN negatives: at most16proposed points per
image in known-background mask: fourhighestconfidence plus12seededuniform
remaining points. Ignore all other projected light regions, do not assume they
are background. Log total proposals, retained/unselected/unknown and perclass/
location counts. Original876TRAINimages remain the data denominator.

One fresh verifier(seed42): Conv3x3channels16/32/64/64,strides1/2/2/2,SiLU;
flatten4x4x64→Linear64→SiLU→Linear4(background/red/yellow/green).
Batch64,16uniformexamplesperclass,AdamWlr.0003/wd.0001,clip1,BF16forward/FP32CE,
1000updates; no phase-changing augmentation. Everycheckpoint0/100/250/500/1000
retainsmodel/optimizer/RNG. No tuning sweep or extra updates.

At inference score every proposal with RGBcontext. Retain a point iff its
maximumforeground softmaxprobability≥.5; state is its foregroundargmax and
confidence is verifier probability. Same point-in-housing score-order one-to-one
matching and72knownlabels, known-background peak count and ignored-regionunknowns.
Report proposal ceiling independently. This is not bounding-boxAP or driving.

Prospective selection: candidate must preserve parentjoint44 and eachtown
7/15/12/10 and eachstate22red/15yellow/7green. Among protected candidates,
minimize known-backgroundpeaks, then maximizejoint, earlier tie. If none,
select no verifier; parent itself remains unqualified. Component pass also needs
≥90%backgroundpeak reduction (≤3724), with all proposals scored and every
coverage protection. Report all checkpoints and exact original baseline.
No automatic driving pilot or claim of rule applicability/stop-release memory.

Cache/miningcap600s,fit1200s,total2400s,newartifacts250MB; originalsharedstorage
guardian/STOP/baseline. Persist TRAINpatchcache/proposals/metadata/hashes; temporary
VALpatches are reproducible from originalRGB/frozencoordinates and hashedrecipe.
Measure full three-view proposal+crop+verifier latency separately before adoption;
candidatecount controls runtime. No M4primary/reward/benchmark/M3/M6suite change.
