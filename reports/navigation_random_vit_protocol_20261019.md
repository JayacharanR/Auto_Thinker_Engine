# Same-architecture random ViT diagnostic — before training

Additional to the completed66 main cells, not a substitute for their four strong
baselines. Freeze one ViT-S/16 with the identical own-JEPA architecture, explicit
4x8x8→4x9x16 positions,8past frames/three256x144 views/spatial4x4 tokens, but fresh
random weights initialized once with42, no pretraining or local labels. Freeze it
and cache features with the same BF16-forward/FP16 export/runtime quantization.
Registry kind jepa_random identifies this honestly; never call it pretrained.

Fresh same visual-decision head,100%/16639 labels/seed42, same2000 whole-episode
draws, loss, optimizer, precision, validation0/500/1000/1500/2000 and selector as
the main matrix. Invoke the unchanged train_navigation_visual_student.train
function through a diagnostic-only wrapper to avoid editing its training loop.
The wrapper/source and random checkpoint recipe are separately archived. Shared
encoder/runtime/cache source changes only add the random-kind branch; all prior
source snapshots/66 trained models remain unchanged and no study is restarted.

Compare exact2000 sampler draw sequence and common initialized head. Report its
offline result beside comma/CARLA-JEPA100%/seed42, with pretraining costs and prior
development history. This single-seed diagnostic is not a label-efficiency curve,
new native driving candidate or final qualification. No student test-town or
heldout_dev selection, new annotations, perception inheritance or reward change.
Cache resource/STOP checks remain mandatory; serialize after native setup probes.
