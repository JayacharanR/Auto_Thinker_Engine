# TRAIN-only JEPA continuation details — before any continuation update

This specifies the already registered M4 continuation. Start from the exact
comma2k19 context/predictor SHA in navigation_backbones_20261019.json. Preserve
encoder256 positional parameters and interpolate explicit4x8x8→4x9x16 grids;
predictor positional parameters are initialized by the same interpolation.
No action-conditioning branch or labeled training loss is present.

All59 registered TRAIN episodes are eligible; reserve the first floor(29/5)=5
location groups ordered by SHA256 of `20261019:(town,junction_ids)` for unsupervised
diagnostics. Train on the rest. This holdout is inside TRAIN, disjoint by location,
and unrelated to driving validation/heldout_dev/Town05. Save exact episode IDs,
frame counts and source hashes. No driving-label selection or subset search.

An update samples eight episode endpoints uniformly by episode, takes the eight
past10Hz frames, and uses all three256x144 camera views. Each view masks72 of144
spatial cells, with the identical mask at all four temporal tubelets. Context
encoder receives only visible patches. EMA target sees the full past clip;
SmoothL1(beta1) predicts its masked latent tokens. All views are normalized with
the existing common ImageNet mean/std, no image augmentation. Target momentum
is.996; AdamW lr1e-4/weight decay.05,500-step linear warmup then cosine to.1x,
grad cap1, BF16/no scaler. Four6-view microbatches accumulate one24-view update.
Train exactly10000 updates within6h; profile three full gradients with zero
optimizer updates first and retain the profile. Profile is not a downstream trial.

No best-by-moving-target-loss selection: final fixed-budget encoder is the arm.
At0/every1000 updates, repeat16 fixed TRAIN-holdout clips with a separate fixed
RNG20261019 and record latent loss, within-clip variance and between-scene variance
at each corresponding camera. Fail if either variance falls below1% of its
initial value or1e-6; preserve any collapsed run, do not promote it or silently
restart. These checks diagnose collapse; they do not prove useful representations.

Checkpoint full context, teacher, predictor, optimizer and RNG/sampler every250
updates; explicit identical-source resume only. Check STOP/wall/disk on each
microbatch. Preserve a25GB disk reserve; require26GB before starting and before
first checkpoint. Save only one rotating full recovery checkpoint and one final
inference export. Later causal, ordered-action future prediction remains M5.
