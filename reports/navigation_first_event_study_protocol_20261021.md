# Scripted first-contact study — prospective engineering generation

This is a new exploratory M5 engineering study, not the M5 qualification suite,
M4 encoder comparison or driving benchmark. Use exactly68verified recordings
(50TRAIN/18calibration, eight/two original locations), one observation at branch
frame per episode. No new labels/simulator cases or failed-case retries. Retain
all73attempts/five unknowns in corpus reporting. Historical frozen JEPA exposure,
scripted actors, frozen-green lights and rear camera blind area are disclosed.
Calibration has four first contacts across two familiar locations: no independent
safety confidence interval or general rule claim.

Inputs: causal48x384camera grid; complete eight-timestamp shared FPN values and
presence; declared ego/navigation and exactlytwoactualowncommands; proposed80
commands. Town/route/mechanism/role/case/event/censoring/actor/world/teacher fields
are never model inputs. Future RGB embeddings are regression targets only.
Same shared detector history and approved ego/action geometry available to CV,
RSSM and JEPA. CV ignores latent embeddings by design; its handcrafted ground
tracking is explicitly reported, and every unlocated detection stays counted.

Baselines frozen before outcomes: TRAIN-MLEconstant20Hzhazard; L2logistic action-
only endpoint model (mean steer/throttle/brake and initial speed, C=1, no class
balancing); original camera tracker with original2m cross-view merge, original
flat-ground/sizes/uncertainties and frozen2tickego forecast, evaluated at20Hz.
CV raw proximity is a score, not a probability. Fit only its monotone sigmoid
scale/intercept on TRAIN eight-horizon binary outcomes (no class balancing,
1e-3slopeL2, slope0..20, intercept-20..20), then freeze before calibration targets.
Report both raw and calibrated. Unknown ground contacts remain disclosed.

Learned arms: three independently initialized JEPA and RSSM members each, seeds
42/1051/2060. Identical50TRAINexamples/uniformbatch12draws,400AdamWupdates,
lr3e-4,weightdecay1e-4,gradientclip1, BF16 forward/FP32 loss, hidden96.
Same spatial residual decoder/risk readout and shared object-history encoder.
RSSM adds Gaussian recurrent prior and training-only posterior regularizer.
Causal per20Hzhazard cannot depend on later commands. Predicted grid carries
hazard gradients; seven0.5sfuture latent targets, censored at first contact or
intervention; eighth target unknown. No positive weighting/oversampling.

Objective: proper censored NLL, SUMobservedloglikelihood per sequence then average;
plus1xmaskedSmoothL1latentloss; RSSMonly .01xmaskedposteriorKLwith1free nat.
Hazard output initially has zero final weights and TRAIN-MLElogitbias; latent
objective may still learn. Evaluate at0/100/200/300/400steps. Select lowest
calibration NLL; ties choose earlier, preserve latest and selected. All updates
and evaluations fixed; no extension or posthoc threshold changes. Keep untrained
selected0 if every update loses. Primary result is ensemble of selected members,
with all individual and latest results reported. Three members are not three
independent calibration locations.

No prediction is safety-qualified. Before any native planner use, the ensemble
must beat both constant and calibrated CV on mean endpoint Brier and average
sequence NLL where meaningful, and must not increase .5threshold misses or safe-
case false alarms relative toCV. Report .5threshold confusion, each location,
each mechanism, and within-context action ranking; no dropping rear contacts.
Action-only baseline is a simplicity check; beating it remains required before
claiming learned visual understanding. Full15candidate predictions are recorded,
but unexecuted alternatives have UNKNOWN outcomes. Only observed options can
supply ranking evidence. Any reactive/red/300encounter gates remain OPEN.

Per-member hard600swallcap, totalstudy3,900s/500MBartifactcap under original14GB/
10GBguardian; stop through root/STOP. No automatic deployment/native evaluation,
no finalTown05 run. Source/manifest/feature hashes and RNG/optimizer/scaler state
are retained. Existing inference packages/drivers remain unchanged.
