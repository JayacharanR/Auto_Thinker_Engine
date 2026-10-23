# First factual dynamics fit — registered before updates

This is a diagnostic training stage, not M5 planning qualification. Use the exact
69-source corrected factual manifest and3800windows:2845TRAIN/955whole-location
validation, with all original camera validation, heldout_dev, Town05 and the480
branch-pilot timestamps excluded from fitting/selection. The clean CARLA-JEPA
encoder has seen training-town video, including dynamics-validation video; disclose
this instead of claiming unseen encoder locations. Freeze the encoder and use
only its past8-frame three-camera48×384 token grid,35declared navigation/ego
values and80ordered20Hz executed controls as model inputs. Future grids, outcomes
and masks are training/scoring targets only.

Train ordered-JEPA and the small continuous RSSM,3independent members each:
seeds42,1051,2060. Both use hidden128,8half-second horizons,AdamW lr3e-4,
weight decay.01,gradient norm1,BF16forward andFP32loss,2000updates/member,
batch16. Each member bootstraps complete TRAIN episodes with replacement, stratified by
TRAIN contact-positive versus remaining episode status, preserving the number
of episodes in each stratum, and
uses the same actual batch-index sequence in both model arms. Per batch,8anchors
are episode-uniform within that bootstrap,8are uniformly sampled from its windows
with a known future collision; fall back to episode-uniform only if no positive
window exists. The oversampling and class counts are disclosed; validation is
never reweighted or resampled. No sample/seed/budget chosen from driving outcomes.

Use the existing masked smooth-L1 future-token and progress(/20m) objective plus
collision BCE. The existing independently masked red term is disabled for this
first fit because these sources contain zero red-positive horizons. Freeze that
output row's gradient and report it as unqualified; do not deploy its random or
all-negative red score. RSSM adds.01posterior-KL with1free nat, future targets
entering that loss only; training transitions sample their prior Gaussian; forward deployment/validation
uses the deterministic prior mean.
Future supervised grids are loss-only; current state never uses them. Masks
remove unknown actuator tails and assisted counterfactual safety claims, while
retaining observed positive collisions and actual factual progress.

Evaluate at0,500,1000,1500,2000 on every validation window. Select a member by the
lowest common masked prediction objective (excluding the RSSM-specific KL);
no driving model selection. Save complete optimizer/RNG/sampler/bootstrapped
population/request/source identity and atomic checkpoints every100updates and
at validation. Fixed two-hour member cap and25GBdisk floor. Report unweighted
collision Brier/BCE, positive/negative horizon counts, missed hazards@.5,
calibration bins, progress MAE by horizon and latent error. Ensemble mean and
between-member spread are measured; spread is not automatically calibrated
uncertainty. No independence assumption from overlapping windows.

Neither lower latent loss nor a factual validation score permits M5 promotion.
Next require physically replay-comparable alternative actions with reactive
traffic, red-positive training/calibration sources, common candidate/controller/
perception costs, and paired development safety/completion evidence against CV.
The first stationary-lead pilot failed elapsed-phase replay tolerance and cannot
serve as a passed deterministic calibration gate. Keep every outcome/contact.

Pre-fit sampler audit: ordinary seed42 whole-episode bootstrap omitted all three
TRAIN contact-positive episodes, yielding zero positive windows. Register the
stratified bootstrap before any optimizer update; seeds and budgets unchanged.
The corpus has108TRAIN positive horizons in24windows/3episodes and49validation
positive horizons in10windows/1episode. Treat effective hazard evidence as sparse
and clustered; no thousands-of-independent-positive-examples claim.
