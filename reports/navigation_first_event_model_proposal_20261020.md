# First-contact prediction: staged causal countermeasure, no fit

The fixed-plan diagnostic is a failed test of both original learned ensembles.
Do not extend those fits blindly. Both miss all 14 collision-positive horizons
in six observed alternative-action rollouts. JEPA also ranks braking as riskier
than go. CV ranks braking first but is excessively conservative. All are
unqualified; this one TRAIN scenario is not a safety benchmark result.

The subsequent source audit finds only four collision-containing episodes:
two noisy and two expert. Two of three TRAIN collision episodes involve rear
impacts while the ego is nearly stopped. All 144 positive horizons whose input
precedes first contact contain braking; 136 also include commands after first
contact. Another 13 positive horizons start after the episode's first contact.
These are legitimate retained factual contacts, including repeat/ongoing contact.
They are weak coverage for learning which prospective action avoids a first hit.
This observational association does not prove the cause of the inverted scores.

Preserve the old models, datasets, prediction task and every headline contact.
The proposed auxiliary task predicts the first episode contact at 20 Hz. At
observation f, action a_f acts in the next observed physics tick. Score event
hazards using only actions up to that event. Stop the valid loss prefix at first
contact, the first future privileged intervention, unavailable controls or episode
end. Windows already at/after first episode contact do not supply new-first-event
labels; this changes the supervised task, not evaluation collision accounting.
Latent transition/recovery data and original factual labels remain preserved.

The standalone prototype is in
`outputs/navigation_first_event_staging_20261020`. It retains camera/grid identity
and predicts a spatial token grid at every physics tick. The hazard readout uses
that predicted grid, so its gradients reach the token decoder. Shared survival
math turns 80 first-event hazards into eight cumulative collision values. Inputs
are past camera tokens, 80 proposed controls and 41 current navigation/ego/two-prior
actuator values. The latter six values use the separately approved ego feedback.
No future label, event time, simulator traffic state or intervention tag enters
forward. It remains outside the frozen inference package and is not deployed.

Three tests pass: correct event/reaction ordering and censoring; masked unknown
tails cannot affect the objective; and later commands cannot change earlier
hazards, latent predictions or scored loss. Collision gradients reach the spatial
token projection, camera assignment matters, and cumulative risks are monotone.
These construction tests establish no predictive or driving improvement.

Before any fit:

- Collect substantially more independent, bounded action-intervention encounters,
  including front, side and rear contact mechanisms and reactive crossing traffic.
  Use declared TRAIN locations excluding the original camera validation, M3,
  held-out development and Town05; freeze any new internal calibration locations
  before collection and exclude their geography from all new supervised fitting.
- Keep the current six branch rollouts as a retained engineering diagnostic;
  do not silently add them to TRAIN after observing their outcomes.
- Register matched data, sampling, selection and compute for three independent
  JEPA and RSSM members. Their temporal resolution and event semantics must match.
  Give CV the same 20 Hz ego/action geometry and current perception facts.
- Resolve the same-perception comparison: the old FPN CV pipeline and latent-only
  neural pipeline differ. This prototype alone does not fix that confound. Freeze
  a shared camera-perception input pack for all three before claiming M5 evidence.
- Calibrate on separate TRAIN calibration episodes, report hazard misses and
  false braking, and compare constant/always-safe alternatives. No learned red
  costs without independent positive red examples. Do not filter rear impacts.

All current M4 pilots and the queued approved feedback activation stay unchanged.
No new training, reward, benchmark route, score, qualification gate or final
Town05 evaluation is authorized by these structural tests alone.

A further centre-only camera audit identifies an observability limit: in the two
rear-contact TRAIN episodes, the other vehicle centre lies outside all three
horizontal camera frusta in 37/64 and 47/64 past-context snapshots, respectively.
Four and five of their eight positive windows have all eight centres outside.
The VAL collision actor is absent from 53/64 full world snapshots in those input
contexts. This does not certify full-object visibility or prove a failure cause.
It reinforces the need for longer causal belief memory and action interventions
with visible actors before the decision. No collision is removed or reclassified.
See `outputs/navigation_action_confound_audit_20261020/centre_frustum_diagnostic.json`.

The next data-engine countermeasure should first profile a small synchronized
TRAIN-only collection with privileged semantic/instance/depth camera labels,
while keeping runtime policy inputs to the declared three RGB views, navigation,
ego motion and approved own-actuator feedback. Such labels can audit object
visibility and metric ground projection and train the shared perception pack.
Use a bounded storage/throughput pilot before scheduling a larger collection;
charge every supervised timestamp to any future label-efficiency comparison.
Do not reinterpret the existing M4 study as containing this untrained component.
