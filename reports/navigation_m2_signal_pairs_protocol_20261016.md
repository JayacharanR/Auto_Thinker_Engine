# Static signal supervision pilot — frozen before collection

The selected camera model barely responds to clear red/green bulb changes in
three familiar TRAIN poses. Collect broader, localized phase supervision before
another camera fine-tune. This is static training data, not a driving benchmark.

Frozen plan: `navigation_m2_signal_pairs_plan_v2_20261016.json`, SHA256
`d890973176b5d0fd24754cc04eee0bfd39327adcc327944edd1af500a1099b42`.
Original guard/internal-manifest hashes are referenced unchanged. The discarded
empty first plan remains preserved; no images came from it. Its waypoint-ID
boundary bug was fixed by projecting line centers onto approach segments.

Selection uses no driving/image outcomes: signal_stop before memory, route ID
order, first four distinct eligible junctions per town. Existing235-route geometry
eligibility and all original internal-validation/heldout_dev exclusions apply.
Three poses at roughly40/20/6m before the evaluator line; deduplicate start clamps.
Result:51 poses,17 junctions:12 poses each in Town01–04,3 in Town10HD. Town05 unused.
The stopped vehicle is placed at a declared road pose plus0.35m; original camera
mount/calibration unchanged. No controls, expert action, or future-speed targets.

One owned server at a time; root wall cap900s, STOP propagation and bounded cleanup.
Capture native512×288 RGB in front/left/right. Freeze light timing, randomize
background light states from a fixed seed per pose, and intervene only on the
applicable physical lamp: red/green/yellow/red-repeat. Verify every other lamp
state stays fixed, all target phases occurred, ego drift<=1mm and camera-matrix
error<=1e-3. Drain exact requested RGB frame IDs after two extra rendering ticks.
Preserve all failed attempts and raw housings, including back-facing/weak crops.

Static phase-response heuristic, chosen using prior TRAIN probes: mean absolute
RGB red/green difference>=2 intensity units and>=5*max(repeated-red difference,.25);
red,green,repeated-red channel dominance>=30. This is a conservative training-label
candidate, not a certified visibility measure. Yellow labels additionally need
max(min(R,G)-B)>=30 inside that housing. Unknown crops stay unlabelled. Dynamic
traffic is absent, so no label certifies original episode visibility. Every label
must count toward later supervised-label budgets; never call it free unlabeled JEPA data.

Data gate: all51 frozen cases and612 native PNGs decode/verify; sources, plan,
map, pose, frame and physical light/stop-lane identity match; phase-responsive
crops span all five towns and at least10 distinct junctions. If the gate fails,
preserve and diagnose it; do not relax it to manufacture a pass. Frames/crops
are correlated. No learned model, driving success or rule-understanding claim.

The original student, controller, cache, split, loss, reward and benchmark stay
unchanged. No multi-hour run or push is authorized by this pilot.
