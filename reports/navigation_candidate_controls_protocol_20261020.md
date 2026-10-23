# Shared executable candidate construction

Before any M5 planner drives, retain the registered3path×5speed desired grid,
common tracking controller and costs. Convert each candidate to80ordered20Hz
steer/throttle/brake commands with the same pure exported expert ego model and
TRAIN-selected2tick response delay. Inputs are current declared navigation,
camera-derived path/speed proposals, current ego speed and the two previously
executed ego commands in chronological order. No future simulator state, other
actor state or expert trajectory may enter this construction.

At each synthetic tick, project onto the candidate's fixed route-corrected path,
keep the segment index monotonic, transform the next vertices into the internally
predicted ego frame, and call the unchanged conventional tracker. Each desired
half-second speed is held for ten control ticks. Propagate using the delayed
command queue. Return both the desired proposal and its forecast realization;
do not pretend a desired acceleration bound guarantees exact physical tracking.
CV and learned predictors receive the same80commands. CV geometry and common
comfort/progress terms use the same forecast realization. The actuator commands
remain bounded; future native branch audits must measure actual tracking error.

The prior two commands must be known executed commands. A scripted prefix can
ignore policy proposals, so those proposals cannot substitute for actual motor
history. An eventual worker adapter must declare and verify own-actuator
feedback at handoff; no actor/signal/teacher facts are allowed. Until that is
implemented this primitive is offline-only, not an executable camera planner.

A discovered grid edge case is corrected before any planner trial: measured
speed above9.75m/s could be instantly clamped to8m/s when lower/upper clip bounds
crossed. The8m/s desired cap stays; actual overspeed now decelerates within the
same3.5m/s² desired envelope. All candidates at measured speeds<=8m/s are unchanged.
No route, seed, scoring threshold, reward or trained dynamics source changes.

The frozen ego model still has1.859mmean/6.691mp95 factual4svalidation error and
known fractional-brake/coast approximations. Construction/equation tests alone
do not qualify physical alternatives, calibrated risks or driving improvement.
