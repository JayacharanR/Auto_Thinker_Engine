# TRAIN-only actuator timing diagnosis

The first ego model remains preserved and unqualified (4 s validation position
MAE2.495 m/p9510.789 m; longitudinal bias reaches its bound). Before proposing a
replacement, inspect only the52registered dynamics-TRAIN sources. Never open
the17dynamics-VAL sources or choose a lag from their outcomes.

Keep the official native control acknowledgment at f+1 unchanged. Compare whether
the measured transition f→f+1 responds immediately or to a previous acknowledged
command. Diagnose causal lags0…8ticks (0…400 ms), with identical eligible samples
for all lags. Fit the same four bounded longitudinal constants as the original:
throttle[0,20], brake[0,30], drag[0,5], bias[-2,2]. Five leave-one-training-town-out
folds report predictive acceleration error; this is internal TRAIN diagnosis.

Exclude the first28ticks, speeds<=.5 m/s, unsupported actuators and the original
contact neighborhood only for system identification. These exclusions never
change driving/collision counts. Also compare position-derived displacement speed
with snapshot velocity, report acceleration outliers and control regime bins.
Save exact sources/hashes and every lag, including failures. No fitted lag is
automatically deployed, no model/benchmark/source-registered fit is modified, and
no factual residual alone establishes counterfactual candidate accuracy.
