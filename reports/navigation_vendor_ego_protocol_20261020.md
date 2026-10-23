# Frozen expert ego physics comparison

The pinned PDM-Lite reference already contains a polynomial ego speed model and
slip-angle bicycle geometry. Before fitting another bespoke model, score that
exact implementation/coefficients from vendor revision48e6a6d. Compare its native
zero-delay form and a2tick delayed form, selected from the separate TRAIN-only
timing diagnosis before this evaluation. Preserve both, even if one is worse.

Use exactly the original69dynamics sources,52TRAIN/17VAL, all4second80control
windows at0.5second stride. Current ego pose/odometry and two previous acknowledged
commands initialize the delayed form; all later states are propagated internally.
The native recording's uniquely established startup offset supplies those two
past commands, never a guessed prefix or a future state. All three predictors
use identical window denominators and actual ordered controls. Future positions
and velocities are scoring-only. Keep contact/assisted windows and disclose
fractional brakes (the vendor treats any positive brake as full braking).

No constants are fitted, no lag chosen from validation, and no alternative window
or outcome is excluded after scoring. Record vendor source/config hashes and
per-window error arrays. Factual error does not establish reactive counterfactual
calibration, a collision-risk guarantee, or a planner gate. Existing registered
JEPA/RSSM fits and their sources stay unchanged.
