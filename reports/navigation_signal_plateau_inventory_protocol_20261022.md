# One representative per RGB local-maximum plateau — TRAIN inventory

Separate candidate architecture, registered before execution. The all-pixel global
proposal inventory stopped at15millionpoints after2164complete images, before any
classifier inference. Original failure/source archives remain; it is not retried
or relabeled a success. It retained no final coordinates because the cap fired
before serialization; its partial-progress limit is disclosed.

Keep the same native5×5 per-channel local maxima and16/255localcontrast. Instead
of every plateau pixel, use one actual native pixel nearest the centroid of every
eight-connected maximum plateau, separately for each ofR/G/B. Ties use y then x.
Keep every component, merge duplicate coordinates, union ALL previous1,770,643
refined points. No score/top-K/intensity/hue/position filter. This is spatial
consolidation of every qualifying region, not outcome-based pruning. No parameter
search or learned-head change.

Same TRAIN-only inventory measures/gates as the previous protocol: all2,844images,
all1,248housinglabels and59residualmisses, count candidate coverage and exact
recognized-oracle membership (diagnostic only); require positive gain in each,
≤15millionunionpoints and120MB. No classifier predictions/fit/componentVAL/native
driving. Record per-image partial counts incrementally so cap failures retain their
measured growth. Original benchmark/threshold.5/gates are unchanged.

Run prepare_navigation_signal_global_inventory.py --plateau with new output
outputs/navigation_signal_plateau_inventory_20261022.600s cap/120MB, original
storagebudget14GBnet/10GBminimum, STOP/no retries. A distinct output preserves
both the failed dense architecture and this registered consolidated architecture.
