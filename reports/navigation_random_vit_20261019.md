# Same-architecture random ViT diagnostic

One full-label seed42 diagnostic, same2000 exact episode draws and identical initialized driving head as both JEPA arms.

| Frozen encoder | Selected score | Stop recall | False holds | Go MAE |
|---|---:|---:|---:|---:|
| jepa_random | 0.4122 | 62.3% | 27.5% | 4.318 |
| jepa_comma2k19 | 0.3809 | 53.8% | 13.2% | 4.176 |
| jepa_carla | 0.3824 | 56.1% | 17.7% | 3.714 |

Single-seed familiar validation, not a label-efficiency curve or native driving claim. The main66 fits and strong baselines remain unchanged.
