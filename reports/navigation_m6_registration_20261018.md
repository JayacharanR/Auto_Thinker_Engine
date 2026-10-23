# Town05 registration completed —18 October2026

The final suite is frozen; no student was evaluated. This is M6 preparation,
not completion of M2–M6 or evidence of driving performance.

Twelve routes range from1.006 to2.454km. Seeds3031/4031/5031 give36 missions,
84 planned stop opportunities and36 planned held-red opportunities. Every one
of the five explicit mapped stop locations occurs in two required routes. All
33 native runtime stop actors remain scored. Actual opportunities/setup must
still be audited during final driving, including early failures.

The intended pre-student registration deadline was missed. The frozen manifest
discloses prior camera/component development and registration before any new
fit or Town05 student outcome. Town05 geometry is not training/pretraining data.

The endpoint-only v1 attempt is retained, stopped and unfrozen. V2 uses fixed
geometry coverage waypoints and seeded endpoints; no driver outcomes informed
route generation. The independent audit calls the pinned harness interpolation
and verifies serialized lengths, mandatory-stop passage and exact scenario
entry-lane/light bindings for all12 routes. The five XML signal nodes are only
a subset of33 runtime stop actors; physical identities are matched, never
assumed persistent across simulator sessions.

| Route | Length (m) | Planned stop opportunities per seed |
|---|---:|---:|
| 0 | 1662.8 | 3 |
| 1 | 1516.3 | 2 |
| 2 | 2342.0 | 3 |
| 3 | 1283.5 | 3 |
| 4 | 2453.9 | 2 |
| 5 | 1931.0 | 2 |
| 6 | 1794.0 | 4 |
| 7 | 1353.5 | 4 |
| 8 | 1898.3 | 2 |
| 9 | 1888.2 | 2 |
| 10 | 1437.3 | 1 |
| 11 | 1005.5 | 0 |

Registered artifacts:

- configs/routes/town05_final_v2/suite.json and suite.sha256
- Town05_Opt.xml/json plus source archive and registration_request.json
- reports/navigation_m6_registration_20261018.json
- outputs/navigation_final_geometry_20261018: static-only inventory/source

Reproduce the independent CPU audit:

```bash
.venv/bin/python scripts/audit_navigation_final_registration.py
```

Before final driving: finish M2 camera/encounter evidence, M3 override-disabled
rule counterfactuals, M4 fair encoder/label-efficiency study and M5 predictive
planning comparisons; obtain the new heldout_dev expert reference. Register
checkpoint candidates once, count every collision/infraction and do not tune
on these final results.
