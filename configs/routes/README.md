# Development route manifests

These are M0 infrastructure missions, not a frozen evaluation benchmark.
Town05 has not been used. Failed generation attempts remain as evidence.

| Route | Use |
|---|---|
| `navigation_m0_town03_opt_v2` | Default; completed 1.23km expert integration drive |
| `navigation_m0_town01_v3`, `town02_v3`, `town04_v3` | Serialized-input generation; 60-frame map/camera probes passed |
| `navigation_m0_town10hd_v3` | Serialized-input generation; 60-frame probe passed |
| `navigation_m0_town03`, `navigation_m0_town03_opt` | Deprecated: intermediate-keypoint re-interpolation changed route length |
| `navigation_m0_town10hd_probe` | Deprecated: snapped endpoint export changed 389m to 161m; setup rejected |

The generator selects the first seeded pair meeting the requested length bounds,
without considering expert or student driving outcomes. Each XML has a JSON hash
manifest. The runtime checks effective route length before driving. Never replace
an expert failure with an easier route; generation corrections are infrastructure
changes and remain documented with the original attempt.
