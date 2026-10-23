# Compact local storage plan — 2026-10-20

Status: user approved. Register the14 GB budget now; existing source-registered jobs retain their25 GB checks, then future generations use the approved10 GB minimum. User asks to fit remaining work within available24 GB. Proposed cap14 GB of additional retained artifacts, with at least10 GB actual disk free. No external drive needed if this capacity remains available.

Full native CARLA recordings, all control/world traces, event counts, failures, source hashes and training data stay retained. Existing artifacts stay unchanged. One regenerable feature cache uses bounded RAM/tmpfs.

For future M6, retain full10 fps three-camera showcase videos for routes0/4/8 at seed3031, identically for expert and student, selected before any final driving. All72missions keep native recordings and complete traces. Other missions need no MP4 preview; native evidence remains available for a later bounded re-render. Keep full videos for every short M3 controlled rule trial. This changes recording selection only, with no change to sensor inputs, cadence, behavior, routes, seeds, scoring or evaluation denominators. Implement only in a fresh source generation after the currently registered setup queues.

| Allocation | Additional cap |
|---|---:|
| Remaining training artifacts |1.5 GB|
| M2 development and pilots |2.0 GB|
| M3 rule trials |1.0 GB|
| M5 calibration/corrective data |2.0 GB|
| M6 full native evidence and selected videos |5.0 GB|
| Contingency |2.5 GB|
| Total |14.0 GB|

Measured320clean TRAIN frames from5towns: compact5 fps H.264 is58.4% smaller than the current preview, but that codec change is optional and has not been deployed. The primary saving comes from the fixed showcase subset, which can keep current10 fps quality.

Measured fresh copies on the actual Btrfs filesystem:99.68 MB world trace costs about12.53 MB physical;16.66 MB native recorder costs5.64 MB;159.15 MB video costs159.16 MB. Thus logical trace sizes cannot be added directly to physical estimates. All probe copies round-trip verified; only synthetic copies removed.

Frozen final suite totals123.40 km over expert and student. Scaling one measured1.23 km drive gives approximately3.46 GB for native evidence plus the six selected full-drive videos. This is a storage estimate, not a Town05 result; slower traffic, failures and additional training can consume the margin. Enforce both cumulative cap and actual free space; stop cleanly at either, retain incomplete cases honestly, and never shorten or resample a benchmark to fit.

Evidence: `outputs/navigation_compact_storage_profile_v2_20261020`, `outputs/navigation_physical_storage_profile_20261020`, exact hashes/projection in the adjacent JSON plan. The proposed10 GB minimum would replace the current25 GB value in `configs/navigation.yaml:14` for future source generations only. No active training or registered setup source was changed.
