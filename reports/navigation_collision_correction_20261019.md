# Collision accounting correction —19 October2026

The pinned native evaluator ignores collisions when ego speed is below0.1m/s,
and merges nearby/repeated contacts. Its collision counter does not satisfy the
project's requirement to report every collision without fault filtering. Historical
native score files are retained unchanged. Their collision counts must be labeled
native events, never complete physical contact evidence.

Preserved CARLA binary recorders retain ego-contact samples independently of this
criterion. The new offline reader follows the official0.9.15 recorder format.
Every ordered row from all313 complete recordings agrees exactly with official
`show_recorder_collisions` queries in both hero/any directions, on an owned server
with no policy or mission. One partial expert crash recording remains incomplete;
its recorded prefix cannot certify an unobserved zero.

| Result | Native events | Unfiltered contact starts |
|---|---:|---:|
| New held-out expert, all attempts | 2 | 6 across3 missions |
| First visual1.23km drive | 5 | 13 |
| Spatial/visual six-case pilots | 0 /0 | 0 /0 |

The expert's route170 was native Perfect but has two recorder contacts. Contact-free
rule-compliant success is therefore **293/300**, previously294/300; strict success
is56/300, previously57/300. Two missing scenario setups remain failures. Missions
102/170/177 have contacts; no fault attribution filters them. Camera long-drive
safety remains failed; its arrival, two red and one stop violation are unchanged.

Count unit: unordered ego-involved actor-pair contact starts, merging continuation
only on adjacent recorded frames. Keep every raw sample and official ordered start
as diagnostics. This grouping is disclosed; native DS remains supplementary.
Historical data/results are not overwritten or declared comparable without this
reanalysis. Future scoring must use preserved native recorders and keep partial
or absent recorder evidence uncertified, not zero.

Evidence: [correction JSON](navigation_collision_correction_20261019.json),
outputs/navigation_recorder_contact_reference_20261019/audit.json,
outputs/navigation_recorder_contact_pilots_20261019/audit.json and
outputs/navigation_recorder_rpc_20261019/verification.json/raw official texts.
Official format/query sources are archived with SHA in
outputs/navigation_recorder_spec_20261019/sources.json.

[CARLA0.9.15 native recorder query implementation](https://github.com/carla-simulator/carla/blob/0.9.15/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Recorder/CarlaRecorderQuery.cpp).
