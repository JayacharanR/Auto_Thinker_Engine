# Targeted expert corrections — collection protocol, 16 October 2026

Continue M2 from the failed camera baselines. This is **training data collection
with privileged assistance**, not an evaluation of the camera model. No benchmark,
reward, frozen route or input contract changes. No multi-hour run is authorized.

The source student is the unpromoted stop/go best1500 checkpoint (SHA256
`80bee4e338eec8f55f57d96ad4cf0bf27c736e47bc641e2b1dd4e8bb0cfdc3e4`).
It runs in the existing isolated inference process with exactly the declared
three RGB views, map route and ego motion. Oracle imports and extra fields remain
blocked. Teacher state/actions never enter its packet.

The privileged reference runs alongside it in the collector and labels the
current observed state. Normally the student acts. Teacher braking/hazards or a
zero teacher target select the teacher command. After4s stationary while the
teacher would go, select the teacher for3s; new hazards cancel recovery. Every
teacher-selected tick is marked, even if its command equals the student's.
This is a collection behavior policy, not a safety guarantee or a forced-crawl
evaluation fix. The teacher can fail; every recorded collision remains reported.

Store student-requested, teacher, selected and next-tick observed controls
separately, using the existing lossless atomic shards. `trace.jsonl` retains the
expert proposal; `dagger_trace.jsonl` records20Hz mixing; shard labels contain
10Hz feedback-confirmed execution. Videos say privileged assistance is ON.
Assisted episodes are censored and never clean imitation demonstrations. A
separate reader exposes only instantaneous expert corrections after durable
rule-safe outcome, setup and actuator audits. No student/executed action is a
training target. No future speed/dynamics target is inferred across the mixture.

Keep the current internal validation junctions reserved as well as heldout_dev
and Town05. [Frozen guard](navigation_m2_dagger_guard_20261016.json), SHA256
`ae79acb29cb658544b9455f4fa19863170a3e677f95c4a6d96e02664c71d62f6`,
derives from the unchanged raw-cache manifest and approved location assignments.
Exclude every route within1m of reserved junction geometry and every route with
a reserved crossed junction.235 routes remain globally eligible. Runtime checks
the actual trajectory before every20Hz policy call and fails closed on violation.

The first pilot uses the first eligible internal-TRAIN Town03 route per family:
86 crossing/yield,101 stopped lead,111 junction,123 release/memory,133 signal stop,
143 stop sign. Fresh seed202610160+route id, one owned CARLA server,900s wall cap,
STOP-file control, fresh output and source archive. These are training locations;
newly collected data may be used, previous diagnostic recordings remain excluded.
Any later repeat on these same cases is training-task monitoring, not held-out
generalization evidence.

Before any learning from these corrections, require full pixel/frame/feedback,
source/checkpoint, scenario and independent spatial audits; no unknown agent/setup
failure; zero unexplained actuator differences. Require eligible correction
labels for both teacher-stop and teacher-go, at least one marked recovery, and
at least30 stationary teacher-go timestamps after5s from at least two episodes.
This is a small data/exposure gate, not a model-performance or statistical safety
threshold. Unsafe, unknown and truncated episodes stay prediction data only.

Record the applicable light's session ID for later localized label work; its
geometry/state comes from the same world snapshot. Visibility remains unknown.
The80m mask is not upgraded to a visibility claim.

Initial implementation verification:244 CPU tests pass,2 skipped. Existing M1
cache builder, cache, selected checkpoints and benchmark criteria are unchanged.
