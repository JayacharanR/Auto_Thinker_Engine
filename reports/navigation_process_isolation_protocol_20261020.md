# Native process isolation instrument v2 — before execution

The earlier two-cycle ClientLease instrument remains FAILED: cycle0records20
stationary frames then engine−11; cycle1crashes during warmup. No clientSIGABRT
was observed there, but dropping one Python Client reference did not establish
engine stability or complete ownership. Do not rerun or replace those old cycles.

New architecture: the supervising process never imports CARLA bindings. It owns
the engine through a direct Popen handle with the unchanged binary/flags. Each
native episode runs in its own Python child. The child connects only after the
recorded owned PID/binary/user/process-group/start-time matches and that process
owns the expected listening port. It releases every Client/world/actor and exits
before normal engine shutdown. Thus no native callback thread can survive into
the next episode's Python process or kill the supervisor. An engine may still
crash; every child/native exit code and missing observation remains visible.

Exactly two NEW stationary instrument cycles, using the same braked40/10/20tick
protocol and existing strict TRAINroute6site as the prior instrument. No model,
driving action study, source replacement, RGBtraining dataset or test-town visit.
Every child returns plain hashed-frame/native-contact/geography receipts only;
all observations are excluded from fitting. This is a changed infrastructure
method, not a retry of an original pilot driving case or a benchmark candidate.

Parent waits for child exit, allows1sof unmeasured teardown cooldown, then
touches the owned case STOP. Only that request permits verified normalSIGTERM
cleanup, at most one normal retry; no SIGKILL. The parent validates UID,command,
group and process creation time before every signal, even with a Popen handle.
An outer STOP is forwarded to the child. It must stop before each measured tick;
after30sgrace normal owned engine shutdown may unblock a stuck RPC. No automatic
case retry. If an owned process remains after normal cleanup, stop the queue.

The instrument tests **process containment**, not simulator safety: publish both
child exit codes, native exit codes and all incomplete evidence. Containment is
verified only if each child exits0, parent never imports bindings, both normal
cleanup handlers finish and the supervisor survives both cycles. Native−11is
still an infrastructure error, never a normal exit or driving success. A positive
containment result does not certify engine stability, replay, policies or M6.

300stotal/20MBartifact cap,120sper child, same14GBnet/10GBminimum originalguardian,
one global navigation owner. Source/config hashes archived before execution.
Record all results; no fitting, source-label change or automatic larger launch.
