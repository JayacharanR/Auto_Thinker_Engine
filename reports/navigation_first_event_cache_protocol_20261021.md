# Shared intervention features, frozen before computation

Use only the separate verified v2 factual manifest: 68 eligible recordings from
73 original attempts, 50 TRAIN / 18 calibration. Preserve all five unknowns and
all 16 observed contact starts. Calibration sites are original Town03/1221 and
Town04/916; no calibration target enters fitting. The frozen CARLA JEPA backbone
has its previously disclosed historical corpus exposure; this is not a new clean
encoder benchmark. Original M4 primary and label-efficiency comparison unchanged.

One observation per episode: its predeclared branch frame (physics tick80,
RGB sample40). Use the eight RGB timestamps ending there for shared road-user
history. Frozen FPN weights, original classes/score threshold and complete
navigation-camera-road-users-v1 values/presence are identical and available to
CV/RSSM/JEPA. Keep every raw accepted box, including unlocated observations;
no privileged mask/depth/actor state enters an input. Shared history odometry
contains only original declared ego speed/yaw/yaw-rate. Own-actuator history is
exactly the last two applied ego commands, already independently native-verified.

The frozen in-domain JEPA encoder uses only its causal eight-RGB clips. Cache
current 48x384 tokens plus targets at0.5s intervals. The source does not store the
last4.0s RGB frame: its eighth latent target is UNKNOWN, never copied from3.9s.
Targets at/after first contact or first privileged intervention are masked from
prospective latent regression. Future RGB embeddings are TARGETS only.

Proposed controls are the complete80-command plan saved before executing the
branch, not post-contact reaction commands. First-contact labels are discrete
hazards at20Hz, observed through the contact or censoring point. Contact at
branch+65 appears at zero-based index64; subsequent hazards unknown. The
original native controls and event/contact headlines remain intact. No red-risk
model: all traffic lights were frozen green. Scripted actors are nonreactive;
these data cannot qualify M5 reactive planning or any camera driver.

GPU/native owner lock, STOP-based shutdown, 1,200s cache wall cap and250MB artifact
cap under the SAME approved14GBnet/10GBminimum guardian. Persistent compressed
features avoid a regenerable RAM cache disappearing at the next interruption.
Source/weights/manifest/source-shard hashes archived; read only policy members
for inputs. Parse audited outcomes/plan only in the target assembler. Save
pixel-derived features before attaching labels. No fitting starts automatically.

Later registered models must use a proper censored survival likelihood: SUM
log-likelihood over each observed prefix, then average over windows. Dividing
each window by its variable observed duration overweights early-event examples.
The earlier untrained prototype has that normalization issue; preserve it as
history and fix a new generation before fitting. No prior fit used that code.
