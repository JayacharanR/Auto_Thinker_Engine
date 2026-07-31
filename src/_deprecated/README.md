# Deprecated Files

These files were part of the original custom Dreamer implementation
that has been replaced by CarDreamer integration.

They are kept for reference (git history) but are no longer imported
by the active codebase.

## Why deprecated

The code review identified that the custom RSSM, actor-critic, and
training loop introduced bugs #2 (no actual CarDreamer), #7 (zero-action
RSSM mismatch), and #8 (train-ratio/autograd issues). Switching to
CarDreamer's tested DreamerV3 eliminates all three simultaneously.

## Files

- `carla_wrapper.py` → replaced by `car_dreamer.create_task()`
- `reward.py` → replaced by CarDreamer's built-in reward functions
- `rssm_wrapper.py` → replaced by CarDreamer's DreamerV3 RSSM
