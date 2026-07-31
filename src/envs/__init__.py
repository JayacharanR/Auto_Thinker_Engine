"""
CARLA environment interfaces.

The original custom carla_wrapper.py and reward.py have been
deprecated in favor of CarDreamer's built-in task environments.
See src/_deprecated/ for the old implementations.

Current usage:
    import car_dreamer
    task, task_configs = car_dreamer.create_task('carla_right_turn_simple')
"""
