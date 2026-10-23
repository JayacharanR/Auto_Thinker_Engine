"""Predeclared collection-only perturbations, never used in benchmark evaluation."""

import numpy as np


class SteeringPerturbation:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.remaining = 0
        self.offset = 0.0
        self.noise_ticks = self.safety_stop_ticks = 0

    def control(self, teacher, tick, speed, in_junction, hazard):
        steer, throttle, brake = teacher
        if self.remaining and (hazard or brake > 0):
            self.remaining = 0
            self.safety_stop_ticks += 1
            return (steer, 0.0, 1.0), "privileged_safety_stop"
        if in_junction:
            self.remaining = 0
        if tick % 40 == 0 and 0.5 <= speed <= 8 and not in_junction and not hazard and brake == 0:
            self.offset = float(self.rng.uniform(0.04, 0.08) * self.rng.choice([-1, 1]))
            self.remaining = 6  # 0.3s pulse at most once per2s, only on clear approaches
        if self.remaining:
            self.remaining -= 1
            self.noise_ticks += 1
            return (float(np.clip(steer + self.offset, -1, 1)), throttle, brake), "steering_noise"
        return teacher, None
