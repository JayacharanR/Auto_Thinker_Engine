"""
CARLA environment wrapper with Gymnasium interface.

Wraps CarDreamer's CARLA environment to provide a standard Gymnasium
API for DreamerV3 integration. Handles CARLA server lifecycle,
observation preprocessing, and action space configuration.

This wrapper is designed to work with CarDreamer's Observer-Handler
for the front-facing camera setup at 128x128 resolution.
"""

import subprocess
import time
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import yaml


class CarlaEnvWrapper(gym.Env):
    """
    Gymnasium wrapper around CarDreamer's CARLA environment.

    Manages:
    - CARLA server lifecycle (launch/connect/cleanup)
    - Observation space: single front-facing camera image
    - Action space: continuous [steer, throttle_brake]
    - Reward computation via the shared RewardFunction
    - Episode management with max step limits

    Args:
        config: Environment section of the YAML config.
        reward_fn: RewardFunction instance (shared across phases).
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        config: dict[str, Any],
        reward_fn: Any = None,
    ):
        super().__init__()

        self.config = config
        self.reward_fn = reward_fn

        # CARLA connection config
        carla_cfg = config.get("carla", {})
        self.carla_host = carla_cfg.get("host", "localhost")
        self.carla_port = carla_cfg.get("port", 2000)
        self.carla_timeout = carla_cfg.get("timeout", 30.0)
        self.synchronous_mode = carla_cfg.get("synchronous_mode", True)
        self.fixed_delta = carla_cfg.get("fixed_delta_seconds", 0.05)

        # Observation config
        obs_cfg = config.get("observation", {})
        cam_cfg = obs_cfg.get("camera", {})
        self.img_width = cam_cfg.get("width", 128)
        self.img_height = cam_cfg.get("height", 128)
        self.camera_fov = cam_cfg.get("fov", 90)

        # Task config
        self.task = config.get("task", "right_turn")
        self.max_episode_steps = config.get("max_episode_steps", 1000)
        self.num_eval_episodes = config.get("num_eval_episodes", 20)

        # Define spaces
        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(3, self.img_height, self.img_width),
                    dtype=np.uint8,
                ),
            }
        )

        # Continuous action space: [steer, throttle_brake]
        action_cfg = config.get("action", {}).get("space", {})
        steer_range = action_cfg.get("steer_range", [-1.0, 1.0])
        throttle_range = action_cfg.get("throttle_brake_range", [-1.0, 1.0])

        self.action_space = gym.spaces.Box(
            low=np.array([steer_range[0], throttle_range[0]], dtype=np.float32),
            high=np.array([steer_range[1], throttle_range[1]], dtype=np.float32),
            dtype=np.float32,
        )

        # CARLA objects (initialized on connect)
        self._client = None
        self._world = None
        self._vehicle = None
        self._camera_sensor = None
        self._collision_sensor = None
        self._latest_image = None
        self._collision_event = None
        self._step_count = 0
        self._episode_count = 0

    def connect(self) -> None:
        """
        Connect to CARLA server and configure world settings.

        This must be called before reset(). On the target hardware,
        the CARLA server should already be running.
        """
        import carla

        self._client = carla.Client(self.carla_host, self.carla_port)
        self._client.set_timeout(self.carla_timeout)

        self._world = self._client.get_world()

        # Configure world settings
        settings = self._world.get_settings()
        settings.synchronous_mode = self.synchronous_mode
        settings.fixed_delta_seconds = self.fixed_delta
        settings.no_rendering_mode = self.config.get("carla", {}).get(
            "no_rendering", False
        )
        self._world.apply_settings(settings)

        # Set quality level
        quality = self.config.get("carla", {}).get("quality_level", "Low")
        if quality == "Low":
            self._world.unload_map_layer(carla.MapLayer.Buildings)
            self._world.unload_map_layer(carla.MapLayer.Decals)
            self._world.unload_map_layer(carla.MapLayer.Foliage)
            self._world.unload_map_layer(carla.MapLayer.ParkedVehicles)
            self._world.unload_map_layer(carla.MapLayer.StreetLights)

    def _spawn_vehicle(self) -> None:
        """Spawn ego vehicle at a task-appropriate spawn point."""
        import carla

        blueprint_library = self._world.get_blueprint_library()
        vehicle_bp = blueprint_library.find("vehicle.tesla.model3")

        # Get spawn points and select based on task
        spawn_points = self._world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available in CARLA map.")

        # Use first spawn point (task-specific selection can be added)
        spawn_point = spawn_points[0]
        self._vehicle = self._world.spawn_actor(vehicle_bp, spawn_point)

    def _setup_sensors(self) -> None:
        """Attach camera and collision sensors to ego vehicle."""
        import carla

        blueprint_library = self._world.get_blueprint_library()

        # Front-facing RGB camera
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", str(self.img_width))
        camera_bp.set_attribute("image_size_y", str(self.img_height))
        camera_bp.set_attribute("fov", str(self.camera_fov))

        # Mount on vehicle hood
        camera_transform = carla.Transform(
            carla.Location(x=1.5, z=2.4),
            carla.Rotation(pitch=-15),
        )
        self._camera_sensor = self._world.spawn_actor(
            camera_bp, camera_transform, attach_to=self._vehicle
        )
        self._camera_sensor.listen(self._on_camera_image)

        # Collision sensor
        collision_bp = blueprint_library.find("sensor.other.collision")
        self._collision_sensor = self._world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self._vehicle,
        )
        self._collision_sensor.listen(self._on_collision)

    def _on_camera_image(self, image) -> None:
        """Callback for camera sensor data."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))  # BGRA
        # Convert BGRA to RGB and transpose to CHW
        self._latest_image = array[:, :, :3][:, :, ::-1].transpose(2, 0, 1).copy()

    def _on_collision(self, event) -> None:
        """Callback for collision events."""
        self._collision_event = event

    def _get_observation(self) -> dict[str, np.ndarray]:
        """Construct observation dict from sensor data."""
        if self._latest_image is None:
            # Return zeros if no image received yet (first tick)
            image = np.zeros(
                (3, self.img_height, self.img_width), dtype=np.uint8
            )
        else:
            image = self._latest_image

        return {"image": image}

    def _get_reward_obs(self) -> dict[str, Any]:
        """
        Extract reward-relevant observations from CARLA state.

        Returns dict matching the RewardFunction.compute() expected keys.
        """
        import carla

        obs: dict[str, Any] = {}

        if self._vehicle is None:
            return obs

        # Collision
        obs["collision"] = self._collision_event is not None

        # Velocity
        vel = self._vehicle.get_velocity()
        obs["velocity"] = np.array([vel.x, vel.y, vel.z])

        # Lane keeping — get lateral deviation from waypoint
        vehicle_location = self._vehicle.get_location()
        waypoint = self._world.get_map().get_waypoint(
            vehicle_location, project_to_road=True
        )
        if waypoint is not None:
            wp_loc = waypoint.transform.location
            obs["lateral_deviation"] = vehicle_location.distance(wp_loc)
        else:
            obs["lateral_deviation"] = 0.0

        # Progress — approximate by speed along route direction
        speed = np.linalg.norm(obs["velocity"])
        obs["route_progress"] = speed * self.fixed_delta  # distance this step

        # Time-to-collision — simplified using nearby vehicles
        obs["distance_to_leading"] = self._get_distance_to_leading()
        obs["relative_velocity"] = self._get_relative_velocity_to_leading()

        # Traffic signals
        obs["traffic_light_state"] = self._get_traffic_light_state()
        obs["at_stop_sign"] = False  # Extended in task-specific subclass
        obs["stop_sign_wait_time"] = 0.0

        return obs

    def _get_distance_to_leading(self) -> float:
        """Get distance to the nearest vehicle ahead."""
        if self._vehicle is None or self._world is None:
            return float("inf")

        ego_location = self._vehicle.get_location()
        ego_forward = self._vehicle.get_transform().get_forward_vector()

        min_distance = float("inf")
        for actor in self._world.get_actors().filter("vehicle.*"):
            if actor.id == self._vehicle.id:
                continue

            other_location = actor.get_location()
            direction = other_location - ego_location
            distance = ego_location.distance(other_location)

            # Check if vehicle is roughly ahead (dot product with forward vector)
            dot = (
                direction.x * ego_forward.x
                + direction.y * ego_forward.y
                + direction.z * ego_forward.z
            )
            if dot > 0 and distance < min_distance:
                min_distance = distance

        return min_distance

    def _get_relative_velocity_to_leading(self) -> float:
        """Get closing speed to nearest vehicle ahead."""
        if self._vehicle is None:
            return 0.0

        ego_vel = self._vehicle.get_velocity()
        ego_speed = np.sqrt(ego_vel.x**2 + ego_vel.y**2)

        # Simplified: assume leading vehicle is slower
        # Full implementation would track the specific leading vehicle
        return max(0.0, ego_speed)

    def _get_traffic_light_state(self) -> str:
        """Get the state of the traffic light affecting ego vehicle."""
        if self._vehicle is None:
            return "none"

        try:
            import carla

            if self._vehicle.is_at_traffic_light():
                traffic_light = self._vehicle.get_traffic_light()
                if traffic_light is not None:
                    state = traffic_light.get_state()
                    if state == carla.TrafficLightState.Red:
                        return "red"
                    elif state == carla.TrafficLightState.Yellow:
                        return "yellow"
                    elif state == carla.TrafficLightState.Green:
                        return "green"
        except Exception:
            pass
        return "none"

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        Reset environment for a new episode.

        Args:
            seed: Optional seed (used by Gymnasium API).
            options: Optional reset options.

        Returns:
            Tuple of (observation, info).
        """
        super().reset(seed=seed)

        # Clean up previous episode
        self._cleanup_actors()

        # Spawn new vehicle and sensors
        self._spawn_vehicle()
        self._setup_sensors()

        # Reset state
        self._collision_event = None
        self._latest_image = None
        self._step_count = 0
        self._episode_count += 1

        # Tick world to get initial sensor data
        self._world.tick()
        time.sleep(0.1)  # Allow sensor callbacks to fire

        obs = self._get_observation()
        info = {"episode": self._episode_count}

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """
        Take one step in the environment.

        Args:
            action: Array of [steer, throttle_brake].

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        import carla

        # Apply action
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle_brake = float(np.clip(action[1], -1.0, 1.0))

        control = carla.VehicleControl()
        control.steer = steer
        if throttle_brake >= 0:
            control.throttle = throttle_brake
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = -throttle_brake

        self._vehicle.apply_control(control)

        # Tick simulation
        self._world.tick()
        self._step_count += 1

        # Get observation
        obs = self._get_observation()

        # Compute reward
        reward_obs = self._get_reward_obs()
        if self.reward_fn is not None:
            reward_result = self.reward_fn.compute(reward_obs)
            reward = reward_result.total
            terminated = reward_result.terminal
            info = reward_result.to_log_dict()
        else:
            reward = 0.0
            terminated = self._collision_event is not None
            info = {}

        # Check truncation (max steps)
        truncated = self._step_count >= self.max_episode_steps

        # Reset collision event for next step
        self._collision_event = None

        info["step"] = self._step_count
        info["episode"] = self._episode_count

        return obs, reward, terminated, truncated, info

    def _cleanup_actors(self) -> None:
        """Destroy all spawned actors."""
        actors = [self._camera_sensor, self._collision_sensor, self._vehicle]
        for actor in actors:
            if actor is not None:
                try:
                    actor.destroy()
                except Exception:
                    pass
        self._camera_sensor = None
        self._collision_sensor = None
        self._vehicle = None

    def close(self) -> None:
        """Clean up environment."""
        self._cleanup_actors()

        # Reset world settings
        if self._world is not None:
            try:
                settings = self._world.get_settings()
                settings.synchronous_mode = False
                self._world.apply_settings(settings)
            except Exception:
                pass

    def render(self) -> Optional[np.ndarray]:
        """Return current camera frame for visualization."""
        if self._latest_image is not None:
            # Convert CHW to HWC for rendering
            return self._latest_image.transpose(1, 2, 0)
        return None


def create_carla_env(config_path: str) -> CarlaEnvWrapper:
    """
    Create a CarlaEnvWrapper from a YAML config file.

    Args:
        config_path: Path to YAML config containing 'environment' and 'reward' sections.

    Returns:
        Configured CarlaEnvWrapper instance (call .connect() before use).
    """
    from src.envs.reward import RewardFunction

    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)

    env_config = full_config["environment"]
    reward_config = full_config.get("reward", {})
    reward_fn = RewardFunction(reward_config) if reward_config else None

    return CarlaEnvWrapper(config=env_config, reward_fn=reward_fn)
