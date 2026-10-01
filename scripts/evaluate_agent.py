#!/usr/bin/env python3
"""
Standalone Closed-Loop Evaluator and Telemetry Video Recorder (P0/P1/P5).

Evaluates trained DreamerV3 autonomous driving agents in CARLA 0.9.15,
enforces independent action shielding with SafetySupervisor, generates
comprehensive P0 driving metrics (success, collision, completion %, TTC, latency),
and renders high-definition driving videos with real-time HUD telemetry overlays.

Usage:
    python scripts/evaluate_agent.py \
        --checkpoint outputs/logs/cnn_seed42/latest.pt \
        --task carla_right_turn_simple \
        --episodes 5 \
        --record-video \
        --output-dir outputs/eval_results
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import pathlib
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gym
import numpy as np
import torch

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DREAMER_DIR = PROJECT_ROOT / "third_party" / "dreamerv3_torch"
sys.path.insert(0, str(DREAMER_DIR))

CARDREAMER_DIR = PROJECT_ROOT / "third_party" / "CarDreamer"
sys.path.insert(0, str(CARDREAMER_DIR))

from scripts.train_cardreamer import (
    CarlaImageObservation,
    build_encoder_hook,
    load_dreamer_config,
    load_phase3_config,
    make_carla_env,
    seed_everything,
)
from src.eval.metrics import MetricsTracker
from src.safety.supervisor import SafetySupervisor
from src.utils.manifest import create_run_manifest


def draw_telemetry_hud(
    frame: np.ndarray,
    speed_kmh: float,
    steer: float,
    throttle: float,
    reward: float,
    cumulative_reward: float,
    route_completion: float,
    ttc: Optional[float],
    intervention_info: Optional[Dict[str, Any]],
    step: int,
    episode: int,
    bev_frame: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Render a professional driving telemetry HUD overlay onto a camera frame.

    Args:
        frame: RGB image (H, W, 3) uint8.
        speed_kmh: Vehicle speed in km/h.
        steer: Steering command in [-1, 1].
        throttle: Throttle/brake in [-1, 1].
        reward: Step reward.
        cumulative_reward: Total episode reward so far.
        route_completion: Fraction of route completed [0, 1].
        ttc: Time to collision in seconds (or inf/None).
        intervention_info: Safety supervisor intervention details.
        step: Current episode step.
        episode: Current episode number.
        bev_frame: Optional bird's-eye view waypoint map.

    Returns:
        Annotated BGR image ready for video encoding.
    """
    # Resize frame to standard HUD resolution (512x512)
    target_size = (512, 512)
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    canvas = cv2.resize(bgr, target_size, interpolation=cv2.INTER_LINEAR)
    h, w = target_size

    # Semi-transparent top and bottom telemetry bands
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 55), (20, 20, 25), -1)
    cv2.rectangle(overlay, (0, h - 75), (w, h), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    # Top Header
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "AUTO THINKER ENGINE | CARLA 0.9.15", (15, 22), font, 0.55, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"EPISODE {episode} | STEP {step:04d}", (15, 44), font, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    # Safety Shield Badge
    intervened = intervention_info and intervention_info.get("intervened", False)
    if intervened:
        reasons = ",".join(intervention_info.get("reasons", ["OVERRIDE"]))
        cv2.rectangle(canvas, (w - 240, 8), (w - 10, 48), (0, 0, 180), -1)
        cv2.putText(canvas, "SAFETY INTERVENTION", (w - 230, 25), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, reasons[:26], (w - 230, 42), font, 0.35, (220, 220, 220), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(canvas, (w - 170, 10), (w - 10, 45), (30, 120, 30), -1)
        cv2.putText(canvas, "SHIELD: SAFE", (w - 155, 32), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Top Right Bird's-Eye View (BEV) Mini-Map (Picture-in-Picture)
    if bev_frame is not None and isinstance(bev_frame, np.ndarray) and bev_frame.size > 0:
        try:
            bev_size = 110
            bev_resized = cv2.resize(bev_frame, (bev_size, bev_size), interpolation=cv2.INTER_NEAREST)
            if bev_resized.ndim == 3 and bev_resized.shape[-1] == 3:
                bev_bgr = cv2.cvtColor(bev_resized, cv2.COLOR_RGB2BGR)
            else:
                bev_bgr = cv2.cvtColor(bev_resized, cv2.COLOR_GRAY2BGR)
            bx = w - bev_size - 12
            by = 58
            cv2.rectangle(canvas, (bx - 2, by - 2), (bx + bev_size + 2, by + bev_size + 2), (200, 160, 40), 1)
            canvas[by:by + bev_size, bx:bx + bev_size] = bev_bgr
            cv2.putText(canvas, "BEV MAP", (bx + 4, by + 12), font, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        except Exception:
            pass

    # Bottom Dashboard Telemetry
    # 1. Speed
    speed_color = (100, 230, 100) if speed_kmh > 3.0 else (180, 180, 180)
    cv2.putText(canvas, f"{speed_kmh:4.1f}", (15, h - 35), font, 0.8, speed_color, 2, cv2.LINE_AA)
    cv2.putText(canvas, "KM/H", (15, h - 15), font, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

    # 2. Steering Gauge Bar [-1 .. 0 .. +1]
    bar_x, bar_y, bar_w, bar_h = 110, h - 45, 120, 10
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), 1)
    center_x = bar_x + bar_w // 2
    cv2.line(canvas, (center_x, bar_y - 2), (center_x, bar_y + bar_h + 2), (120, 120, 120), 1)
    steer_pos = int(center_x + (steer * (bar_w // 2)))
    steer_pos = max(bar_x, min(bar_x + bar_w, steer_pos))
    cv2.circle(canvas, (steer_pos, bar_y + bar_h // 2), 5, (0, 220, 255), -1)
    cv2.putText(canvas, f"STEER: {steer:+.2f}", (bar_x, h - 15), font, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    # 3. Throttle/Brake Bar
    th_x = 245
    cv2.rectangle(canvas, (th_x, bar_y), (th_x + 90, bar_y + bar_h), (60, 60, 60), 1)
    if throttle >= 0:
        th_w = int(throttle * 90)
        cv2.rectangle(canvas, (th_x, bar_y), (th_x + th_w, bar_y + bar_h), (80, 200, 80), -1)
        cv2.putText(canvas, f"THROT: {throttle:.2f}", (th_x, h - 15), font, 0.38, (80, 220, 80), 1, cv2.LINE_AA)
    else:
        brk_w = int(abs(throttle) * 90)
        cv2.rectangle(canvas, (th_x, bar_y), (th_x + brk_w, bar_y + bar_h), (50, 50, 220), -1)
        cv2.putText(canvas, f"BRAKE: {abs(throttle):.2f}", (th_x, h - 15), font, 0.38, (80, 80, 240), 1, cv2.LINE_AA)

    # 4. Route Progress Bar
    prog_x = 355
    prog_w = int(np.clip(route_completion, 0.0, 1.0) * 140)
    cv2.rectangle(canvas, (prog_x, bar_y), (prog_x + 140, bar_y + bar_h), (60, 60, 60), 1)
    cv2.rectangle(canvas, (prog_x, bar_y), (prog_x + prog_w, bar_y + bar_h), (220, 160, 40), -1)
    cv2.putText(canvas, f"ROUTE: {route_completion*100:4.1f}%", (prog_x, h - 15), font, 0.38, (220, 180, 60), 1, cv2.LINE_AA)

    # Return & TTC in bottom right
    ttc_str = f"TTC: {ttc:.1f}s" if (ttc and not np.isinf(ttc) and ttc > 0) else "TTC: SAFE"
    cv2.putText(canvas, f"RETURN: {cumulative_reward:+.1f} | {ttc_str}", (prog_x - 10, h - 55), font, 0.38, (180, 180, 180), 1, cv2.LINE_AA)

    return canvas


def load_agent(
    checkpoint_path: str,
    obs_space: gym.spaces.Dict,
    act_space: gym.spaces.Box,
    config: argparse.Namespace,
    device: str,
    arm: str,
    phase3_config: dict,
) -> torch.nn.Module:
    """
    Construct Dreamer agent and load weights from checkpoint.
    """
    import tools
    from dreamer import Dreamer

    # Configure action dimensions for world model & actor critic
    acts = act_space
    config.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
    if getattr(acts, "discrete", False):
        config.actor = dict(config.actor)
        config.actor["dist"] = "onehot"
        config.actor["std"] = "none"

    hook = build_encoder_hook(arm, phase3_config, device)
    logger = tools.Logger(pathlib.Path(config.logdir), 0)

    # For evaluation, we do not require replay training datasets
    agent = Dreamer(
        obs_space,
        act_space,
        config,
        logger,
        dataset=None,
        custom_encoder=hook,
    ).to(device)

    agent.requires_grad_(requires_grad=False)
    agent.eval()

    if os.path.exists(checkpoint_path):
        print(f"[eval] Loading checkpoint weights from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "agent_state_dict" in ckpt:
            agent.load_state_dict(ckpt["agent_state_dict"], strict=False)
            print(f"[eval] Checkpoint loaded (Step {ckpt.get('step', 'unknown')}).")
        else:
            print("[eval] WARNING: 'agent_state_dict' not found in checkpoint.")
    else:
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    return agent


def run_evaluation(
    checkpoint_path: str,
    task: str = "carla_right_turn_simple",
    arm: str = "cnn",
    episodes: int = 5,
    seed: int = 42,
    output_dir: str = "outputs/eval_results",
    record_video: bool = True,
    use_safety_shield: bool = True,
    fps: int = 20,
) -> Dict[str, Any]:
    """
    Execute closed-loop evaluation in CARLA, record video, and compute P0 metrics.
    """
    import tools
    os.environ.setdefault("CARLA_DISABLE_MONITOR", "1")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(seed)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_dir = out_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 65)
    print(f"CARLA AUTONOMOUS DRIVING EVALUATION")
    print(f"Task: {task} | Arm: {arm} | Episodes: {episodes} | Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output Directory: {out_dir}")
    print("=" * 65 + "\n")

    # Load configurations
    phase3_config = load_phase3_config()
    image_size = (64, 64) if arm == "cnn" else (224, 224)
    config = load_dreamer_config(arm, task, seed, steps=1000, image_size=image_size, device=device)
    config.logdir = str(out_dir)

    # Initialize environment
    env = make_carla_env(task, seed=seed, image_size=image_size)

    # Initialize agent
    agent = load_agent(
        checkpoint_path=checkpoint_path,
        obs_space=env.observation_space,
        act_space=env.action_space,
        config=config,
        device=device,
        arm=arm,
        phase3_config=phase3_config,
    )

    # Initialize Safety Supervisor & Metrics Tracker
    supervisor = SafetySupervisor(enabled=use_safety_shield)
    metrics_tracker = MetricsTracker(name=f"{arm}_{task}")

    # Telemetry tracking
    start_eval_time = time.time()
    total_eval_steps = 0
    generated_video_paths = []

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Evaluation loop
    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = False
        step = 0
        ep_return = 0.0
        agent_state = None
        metrics_tracker.start_episode(episode_id=ep)
        supervisor.reset()
        video_frames = []

        print(f"\n[eval] >>> Starting Episode {ep}/{episodes} (Seed {seed + ep - 1})")

        # Initial waypoint count to measure route progress
        initial_waypoints = len(getattr(env.unwrapped, "waypoints", []))

        while not done:
            step += 1
            total_eval_steps += 1

            # Format observation for Dreamer agent (numpy dict with batch dimension)
            agent_obs = {
                k: np.expand_dims(tools.convert(v), 0)
                for k, v in obs.items()
                if "log_" not in k
            }

            # Policy forward pass (greedy evaluation mode)
            with torch.no_grad():
                policy_output, agent_state = agent(
                    agent_obs,
                    reset=np.array([bool(obs["is_first"])]),
                    state=agent_state,
                    training=False,
                )

            # Extract raw action
            raw_action = policy_output["action"][0].detach().cpu().numpy()

            # Collect vehicle telemetry for safety supervisor and HUD
            unwrapped = env.unwrapped
            ego = getattr(unwrapped, "ego", None)
            telemetry = {}
            if hasattr(unwrapped, "planner_stats"):
                telemetry.update(unwrapped.planner_stats)

            # Check TTC, waypoint distance, speed
            ego_location = np.array([0.0, 0.0])
            speed_norm = 0.0
            if ego is not None:
                try:
                    import carla
                    loc = ego.get_location()
                    ego_location = np.array([loc.x, loc.y])
                    vel = ego.get_velocity()
                    speed_norm = float(np.sqrt(vel.x**2 + vel.y**2 + vel.z**2))
                except Exception:
                    pass

            wpt_dis = getattr(unwrapped, "get_wpt_dist", lambda l: 0.0)(ego_location)
            telemetry["wpt_dis"] = wpt_dis
            telemetry["speed_norm"] = speed_norm

            ttc = float("inf")
            try:
                from car_dreamer.toolkit import TTCCalculator
                if hasattr(unwrapped, "_world"):
                    ttc = TTCCalculator.get_ttc(ego, unwrapped._world.carla_world, unwrapped._world.carla_map)
            except Exception:
                pass
            telemetry["ttc"] = ttc

            # Filter action through Safety Supervisor
            safe_action, intervention_info = supervisor.filter_action(raw_action, telemetry)

            # Step CARLA environment (SelectAction wrapper extracts dict['action'])
            next_obs, reward, done, info = env.step({"action": safe_action})
            ep_return += float(reward)

            # Route completion estimation
            cur_waypoints = len(getattr(unwrapped, "waypoints", []))
            if initial_waypoints > 0:
                completed_wpts = max(0, initial_waypoints - cur_waypoints)
                route_completion = completed_wpts / initial_waypoints
            else:
                route_completion = 1.0 if info.get("destination_reached", False) else 0.0

            # Record step metrics
            info_record = {
                "reward/total": reward,
                "reward/collision": 1.0 if info.get("is_collision", False) else 0.0,
                "wpt_dis": wpt_dis,
                "speed_norm": speed_norm,
                "ttc": ttc,
                "safety_intervention": intervention_info["intervened"],
            }
            metrics_tracker.step(info_record)

            # Render HUD video frame
            if record_video:
                raw_frame = obs["image"]
                hud_frame = draw_telemetry_hud(
                    frame=raw_frame,
                    speed_kmh=speed_norm * 3.6,
                    steer=float(safe_action[1]),
                    throttle=float(safe_action[0]),
                    reward=float(reward),
                    cumulative_reward=ep_return,
                    route_completion=route_completion,
                    ttc=ttc,
                    intervention_info=intervention_info,
                    step=step,
                    episode=ep,
                )
                video_frames.append(hud_frame)

            obs = next_obs

        # Episode termination evaluation
        is_success = bool(info.get("destination_reached", False) or route_completion >= 0.90)
        is_collision = bool(info.get("is_collision", False))
        is_out_of_lane = bool(info.get("out_of_lane", False))
        is_time_exceeded = bool(info.get("time_exceeded", False))

        term_reason = "destination_reached" if is_success else (
            "collision" if is_collision else (
                "out_of_lane" if is_out_of_lane else "time_exceeded"
            )
        )

        ep_metrics = metrics_tracker.end_episode(
            success=is_success,
            route_completion=route_completion,
            termination_reason=term_reason,
            collision=is_collision,
            out_of_lane=is_out_of_lane,
            time_exceeded=is_time_exceeded,
        )

        print(
            f"[eval] Episode {ep} finished: {term_reason.upper()} | "
            f"Steps: {step} | Return: {ep_return:.1f} | "
            f"Completion: {route_completion*100:.1f}% | "
            f"Interventions: {ep_metrics.safety_interventions}"
        )

        # Save Episode Video
        if record_video and video_frames:
            video_path = video_dir / f"eval_ep{ep}_{term_reason}.mp4"
            H, W = video_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, float(fps), (W, H))
            for f in video_frames:
                writer.write(f)
            writer.release()
            generated_video_paths.append(str(video_path))
            print(f"[eval] Saved episode video: {video_path} ({os.path.getsize(video_path) / 1024:.1f} KB)")

    # Close environment
    try:
        env.close()
    except Exception:
        pass

    # Aggregate performance metrics
    total_time = time.time() - start_eval_time
    steps_per_sec = total_eval_steps / max(total_time, 1e-3)
    peak_vram = (
        torch.cuda.max_memory_allocated() / (1024 * 1024)
        if torch.cuda.is_available()
        else 0.0
    )

    metrics_tracker.set_system_metrics(
        steps_per_sec=steps_per_sec,
        peak_vram_mb=peak_vram,
        total_duration_sec=total_time,
        total_steps=total_eval_steps,
        videos=generated_video_paths,
    )

    # Save metrics.json
    metrics_json_path = out_dir / "evaluation_metrics.json"
    metrics_tracker.save_json(metrics_json_path)

    # Save run manifest
    manifest_path = out_dir / "manifest.json"
    create_run_manifest(
        config=config.__dict__ if hasattr(config, "__dict__") else {},
        arm=arm,
        seed=seed,
        task=task,
        checkpoint_path=checkpoint_path,
        extra_metadata={"metrics": metrics_tracker.summary()},
        output_path=manifest_path,
    )

    print("\n" + metrics_tracker.generate_summary_markdown())
    print(f"[eval] Evaluation metrics saved to: {metrics_json_path}")
    print(f"[eval] Run manifest saved to: {manifest_path}\n")

    return metrics_tracker.to_dict()


def main():
    parser = argparse.ArgumentParser(description="Evaluate autonomous driving agent in CARLA.")
    parser.add_argument("--checkpoint", type=str, default="outputs/logs/cnn_seed42/latest.pt", help="Path to checkpoint")
    parser.add_argument("--task", type=str, default="carla_right_turn_simple", help="CARLA task")
    parser.add_argument("--arm", type=str, default="cnn", choices=["cnn", "custom_jepa", "vjepa2"], help="Encoder arm")
    parser.add_argument("--episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="outputs/eval_results", help="Directory for output metrics/videos")
    parser.add_argument("--no-video", action="store_true", help="Disable video recording")
    parser.add_argument("--no-shield", action="store_true", help="Disable SafetySupervisor shield")
    parser.add_argument("--fps", type=int, default=20, help="Video FPS")

    args = parser.parse_args()

    run_evaluation(
        checkpoint_path=args.checkpoint,
        task=args.task,
        arm=args.arm,
        episodes=args.episodes,
        seed=args.seed,
        output_dir=args.output_dir,
        record_video=not args.no_video,
        use_safety_shield=not args.no_shield,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
