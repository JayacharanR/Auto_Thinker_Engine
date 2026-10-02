#!/usr/bin/env python3
"""
Unified training script for Phase 1 (baseline CNN) and Phase 3 (encoder comparison).

Uses dreamerv3-torch (PyTorch) as the DreamerV3 backbone, with CarDreamer's
CARLA task definitions for the environment.

This replaces the previous train_cardreamer.py that tried to use CarDreamer's
JAX-based DreamerV3 (which was a framework mismatch with our PyTorch encoders).

Usage:
    # Single arm
    python scripts/train_cardreamer.py --arm cnn --task carla_right_turn_simple

    # Full 3-arm comparison (Phase 3)
    python scripts/train_cardreamer.py --comparison --task carla_right_turn_simple

    # Phase 1 baseline
    python scripts/train_cardreamer.py --arm cnn --task carla_right_turn_simple --steps 500000
"""

import argparse
import functools
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import gym
import numpy as np
import torch

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# dreamerv3-torch
DREAMER_DIR = PROJECT_ROOT / "third_party" / "dreamerv3_torch"
sys.path.insert(0, str(DREAMER_DIR))

# CarDreamer (for CARLA task definitions only — NOT for DreamerV3)
CARDREAMER_DIR = PROJECT_ROOT / "third_party" / "CarDreamer"
sys.path.insert(0, str(CARDREAMER_DIR))

from src.dreamer.cardreamer_encoder_hook import (
    DreamerV3EncoderHook,
    PrecomputedFeatureEncoder,
    build_feature_extractor,
)
from src.dreamer.carla_wrappers import DreamerObservation, EpisodeMetricsRecorder
from src.dreamer.encoder_adapter import freeze_parameters

DEFAULT_PROFILE = PROJECT_ROOT / "configs" / "laptop.yaml"


def seed_everything(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_phase3_config(config_path: Optional[str] = None) -> dict:
    """Load Phase 3 encoder comparison config."""
    from ruamel.yaml import YAML

    if config_path is None:
        config_path = str(PROJECT_ROOT / "configs" / "phase3_transfer_arms.yaml")
    ryaml = YAML(typ="safe", pure=True)
    with open(config_path) as f:
        return ryaml.load(f)


def build_encoder_hook(
    arm: str,
    phase3_config: dict,
    device: str,
) -> Optional[DreamerV3EncoderHook]:
    """
    Encoder hook for a transfer arm whose encoder trains (fine-tuning / LoRA).

    Frozen transfer arms do not use it: their features are computed at
    collection time (see build_feature_extractor). The custom_jepa hook loads
    its Phase 2 checkpoint itself (strictly) and fails if it is missing.
    """
    if arm == "cnn":
        return None
    if arm == "custom_jepa":
        checkpoint = phase3_config.get("encoders", {}).get("custom_jepa", {}).get("checkpoint")
        if not checkpoint or not os.path.isfile(checkpoint):
            raise FileNotFoundError(
                f"custom_jepa needs a Phase 2 checkpoint; configured: {checkpoint!r}"
            )
    return DreamerV3EncoderHook(
        arm=arm,
        config=phase3_config,
        device=device,
        obs_key="image",
        num_temporal_frames=phase3_config.get("frame_stacking", {}).get("num_frames", 4),
    )


def build_agent(obs_space, act_space, config, logger, dataset, arm, phase3_config, device):
    """Construct Dreamer for an arm, ready to train.

    Encoders by observation and arm:
    - ``route`` present: encoded (and reconstructed) by dreamerv3-torch's MLP,
      the same module for every arm.
    - ``feat`` present (frozen transfer arm): a trainable adapter over the
      embedding computed at collection time.
    - CNN arm: dreamerv3-torch's own MultiEncoder, the reference baseline.
    - Otherwise (trainable transfer arm): the encoder hook.
    Custom encoders are built before Dreamer so their parameters join the
    world-model optimizer.
    """
    import networks

    from dreamer import Dreamer

    config.num_actions = act_space.n if hasattr(act_space, "n") else act_space.shape[0]
    if getattr(act_space, "discrete", False):
        config.actor = dict(config.actor)
        config.actor["dist"] = "onehot"
        config.actor["std"] = "none"
    spaces = obs_space.spaces
    if "route" in spaces:
        config.encoder = {**config.encoder, "mlp_keys": "^route$"}
        config.decoder = {**config.decoder, "mlp_keys": "^route$"}

    hook = None
    if "feat" in spaces:
        vector_encoder = None
        if "route" in spaces:
            shapes = {k: tuple(v.shape) for k, v in spaces.items()}
            vector_encoder = networks.MultiEncoder(shapes, **{**config.encoder, "cnn_keys": "$^"})
        hook = PrecomputedFeatureEncoder(
            spaces["feat"].shape[0], phase3_config["adapter"], vector_encoder
        )
    elif arm != "cnn":
        if "route" in spaces:
            raise NotImplementedError("route observations need the cnn arm or a frozen encoder")
        hook = build_encoder_hook(arm, phase3_config, device)
    agent = Dreamer(
        obs_space, act_space, config, logger, dataset, custom_encoder=hook
    ).to(device)
    if isinstance(hook, DreamerV3EncoderHook) and (
        phase3_config.get("encoders", {}).get(arm, {}).get("freeze", False)
    ):
        freeze_parameters(hook.encoder.parameters())
    return agent


def make_carla_env(
    task_name: str,
    obs: str = "bev",
    action: str = "discrete",
    image_size: tuple = (64, 64),
    metrics_path: Optional[Path] = None,
    feature_extractor=None,
):
    """
    Create a CarDreamer CARLA task wrapped for dreamerv3-torch.

    ``obs`` selects the observation contract (see DreamerObservation);
    ``feature_extractor`` adds a frozen encoder's ``feat`` for transfer arms.
    ``action`` is CarDreamer's discrete action table (one-hot actor) or
    continuous ``[acceleration, steering]``. Returns ``(env, recorder)``; the
    recorder writes per-episode metrics when ``metrics_path`` is given.
    """
    import car_dreamer
    import envs.wrappers as wrappers

    task_argv = ["--env.action.discrete", str(action == "discrete")]
    if os.environ.get("CARLA_PORT"):
        task_argv.extend(["--env.world.carla_port", os.environ["CARLA_PORT"]])
    env, _ = car_dreamer.create_task(task_name, argv=task_argv)

    env = DreamerObservation(env, obs=obs, size=image_size, feature_extractor=feature_extractor)
    recorder = None
    if metrics_path is not None:
        env = recorder = EpisodeMetricsRecorder(env, metrics_path)
    if isinstance(env.action_space, gym.spaces.Discrete):
        env = wrappers.OneHotAction(env)
    else:
        env = wrappers.NormalizeActions(env)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)

    print(f"[train] CARLA action space: {env.action_space}")
    spaces = env.observation_space.spaces
    shapes = {k: v.shape for k, v in spaces.items() if not k.startswith("is_")}
    print(f"[train] Observation: {obs} -> {shapes}")
    return env, recorder


def load_dreamer_config(
    arm: str,
    task: str,
    seed: int,
    steps: int,
    image_size: tuple,
    device: str,
    profile: Optional[str] = None,
    overrides: tuple = (),
    logdir: Optional[str] = None,
) -> argparse.Namespace:
    """
    dreamerv3-torch defaults, then the run profile, then ``key=value`` overrides.

    ``steps`` is the environment-step budget for training including prefill
    (evaluation steps are not counted).
    """
    from ruamel.yaml import YAML

    ryaml = YAML(typ="safe", pure=True)
    config = dict(ryaml.load((DREAMER_DIR / "configs.yaml").read_text())["defaults"])

    def merge(updates: dict, source: str) -> None:
        for key, value in updates.items():
            if key not in config:
                raise KeyError(f"{source}: unknown Dreamer config key {key!r}")
            if isinstance(config[key], dict) and isinstance(value, dict):
                config[key] = {**config[key], **value}
            else:
                config[key] = value

    profile_path = Path(profile or DEFAULT_PROFILE)
    merge(ryaml.load(profile_path.read_text()).get("dreamer", {}), str(profile_path))
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects key=value, got {item!r}")
        merge({key.strip(): ryaml.load(value)}, "--set")

    config.update({
        "logdir": logdir or str(PROJECT_ROOT / "outputs" / "logs" / f"{arm}_seed{seed}"),
        "seed": seed,
        "steps": steps,
        "task": task,
        "device": device,
        "size": list(image_size),
    })
    return argparse.Namespace(**config)


def _peak_vram_mb() -> Optional[float]:
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 2**20, 1)


def train_arm(
    arm: str,
    task: str,
    seed: int,
    phase3_config: dict,
    steps: int = 500_000,
    obs: str = "bev",
    action: str = "discrete",
    image_size: tuple = (64, 64),
    profile: Optional[str] = None,
    overrides: tuple = (),
    logdir: Optional[str] = None,
    resume: bool = False,
) -> Dict:
    """
    Train one arm with dreamerv3-torch and return its metrics.

    Writes to the log directory: ``latest.pt`` after every chunk,
    ``episodes.jsonl`` (one record per finished episode), ``eval.jsonl`` (one
    summary per evaluation) and ``metrics.json`` at the end.
    """
    import tools
    from parallel import Damy
    from torch import distributions as torchd

    from dreamer import make_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(seed)
    config = load_dreamer_config(
        arm, task, seed, steps, image_size, device, profile, overrides, logdir
    )
    logdir = Path(config.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    train_dir, eval_dir = logdir / "train_eps", logdir / "eval_eps"
    latest_pt = logdir / "latest.pt"

    print(f"\n{'='*60}")
    print(f"Training arm '{arm}' on '{task}' (obs={obs}, action={action}), seed={seed}")
    print(f"Device: {device}, env steps: {steps}, logdir: {logdir}")
    print(f"{'='*60}")

    ckpt = None
    if resume:
        if not latest_pt.is_file():
            raise FileNotFoundError(f"--resume: no checkpoint at {latest_pt}")
        ckpt = torch.load(latest_pt, map_location=device, weights_only=False)
    elif latest_pt.is_file() or any(train_dir.glob("*.npz")):
        raise FileExistsError(f"{logdir} holds an earlier run; pass --resume or another --logdir")

    # Headless training does not need CarDreamer's Flask monitor.
    os.environ.setdefault("CARLA_DISABLE_MONITOR", "1")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.time()

    # One CARLA environment serves training and evaluation in turn (the
    # right-turn task has a single ego spawn point).
    env, recorder = make_carla_env(
        task, obs=obs, action=action, image_size=image_size,
        metrics_path=logdir / "episodes.jsonl",
        feature_extractor=build_feature_extractor(arm, phase3_config, device),
    )
    agent = None

    # Detached jobs ignore SIGINT; make SIGTERM (kill, timeout) save a checkpoint too.
    def _on_sigterm(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        train_envs, eval_envs = [Damy(env)], [Damy(env)]
        acts = env.action_space
        logger = tools.Logger(logdir, (ckpt["step"] if ckpt else 0) * config.action_repeat)
        train_eps = tools.load_episodes(train_dir, limit=config.dataset_size)
        eval_eps = tools.load_episodes(eval_dir, limit=1)

        # --- Prefill replay with random actions ---
        # A resumed run already had its prefill (fresh runs start from an empty logdir).
        prefill = 0 if ckpt else int(config.prefill)
        if getattr(acts, "discrete", False):
            random_actor = tools.OneHotDist(torch.zeros(acts.shape[0]).repeat(config.envs, 1))
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(acts.low).repeat(config.envs, 1),
                    torch.tensor(acts.high).repeat(config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            sample = random_actor.sample()
            return {"action": sample, "logprob": random_actor.log_prob(sample)}, None

        state = None
        recorder.step_fn = lambda: logger.step
        if prefill:
            print(f"[train] Prefilling replay with {prefill} random steps...")
            state = tools.simulate(
                random_agent, train_envs, train_eps, train_dir, logger,
                limit=config.dataset_size, steps=prefill,
            )
            logger.step += prefill * config.action_repeat

        # --- Agent ---
        agent = build_agent(
            env.observation_space, acts, config, logger,
            make_dataset(train_eps, config), arm, phase3_config, device,
        )
        agent.requires_grad_(requires_grad=False)  # tools.RequiresGrad enables per update
        if ckpt is not None:
            agent.load_state_dict(ckpt["agent_state_dict"])
            tools.recursively_load_optim_state_dict(agent, ckpt["optims_state_dict"])
            agent._should_pretrain._once = False
            print(f"[train] Resumed from {latest_pt} at env step {agent._step}")
        recorder.step_fn = lambda: agent._step
        print(f"[train] Actor: {config.actor['dist']} over {config.num_actions} actions; "
              f"encoder: {type(agent._wm.encoder).__name__}")

        run_spec = {"arm": arm, "task": task, "obs": obs, "action": action,
                    "image_size": list(image_size), "seed": seed}
        try:
            from src.utils.manifest import create_run_manifest
            create_run_manifest(
                config=vars(config), arm=arm, seed=seed, task=task,
                checkpoint_path=str(latest_pt) if ckpt else None,
                extra_metadata={"run_spec": run_spec, "profile": str(profile or DEFAULT_PROFILE),
                                "overrides": list(overrides)},
                output_path=logdir / "manifest.json",
            )
        except Exception as e:
            print(f"[train] Warning: Could not create manifest: {e}")

        def save_checkpoint():
            torch.save({
                "agent_state_dict": agent.state_dict(),
                "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
                "step": agent._step,
                "run_spec": run_spec,
                "dreamer_config": vars(config),
            }, latest_pt)

        # --- Train in chunks of eval_every env steps, evaluating after each ---
        target = int(config.steps)
        eval_summary = None
        train_started, train_start_step = time.time(), agent._step
        print(f"[train] Training from env step {agent._step} to {target} (prefill included)")
        while agent._step < target:
            chunk = min(int(config.eval_every), target - agent._step)
            recorder.mode = "train"
            state = tools.simulate(
                agent, train_envs, train_eps, train_dir, logger,
                limit=config.dataset_size, steps=chunk, state=state,
            )
            save_checkpoint()
            print(f"[train] env step {agent._step}/{target}, updates {agent._update_count}, "
                  f"peak VRAM {_peak_vram_mb()} MB")

            if config.eval_episode_num > 0:
                recorder.mode = "eval"
                tools.simulate(
                    functools.partial(agent, training=False), eval_envs, eval_eps,
                    eval_dir, logger, is_eval=True, episodes=config.eval_episode_num,
                )
                eval_summary = {"agent_step": agent._step, **recorder.reset_eval().summary()}
                for key in ("success_rate", "collision_rate", "out_of_lane_rate", "mean_reward"):
                    logger.scalar(f"eval_{key}", eval_summary[key])
                logger.write(step=logger.step)
                with open(logdir / "eval.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(eval_summary) + "\n")
                print(f"[train] eval @ {agent._step}: success {eval_summary['success_rate']:.2f}, "
                      f"collision {eval_summary['collision_rate']:.2f}, "
                      f"return {eval_summary['mean_reward']:.1f}")
                # Evaluation reset the shared environment; restart the train episode.
                state = None
    except KeyboardInterrupt:
        if agent is not None:
            save_checkpoint()
            print(f"\n[train] Interrupted; saved {latest_pt} at env step {agent._step}")
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        try:
            env.close()
        except Exception as e:
            print(f"[train] Warning: env.close() failed: {e}")

    train_seconds = time.time() - train_started
    final_metrics = {
        **run_spec,
        "env_steps": int(agent._step),
        "updates": int(agent._update_count),
        "wall_clock_sec": round(time.time() - started, 1),
        "train_steps_per_sec": round(
            (agent._step - train_start_step) / max(train_seconds, 1e-9), 2
        ),
        "peak_vram_mb": _peak_vram_mb(),
        "final_eval": eval_summary,
        "train": recorder.trackers["train"].summary(),
    }
    if eval_summary:
        final_metrics.update({
            "eval_reward": eval_summary["mean_reward"],
            "eval_length": eval_summary["mean_steps"],
            "eval_success": eval_summary["success_rate"],
        })
    with open(logdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"[train] Done: {json.dumps(final_metrics)}")
    return final_metrics

def summarize_run(logdir: Path, success_threshold: float = 0.8) -> Dict[str, float]:
    """
    Comparison metrics of one finished run, from its metrics.json and eval.jsonl.

    ``steps_to_threshold`` is the first evaluated env step whose success rate
    reached ``success_threshold`` (NaN if never).
    """
    metrics = json.loads((logdir / "metrics.json").read_text())
    final = metrics.get("final_eval") or {}
    evals = []
    if (logdir / "eval.jsonl").is_file():
        evals = [json.loads(line) for line in (logdir / "eval.jsonl").read_text().splitlines()]
    reached = [e["agent_step"] for e in evals if e["success_rate"] >= success_threshold]
    return {
        "success_rate": final.get("success_rate", float("nan")),
        "collision_rate": final.get("collision_rate", float("nan")),
        "out_of_lane_rate": final.get("out_of_lane_rate", float("nan")),
        "eval_return": final.get("mean_reward", float("nan")),
        "steps_to_threshold": float(reached[0]) if reached else float("nan"),
        "wall_clock_hours": metrics.get("wall_clock_sec", float("nan")) / 3600,
        "peak_vram_gb": (metrics.get("peak_vram_mb") or float("nan")) / 1024,
        "train_steps_per_sec": metrics.get("train_steps_per_sec", float("nan")),
    }


def run_comparison(
    task: str,
    phase3_config: dict,
    steps: int = 500_000,
    arms: tuple = ("cnn", "custom_jepa", "vjepa2"),
    output_dir: Optional[str] = None,
    success_threshold: float = 0.8,
    **run_kwargs,
):
    """
    Train every arm with the same seeds, budget and evaluation protocol, then
    write ``comparison.md`` / ``comparison.json`` (mean +- std across seeds).

    Each run has its own log directory under ``output_dir``. Finished runs
    (metrics.json present) are reused and interrupted runs resume, so the
    comparison can be restarted after a crash.
    """
    from src.eval.metrics import ComparisonTable

    seeds = phase3_config.get("experiment", {}).get("seeds", [42, 123, 456])
    out = Path(output_dir or PROJECT_ROOT / "outputs" / "comparison"
               / f"{task}_{run_kwargs.get('obs', 'bev')}")
    out.mkdir(parents=True, exist_ok=True)

    table, runs = ComparisonTable(), []
    for arm in arms:
        for seed in seeds:
            logdir = out / f"{arm}_seed{seed}"
            if not (logdir / "metrics.json").is_file():
                print(f"\n{'#' * 60}\n# Comparison: {arm} / seed {seed}\n{'#' * 60}")
                train_arm(
                    arm=arm, task=task, seed=seed, phase3_config=phase3_config, steps=steps,
                    logdir=str(logdir), resume=(logdir / "latest.pt").is_file(), **run_kwargs,
                )
            summary = summarize_run(logdir, success_threshold)
            table.add_result(arm, seed, summary)
            runs.append({"arm": arm, "seed": seed, **summary})

    table.save_table(str(out / "comparison.md"))
    (out / "comparison.json").write_text(json.dumps(
        {"task": task, "steps": steps, "seeds": seeds, "success_threshold": success_threshold,
         "runs": runs}, indent=2,
    ))
    print(f"\n{table.generate_table()}\n\n[comparison] Results in {out}")


def main():
    parser = argparse.ArgumentParser(description="DreamerV3 training with encoder comparison")
    parser.add_argument("--arm", type=str, default="cnn", choices=["cnn", "custom_jepa", "vjepa2"])
    parser.add_argument("--task", type=str, default="carla_right_turn_simple")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=500_000,
                        help="Environment steps for training, prefill included")
    parser.add_argument("--obs", type=str, default="bev",
                        choices=["bev", "camera", "camera_route"],
                        help="Agent observation: bird's-eye view with route, front camera, "
                             "or front camera + route/speed vector")
    parser.add_argument(
        "--action", type=str, default="discrete", choices=["discrete", "continuous"]
    )
    parser.add_argument("--image-size", type=int, nargs=2, default=[64, 64])
    parser.add_argument("--config", type=str, default=None, help="Phase 3 config path")
    parser.add_argument("--profile", type=str, default=None,
                        help="Dreamer run profile (default: configs/laptop.yaml)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="Override a Dreamer config key, e.g. --set prefill=500 (repeatable)")
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--comparison", action="store_true", help="Run 3-arm comparison")
    parser.add_argument("--arms", nargs="+", default=["cnn", "custom_jepa", "vjepa2"],
                        choices=["cnn", "custom_jepa", "vjepa2"], help="Arms for --comparison")
    parser.add_argument("--resume", action="store_true", help="Resume from <logdir>/latest.pt")
    args = parser.parse_args()

    phase3_config = load_phase3_config(args.config)
    run_kwargs = dict(
        obs=args.obs,
        action=args.action,
        image_size=tuple(args.image_size),
        profile=args.profile,
        overrides=tuple(args.overrides),
    )

    if args.comparison:
        run_comparison(
            task=args.task, phase3_config=phase3_config, steps=args.steps,
            arms=tuple(args.arms), output_dir=args.logdir, **run_kwargs,
        )
    else:
        train_arm(
            arm=args.arm,
            task=args.task,
            seed=args.seed,
            phase3_config=phase3_config,
            steps=args.steps,
            logdir=args.logdir,
            resume=args.resume,
            **run_kwargs,
        )


if __name__ == "__main__":
    main()
