"""Resumable PPO training for the Van de Vusse CSTR.

--total-steps is the CUMULATIVE target. With --resume auto the script picks the newest
checkpoint it can find (this run's save dir first, then anything under /kaggle/input) and
trains only the remaining steps.

  python train.py --total-steps 2000000                      # fresh run to 2M
  python train.py --total-steps 5000000 --resume auto        # continue the same run to 5M
  python train.py --total-steps 5000000 --resume path/to/ppo_2000000.zip
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from cstr.env import VdVEnv

CKPT_RE = re.compile(r"ppo_(\d+)\.zip$")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total-steps", type=int, default=2_000_000, help="cumulative target timesteps")
    ap.add_argument("--save-dir", default="runs/vdv_ppo")
    ap.add_argument("--resume", default="auto", help="'auto', 'none', or a checkpoint .zip path")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--train-mode", default="randomized", choices=["nominal", "randomized", "perturbed"])
    ap.add_argument("--checkpoint-every", type=int, default=250_000)
    ap.add_argument("--eval-every", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-steps", type=int, default=1024, help="rollout length per env")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--move-penalty", type=float, default=0.05,
                    help="reward weight on squared (normalised) valve moves; higher = smoother actuators")
    return ap.parse_args()


def ckpt_steps(path: Path) -> int:
    m = CKPT_RE.search(path.name)
    return int(m.group(1)) if m else -1


def find_checkpoint(arg: str, save_dir: Path):
    if arg == "none":
        return None
    if arg != "auto":
        p = Path(arg)
        if not p.exists():
            sys.exit(f"--resume path not found: {p}")
        return p
    cands = list((save_dir / "checkpoints").glob("ppo_*.zip"))
    if os.path.isdir("/kaggle/input"):
        cands += [Path(c) for c in glob.glob("/kaggle/input/**/ppo_*.zip", recursive=True)]
    cands = [c for c in cands if ckpt_steps(c) >= 0]
    return max(cands, key=ckpt_steps) if cands else None


class Checkpointer(BaseCallback):
    """Saves ppo_<steps>.zip + latest.zip + meta json every `every` timesteps and at the end."""

    def __init__(self, save_dir: Path, every: int, meta: dict):
        super().__init__()
        self.dir = save_dir / "checkpoints"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.every, self.meta = every, meta
        self.last = 0
        self.start_steps = 0
        self.t0 = time.time()

    def _on_training_start(self):
        self.last = self.model.num_timesteps
        self.start_steps = self.model.num_timesteps

    def save(self):
        n = self.model.num_timesteps
        path = self.dir / f"ppo_{n}.zip"
        self.model.save(path)
        shutil.copy(path, self.dir / "latest.zip")
        el = time.time() - self.t0
        info = dict(self.meta, num_timesteps=n, saved_at=datetime.now().isoformat(timespec="seconds"),
                    elapsed_s_this_run=round(el, 1),
                    fps_this_run=round((n - self.start_steps) / max(el, 1e-6), 1))
        (self.dir / f"ppo_{n}.json").write_text(json.dumps(info, indent=2))
        (self.dir / "latest.json").write_text(json.dumps(info, indent=2))
        print(f"[checkpoint] {path}  ({info['fps_this_run']} steps/s)", flush=True)

    def _on_step(self):
        if self.model.num_timesteps - self.last >= self.every:
            self.save()
            self.last = self.model.num_timesteps
        return True


def main():
    args = parse_args()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(f"torch {torch.__version__} | sb3 {sb3.__version__} | device={device} | GPU={gpu} | "
          f"CPUs={os.cpu_count()} | n_envs={args.n_envs}", flush=True)

    vec_cls = SubprocVecEnv if args.n_envs > 1 else DummyVecEnv
    venv = make_vec_env(VdVEnv, n_envs=args.n_envs, seed=args.seed, vec_env_cls=vec_cls,
                        env_kwargs=dict(mode=args.train_mode, move_penalty=args.move_penalty))
    eval_env = make_vec_env(VdVEnv, n_envs=1, seed=10_000, vec_env_cls=DummyVecEnv,
                            env_kwargs=dict(mode="perturbed", move_penalty=args.move_penalty))
    print(f"train_mode={args.train_mode} | move_penalty={args.move_penalty}", flush=True)

    ckpt = find_checkpoint(args.resume, save_dir)
    if ckpt is not None:
        print(f"Resuming from {ckpt}  ({ckpt_steps(ckpt):,} steps)", flush=True)
        model = PPO.load(ckpt, env=venv, device=device)
        local = save_dir / "checkpoints" / ckpt.name
        local.parent.mkdir(parents=True, exist_ok=True)
        if ckpt.resolve() != local.resolve():          # keep this run's output self-contained
            shutil.copy(ckpt, local)
            meta = ckpt.with_suffix(".json")
            if meta.exists():
                shutil.copy(meta, local.with_suffix(".json"))
    else:
        print("Starting a fresh model", flush=True)
        model = PPO("MlpPolicy", venv, n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=10,
                    gamma=0.99, gae_lambda=0.95, learning_rate=args.lr, clip_range=0.2,
                    ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, seed=args.seed, device=device,
                    policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                                       activation_fn=torch.nn.Tanh))

    remaining = args.total_steps - model.num_timesteps
    if remaining <= 0:
        print(f"Already at {model.num_timesteps:,} >= target {args.total_steps:,}; nothing to train.")
        model.save(save_dir / "final_model.zip")
        return

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    model.set_logger(configure(str(save_dir / "logs" / run_id), ["stdout", "csv"]))
    meta = dict(train_mode=args.train_mode, move_penalty=args.move_penalty, device=device, gpu=gpu, n_envs=args.n_envs,
                target_steps=args.total_steps, sb3=sb3.__version__, torch=torch.__version__)
    ck = Checkpointer(save_dir, args.checkpoint_every, meta)
    ev = EvalCallback(eval_env, best_model_save_path=str(save_dir / "best"),
                      log_path=str(save_dir / "eval"), eval_freq=max(args.eval_every // args.n_envs, 1),
                      n_eval_episodes=5, deterministic=True, verbose=1)

    print(f"Training {remaining:,} more steps -> {args.total_steps:,} total", flush=True)
    t0 = time.time()
    try:
        model.learn(total_timesteps=remaining, reset_num_timesteps=False,
                    callback=CallbackList([ck, ev]), progress_bar=False)
    except KeyboardInterrupt:
        print("Interrupted - saving checkpoint", flush=True)
    finally:
        ck.save()
        model.save(save_dir / "final_model.zip")
        summary = dict(meta, num_timesteps=model.num_timesteps,
                       wall_time_min=round((time.time() - t0) / 60, 2))
        (save_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2), flush=True)
        venv.close()


if __name__ == "__main__":
    main()
