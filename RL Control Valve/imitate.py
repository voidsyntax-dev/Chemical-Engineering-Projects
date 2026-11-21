"""MPC-guided PPO pipeline (runs locally; MPC needs CPU cores, not a GPU).

  1. demos     : MPC drives randomized episodes (some with exploration noise), labels = MPC actions
  2. bc        : behaviour-clone the PPO actor onto the MPC            -> <out>/bc_model.zip
  3. finetune  : PPO with lambda-decaying BC regularisation + DAgger    -> <out>/final_model.zip

  python imitate.py                                   # full pipeline with defaults
  python imitate.py --stages finetune --ft-steps 1000000
Each stage reuses the previous stage's file if it already exists.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from cstr.env import VdVEnv
from cstr.imitation import ImitationCallback, bc_train, collect, worker_init
from train import Checkpointer


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="runs/kaggle_v2/run/final_model.zip",
                    help="PPO to start from, or 'fresh' for a new network (required with --integral-obs)")
    ap.add_argument("--train-mode", default="randomized", choices=["randomized", "randomized_wide"],
                    help="scenario distribution for demos, DAgger and PPO (never 'hard': that is held out)")
    ap.add_argument("--integral-obs", action="store_true",
                    help="add 5-min and 20-min error averages to the observation (13 inputs)")
    ap.add_argument("--out", default="runs/imit")
    ap.add_argument("--stages", default="demos,bc,finetune")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    ap.add_argument("--demo-episodes", type=int, default=160)
    ap.add_argument("--bc-epochs", type=int, default=40)
    ap.add_argument("--init-std", type=float, default=0.2, help="policy std after BC")
    ap.add_argument("--ft-steps", type=int, default=2_000_000)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--move-penalty", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lam0", type=float, default=1.0)
    ap.add_argument("--lam1", type=float, default=0.05)
    ap.add_argument("--bc-steps", type=int, default=10)
    ap.add_argument("--dagger-every", type=int, default=250_000)
    ap.add_argument("--dagger-episodes", type=int, default=32)
    ap.add_argument("--mpc-move-penalty", type=float, default=0.05)
    ap.add_argument("--checkpoint-every", type=int, default=250_000)
    ap.add_argument("--critic-warmup", type=int, default=150_000,
                    help="steps of critic-only training before the actor is unfrozen")
    ap.add_argument("--ft-std", type=float, default=0.1, help="exploration std at fine-tune start")
    ap.add_argument("--clip-range", type=float, default=0.1)
    ap.add_argument("--n-epochs", type=int, default=5)
    return ap.parse_args()


def main():
    a = parse()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stages = set(a.stages.split(","))
    t0 = time.time()
    pool = mp.get_context("spawn").Pool(a.workers, initializer=worker_init)
    print(f"workers={a.workers} | stages={sorted(stages)} | out={out}", flush=True)

    demo_file = out / "demos.npz"
    if "demos" in stages and not demo_file.exists():
        tasks = [(100_000 + i, a.train_mode, None, (0.0, 0.05, 0.15)[i % 3], a.mpc_move_penalty,
                  a.integral_obs) for i in range(a.demo_episodes)]
        print(f"Collecting {len(tasks)} MPC demo episodes ...", flush=True)
        O, A, mae = collect(pool, tasks)
        np.savez_compressed(demo_file, O=O, A=A)
        print(f"  {len(O)} samples, MPC-driven cB MAE {mae:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    d = np.load(demo_file)
    O, A = d["O"], d["A"]

    bc_file = out / "bc_model.zip"
    if "bc" in stages and not bc_file.exists():
        if a.base == "fresh":
            env = DummyVecEnv([lambda: VdVEnv(mode=a.train_mode, integral_obs=a.integral_obs)])
            model = PPO("MlpPolicy", env, n_steps=1024, batch_size=512, n_epochs=10, gamma=0.99,
                        gae_lambda=0.95, learning_rate=3e-4, clip_range=0.2, seed=0, device="cpu",
                        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]),
                                           activation_fn=torch.nn.Tanh))
        else:
            model = PPO.load(a.base, device="cpu")
        assert model.observation_space.shape[0] == O.shape[1], (
            f"model expects {model.observation_space.shape[0]} inputs, demos have {O.shape[1]}")
        print(f"Behaviour cloning {a.base} on {len(O)} MPC samples ...", flush=True)
        bc_train(model.policy, O, A, epochs=a.bc_epochs)
        with torch.no_grad():
            model.policy.log_std.fill_(math.log(a.init_std))
        model.save(bc_file)
        print(f"  saved {bc_file}  ({time.time() - t0:.0f}s)", flush=True)

    if "finetune" in stages:
        venv = make_vec_env(VdVEnv, n_envs=a.n_envs, seed=1, vec_env_cls=SubprocVecEnv,
                            env_kwargs=dict(mode=a.train_mode, move_penalty=a.move_penalty,
                                            integral_obs=a.integral_obs))
        eval_env = make_vec_env(VdVEnv, n_envs=1, seed=10_000, vec_env_cls=DummyVecEnv,
                                env_kwargs=dict(mode="perturbed", move_penalty=a.move_penalty,
                                                integral_obs=a.integral_obs))
        model = PPO.load(bc_file, env=venv, device="cpu",
                         custom_objects={"learning_rate": a.lr, "lr_schedule": lambda _: a.lr,
                                         "clip_range": a.clip_range, "n_epochs": a.n_epochs})
        with torch.no_grad():
            model.policy.log_std.fill_(math.log(a.ft_std))
        model.set_logger(configure(str(out / "logs" / datetime.now().strftime("%Y%m%d_%H%M%S")),
                                   ["stdout", "csv"]))
        meta = dict(kind="mpc_guided_ppo", base=a.base, move_penalty=a.move_penalty, lam0=a.lam0,
                    critic_warmup=a.critic_warmup, ft_std=a.ft_std, clip_range=a.clip_range,
                    integral_obs=a.integral_obs, train_mode=a.train_mode,
                    lam1=a.lam1, lr=a.lr, n_envs=a.n_envs, demo_samples=int(len(O)))
        ck = Checkpointer(out, a.checkpoint_every, meta)
        im = ImitationCallback(O, A, pool, str(out), a.ft_steps, lam0=a.lam0, lam1=a.lam1,
                               bc_steps=a.bc_steps, lr=a.lr, dagger_every=a.dagger_every,
                               dagger_episodes=a.dagger_episodes, mpc_w=a.mpc_move_penalty,
                               critic_warmup=a.critic_warmup, integral_obs=a.integral_obs,
                               train_mode=a.train_mode)
        ev = EvalCallback(eval_env, best_model_save_path=str(out / "best"), log_path=str(out / "eval"),
                          eval_freq=max(100_000 // a.n_envs, 1), n_eval_episodes=5, deterministic=True)
        print(f"Fine-tuning {a.ft_steps:,} steps from {model.num_timesteps:,} "
              f"(lambda {a.lam0} -> {a.lam1}, move penalty {a.move_penalty})", flush=True)
        try:
            model.learn(a.ft_steps, reset_num_timesteps=False, callback=CallbackList([im, ck, ev]))
        finally:
            ck.save()
            model.save(out / "final_model.zip")
            (out / "summary.json").write_text(json.dumps(
                dict(meta, num_timesteps=model.num_timesteps, dagger_rounds=im.round,
                     buffer=int(len(im.O)), wall_min=round((time.time() - t0) / 60, 1)), indent=2))
            venv.close()
    pool.close()
    pool.join()
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
