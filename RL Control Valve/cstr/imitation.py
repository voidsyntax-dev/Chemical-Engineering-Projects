"""MPC-guided PPO: demonstrations, behaviour cloning, and a BC-regularised PPO callback.

    loss_total ~ PPO_loss  +  lambda(t) * || mean_pi(obs) - a_MPC(obs) ||^2
    lambda(t)  = lam0 * (lam1 / lam0) ** progress        (geometric decay, the "ratio")

Labels always come from the MPC. Data comes from
  * demos   : MPC (optionally with exploration noise) drives the plant         -> MPC's own states
  * DAgger  : the CURRENT PPO policy drives the plant, MPC runs in shadow mode -> learner's states
Running the MPC in shadow mode keeps its observer correct: after every env.step it is told which
input was really applied (NMPC.sync_applied).
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

_W: dict = {}   # per-worker-process cache (CasADi solver, envs, loaded policy)


def worker_init():
    torch.set_num_threads(1)
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def labeled_episode(task):
    """task = (seed, mode, policy_path or None, noise_std, mpc_move_penalty[, integral_obs]).
    Returns obs (T, 11 or 13) float32, MPC actions (T, 2) float32, episode cB MAE (executor's)."""
    seed, mode, policy_path, noise, mpc_w = task[:5]
    integral = bool(task[5]) if len(task) > 5 else False
    from cstr.env import VdVEnv
    from cstr.mpc import NMPC

    if "mpc" not in _W:
        _W["mpc"] = NMPC(move_penalty=mpc_w)
    mpc = _W["mpc"]
    key = ("env", mode, integral)
    if key not in _W:
        _W[key] = VdVEnv(mode=mode, integral_obs=integral)
    env = _W[key]
    pol = None
    if policy_path:
        if _W.get("pol_path") != policy_path:
            from stable_baselines3 import PPO
            _W["pol"] = PPO.load(policy_path, device="cpu")
            _W["pol_path"] = policy_path
        pol = _W["pol"]
    rng = np.random.default_rng(seed + 7919)

    obs, info = env.reset(seed=seed)
    mpc.reset(env)
    O, A, err = [], [], []
    done = False
    while not done:
        a_mpc = mpc.act(env, info)
        if pol is not None:
            a = pol.predict(obs, deterministic=True)[0]
        else:
            a = np.clip(a_mpc + noise * rng.standard_normal(2), -1.0, 1.0)
        O.append(obs)
        A.append(a_mpc)
        obs, _, term, trunc, info = env.step(a)
        mpc.sync_applied(env)
        err.append(abs(info["cB"] - info["cB_sp"]))
        done = term or trunc
    return np.asarray(O, np.float32), np.asarray(A, np.float32), float(np.mean(err))


def collect(pool, tasks):
    res = pool.map(labeled_episode, tasks, chunksize=1)
    O = np.concatenate([r[0] for r in res])
    A = np.concatenate([r[1] for r in res])
    return O, A, float(np.mean([r[2] for r in res]))


def actor_params(policy):
    return list(policy.mlp_extractor.policy_net.parameters()) + list(policy.action_net.parameters())


def bc_loss(policy, obs, act):
    mean = policy.get_distribution(obs).distribution.mean
    return ((mean - act) ** 2).mean()


def bc_train(policy, O, A, epochs=30, lr=1e-3, batch=512, seed=0, log=print):
    """Plain behaviour cloning of the actor mean onto MPC actions (value net untouched)."""
    dev = policy.device
    Ot = torch.as_tensor(O, device=dev)
    At = torch.as_tensor(A, device=dev)
    opt = torch.optim.Adam(actor_params(policy), lr=lr)
    g = torch.Generator().manual_seed(seed)
    n = len(Ot)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = bc_loss(policy, Ot[idx], At[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(idx)
        if ep == 0 or (ep + 1) % 5 == 0 or ep == epochs - 1:
            log(f"  BC epoch {ep + 1:3d}/{epochs}: mse {tot / n:.5f}")
    return tot / n


class ImitationCallback(BaseCallback):
    """After every PPO update: `bc_steps` minibatch steps of lambda-weighted BC on the aggregated
    MPC-labelled buffer. Every `dagger_every` steps: a DAgger round with the current policy."""

    def __init__(self, O, A, pool, out_dir, total_steps, lam0=1.0, lam1=0.05, bc_steps=10,
                 batch=512, lr=1e-4, dagger_every=250_000, dagger_episodes=32, mpc_w=0.05,
                 max_buffer=400_000, seed0=500_000, critic_warmup=0, integral_obs=False,
                 train_mode="randomized"):
        super().__init__()
        self.train_mode = train_mode
        self.integral_obs = integral_obs
        self.critic_warmup = critic_warmup
        self.frozen = False
        self.O, self.A = O, A
        self.pool, self.out_dir = pool, out_dir
        self.total = total_steps
        self.lam0, self.lam1 = lam0, lam1
        self.bc_steps, self.batch, self.lr = bc_steps, batch, lr
        self.dagger_every, self.dagger_episodes = dagger_every, dagger_episodes
        self.mpc_w, self.max_buffer = mpc_w, max_buffer
        self.seed = seed0
        self.round = 0
        self.rng = np.random.default_rng(1)

    def _on_training_start(self):
        self.start = self.model.num_timesteps
        self.next_dagger = self.start + self.dagger_every
        self.opt = torch.optim.Adam(actor_params(self.model.policy), lr=self.lr)
        if self.critic_warmup > 0:
            self._set_actor_trainable(False)
            print(f"[warm-up] actor frozen for {self.critic_warmup:,} steps; training critic only", flush=True)

    def _set_actor_trainable(self, flag):
        for p in actor_params(self.model.policy) + [self.model.policy.log_std]:
            p.requires_grad_(flag)
        self.frozen = not flag

    def lam(self):
        prog = min(1.0, (self.model.num_timesteps - self.start) / max(1, self.total))
        return self.lam0 * (self.lam1 / self.lam0) ** prog

    def _dagger(self):
        self.round += 1
        path = os.path.join(self.out_dir, f"dagger_policy_{self.round}.zip")
        self.model.save(path)
        tasks = [(self.seed + i, self.train_mode, path, 0.0, self.mpc_w, self.integral_obs)
                 for i in range(self.dagger_episodes)]
        self.seed += self.dagger_episodes
        O, A, mae = collect(self.pool, tasks)
        self.O = np.concatenate([self.O, O])[-self.max_buffer:]
        self.A = np.concatenate([self.A, A])[-self.max_buffer:]
        print(f"[DAgger {self.round}] +{len(O)} samples (buffer {len(self.O)}); "
              f"policy cB MAE on these episodes {mae:.4f}", flush=True)
        self.logger.record("imit/dagger_policy_mae", mae)
        os.remove(path)

    def _on_rollout_start(self):
        if self.model.num_timesteps == self.start:
            return
        if self.frozen:
            if self.model.num_timesteps - self.start < self.critic_warmup:
                return                      # critic-only phase: actor stays = BC policy
            self._set_actor_trainable(True)
            print(f"[warm-up] done at {self.model.num_timesteps:,}; actor unfrozen", flush=True)
        if self.model.num_timesteps >= self.next_dagger:
            self._dagger()
            self.next_dagger += self.dagger_every
        lam = self.lam()
        pol = self.model.policy
        dev = pol.device
        tot = 0.0
        for _ in range(self.bc_steps):
            idx = self.rng.integers(0, len(self.O), self.batch)
            loss = bc_loss(pol, torch.as_tensor(self.O[idx], device=dev), torch.as_tensor(self.A[idx], device=dev))
            self.opt.zero_grad()
            (lam * loss).backward()
            self.opt.step()
            tot += loss.item()
        self.logger.record("imit/lambda", lam)
        self.logger.record("imit/bc_mse", tot / self.bc_steps)

    def _on_step(self):
        return True
