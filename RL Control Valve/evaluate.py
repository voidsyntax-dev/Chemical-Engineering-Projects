"""Compare a trained PPO policy against the dual-PI baseline on identical scenarios.

  python evaluate.py --model runs/vdv_ppo/final_model.zip --device cpu
  python evaluate.py --model runs/vdv_ppo/best/best_model.zip --mode perturbed --episodes 50
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cstr.baseline import DualPI
from cstr.env import VdVEnv


def run_episode(env, seed, policy=None, pi=None):
    obs, info = env.reset(seed=seed)
    if pi is not None:
        pi.reset(env)
    log, R, done = [info], 0.0, False
    while not done:
        a = policy.predict(obs, deterministic=True)[0] if policy is not None else pi.act(env, info)
        obs, r, term, trunc, info = env.step(a)
        R += r
        log.append(info)
        done = term or trunc
    return log, R, term


def metrics(log, R, term, env):
    cB = np.array([d["cB"] for d in log]); sp = np.array([d["cB_sp"] for d in log])
    T = np.array([d["T"] for d in log]); F = np.array([d["F"] for d in log])
    Q = np.array([d["Q"] for d in log])
    dt_min = env.dt_s / 60
    return dict(
        reward=R,
        mae_cB=float(np.mean(np.abs(cB - sp))),
        max_T=float(T.max()),
        min_above_Tsoft=float(np.sum(T > env.T_soft) * dt_min),
        runaway=bool(term),
        B_prod=float(np.mean(F * cB)),                    # mol B / (L h)
        moves_F=float(np.sum(np.abs(np.diff(F))) / env.rate[0]),
        moves_Q=float(np.sum(np.abs(np.diff(Q))) / env.rate[1]),
    )


def plot(logs, labels, path, env):
    fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for log, lab in zip(logs, labels):
        t = [d["t_min"] for d in log]
        ax[0].plot(t, [d["cB"] for d in log], label=lab)
        ax[1].plot(t, [d["T"] for d in log], label=lab)
        ax[2].plot(t, [d["F"] for d in log], label=lab)
        ax[3].plot(t, [d["Q"] for d in log], label=lab)
    ax[0].plot(t, [d["cB_sp"] for d in logs[0]], "k--", label="set-point")
    ax[1].axhline(env.T_soft, color="r", ls=":", label="T soft limit")
    for a, yl in zip(ax, ["cB [mol/L]", "T [C]", "F/V [1/h]", "QK [kJ/h]"]):
        a.set_ylabel(yl); a.grid(alpha=0.3)
    ax[0].legend(); ax[1].legend(); ax[-1].set_xlabel("t [min]")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="*", default=[], help="one or more PPO .zip files")
    ap.add_argument("--labels", nargs="*", default=None, help="column names for --model (default PPO, PPO2, ...)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mode", default="both",
                    help="'both' (nominal,perturbed), 'all' (nominal,perturbed,hard) or a comma list")
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--out", default="results")
    ap.add_argument("--move-penalty", type=float, default=0.05,
                    help="only affects the 'reward' metric; mae/moves are penalty-independent")
    ap.add_argument("--mpc", action="store_true", help="also evaluate the nonlinear MPC baseline")
    ap.add_argument("--mpc-horizon", type=int, default=20)
    ap.add_argument("--mpc-move-penalty", type=float, default=None,
                    help="MPC move weight (default: same as --move-penalty)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    policies = {}
    if args.model:
        from stable_baselines3 import PPO
        labels = args.labels or ["PPO" if i == 0 else f"PPO{i + 1}" for i in range(len(args.model))]
        assert len(labels) == len(args.model), "--labels must match --model"
        for lab, path in zip(labels, args.model):
            policies[lab] = PPO.load(path, device=args.device)
            print(f"Loaded {lab}: {path} on {policies[lab].device}")

    mpc = None
    if args.mpc:
        from cstr.mpc import NMPC
        mw = args.move_penalty if args.mpc_move_penalty is None else args.mpc_move_penalty
        mpc = NMPC(N=args.mpc_horizon, move_penalty=mw)

    modes = {"both": ["nominal", "perturbed"],
             "all": ["nominal", "perturbed", "hard"]}.get(args.mode, args.mode.split(","))
    report = {}
    for mode in modes:
        env = VdVEnv(mode=mode, move_penalty=args.move_penalty)
        env_int = VdVEnv(mode=mode, move_penalty=args.move_penalty, integral_obs=True)
        # same seed -> same scenario in both envs; the extra inputs consume no random numbers
        controllers = {"PI": dict(pi=DualPI())}
        if mpc is not None:
            controllers["MPC"] = dict(pi=mpc)
        for lab, pol in policies.items():
            controllers[lab] = dict(policy=pol)
        res = {c: [] for c in controllers}
        mpc_time = 0.0
        for i in range(args.episodes):
            s = args.seed + i
            logs = []
            for name, kw in controllers.items():
                t0 = time.perf_counter()
                pol = kw.get("policy")
                use = env_int if pol is not None and pol.observation_space.shape[0] == 13 else env
                lg, R, term = run_episode(use, s, **kw)
                if name == "MPC":
                    mpc_time += time.perf_counter() - t0
                res[name].append(metrics(lg, R, term, env))
                logs.append(lg)
            if i == 0:
                plot(logs, list(controllers), out / f"episode_{mode}.png", env)
        if mpc is not None:
            print(f"MPC: {1000 * mpc_time / (args.episodes * env.max_steps):.1f} ms per control step, "
                  f"{mpc.fails} unconverged solves")
        print(f"\n=== mode: {mode}  ({args.episodes} episodes, identical scenarios per controller) ===")
        keys = list(res["PI"][0].keys())
        print(f"{'metric':18s}" + "".join(f"{k:>14s}" for k in res))
        summ = {}
        for k in keys:
            row = {c: float(np.mean([m[k] for m in res[c]])) for c in res}
            summ[k] = row
            print(f"{k:18s}" + "".join(f"{row[c]:14.4f}" for c in res))
        report[mode] = summ
    (out / "eval_summary.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved plots + {out / 'eval_summary.json'}")


if __name__ == "__main__":
    main()
