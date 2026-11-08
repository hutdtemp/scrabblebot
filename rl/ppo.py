#!/usr/bin/env python3
"""Behavior cloning from the GA champion, then PPO, on the Rust simulator.

Apples-to-apples with the GA: same 90 inputs (63 bid + 27 play features), a
policy of about the GA's size (90 -> H tanh -> play/bid heads). The critic is
larger but is training-only and never deployed.

Usage (from the repo root, after `maturin develop` in rlenv/):
  .venv/bin/python rl/ppo.py --out runs/rl1 --teacher runs/ga7/best.json \
      --pool genomes/panel_v2.json runs/ga7/best.json --minutes 20
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import scrabble_rlenv as se


BINS = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 18, 20, 22, 25,
                     28, 32, 36, 40, 45, 50, 60, 75], dtype=torch.float32)


class Actor(nn.Module):
    """Play: 8-way categorical. Bid: discretized Gaussian over BINS (mu, sigma),
    matching src/sim/rl.rs (mu = 10*z0 + 10, sigma = exp(clamp(z1)) + 0.5)."""

    def __init__(self, n_obs, hidden, n_play, n_bid):
        super().__init__()
        assert n_bid == len(BINS)
        self.l1 = nn.Linear(n_obs, hidden)
        self.play = nn.Linear(hidden, n_play)
        self.bid = nn.Linear(hidden, 2)
        self.register_buffer("bins", BINS.clone())

    def forward(self, x):
        h = torch.tanh(self.l1(x))
        z = self.bid(h)
        mu = 10.0 * z[:, :1] + 10.0
        sigma = torch.exp(z[:, 1:].clamp(-0.7, 3.4)) + 0.5
        bid_logits = -((self.bins[None, :] - mu) ** 2) / (2 * sigma ** 2)
        return self.play(h), bid_logits

    def export(self, path):
        sd = {k: v.detach().cpu().numpy().tolist() for k, v in self.state_dict().items()}
        out = {
            "w1": sd["l1.weight"], "b1": sd["l1.bias"],
            "wp": sd["play.weight"], "bp": sd["play.bias"],
            "wb": sd["bid.weight"], "bb": sd["bid.bias"],
        }
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f)
        os.replace(tmp, path)


class Critic(nn.Module):
    def __init__(self, n_obs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_obs, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def observe(env, n, with_teacher=False):
    ob, tb = env.observe(with_teacher)
    x = np.frombuffer(ob, dtype=np.float32).reshape(n, env.n_obs).copy()
    t = np.frombuffer(tb, dtype=np.uint8).reshape(n, 2).astype(np.int64)
    return x, t


def step(env, play, bid):
    r, d, p, s = env.step(bytes(play.astype(np.uint8)), bytes(bid.astype(np.uint8)))
    return (
        np.frombuffer(r, dtype=np.float32).copy(),
        np.frombuffer(d, dtype=np.uint8).astype(bool),
        np.frombuffer(p, dtype=np.float32).copy(),
        np.frombuffer(s, dtype=np.float32).copy(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--pool", nargs="+", required=True)
    ap.add_argument("--model", default="data/maincloud/opponent_model.json")
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--horizon", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=12)
    ap.add_argument("--minutes", type=float, default=20.0)
    ap.add_argument("--bc-minutes", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--snapshot-every", type=int, default=10)
    ap.add_argument("--full-frac", type=float, default=0.5)
    ap.add_argument("--reward", choices=["score", "placement"], default="placement")
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--league", type=int, default=1, help="add own snapshots to the opponent pool")
    ap.add_argument("--init-actor", help="resume from an exported .rl.json actor (skips BC)")
    ap.add_argument("--init-critic", help="resume critic weights (.pt)")
    ap.add_argument("--critic-warmup-minutes", type=float, default=1.5)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dev = torch.device(args.device)
    N, T = args.envs, args.horizon
    env = se.VecEnv(N, args.seed, list(args.pool), args.model, args.teacher, args.full_frac, args.reward)
    actor = Actor(env.n_obs, args.hidden, env.n_play, env.n_bid).to(dev)
    critic = Critic(env.n_obs).to(dev)
    if args.init_actor:
        w = json.load(open(args.init_actor))
        with torch.no_grad():
            actor.l1.weight.copy_(torch.tensor(w["w1"]))
            actor.l1.bias.copy_(torch.tensor(w["b1"]))
            actor.play.weight.copy_(torch.tensor(w["wp"]))
            actor.play.bias.copy_(torch.tensor(w["bp"]))
            actor.bid.weight.copy_(torch.tensor(w["wb"]))
            actor.bid.bias.copy_(torch.tensor(w["bb"]))
    if args.init_critic:
        critic.load_state_dict(torch.load(args.init_critic, map_location=dev))
    n_params = sum(p.numel() for p in actor.parameters())
    print(f"actor params {n_params} (deployed), device {dev}, envs {N}", flush=True)
    opt_a = torch.optim.Adam(actor.parameters(), lr=3e-3)
    opt_c = torch.optim.Adam(critic.parameters(), lr=1e-3)
    log = open(os.path.join(args.out, "log.csv"), "a")
    log.write("phase,update,secs,steps,episodes,pairwise,score,acc_play,acc_bid,entropy,kl,vloss\n")

    gamma, lam = args.gamma, 0.95
    t0 = time.time()
    total_steps = 0
    update = 0
    recent_pair, recent_score = [], []
    snapshots = []

    def rollout(mode):
        """mode: 'teacher' (act with teacher actions) or 'policy' (sample)."""
        nonlocal total_steps
        X = np.zeros((T, N, env.n_obs), np.float32)
        AP = np.zeros((T, N), np.int64)
        AB = np.zeros((T, N), np.int64)
        TP = np.zeros((T, N), np.int64)
        TB = np.zeros((T, N), np.int64)
        LP = np.zeros((T, N), np.float32)
        R = np.zeros((T, N), np.float32)
        D = np.zeros((T, N), np.float32)
        V = np.zeros((T + 1, N), np.float32)
        for t in range(T):
            x, te = observe(env, N, with_teacher=(mode == "teacher"))
            X[t] = x
            TP[t], TB[t] = te[:, 0], te[:, 1]
            with torch.no_grad():
                xt = torch.from_numpy(x).to(dev)
                lp_, lb_ = actor(xt)
                V[t] = critic(xt).cpu().numpy()
                dp = torch.distributions.Categorical(logits=lp_)
                db = torch.distributions.Categorical(logits=lb_)
                if mode == "teacher":
                    ap_ = torch.from_numpy(TP[t]).to(dev)
                    ab_ = torch.from_numpy(TB[t]).to(dev)
                else:
                    ap_, ab_ = dp.sample(), db.sample()
                LP[t] = (dp.log_prob(ap_) + db.log_prob(ab_)).cpu().numpy()
                AP[t], AB[t] = ap_.cpu().numpy(), ab_.cpu().numpy()
            r, d, pw, sc = step(env, AP[t], AB[t])
            R[t], D[t] = r, d.astype(np.float32)
            recent_pair.extend(pw[d].tolist())
            recent_score.extend(sc[d].tolist())
            total_steps += N
        with torch.no_grad():
            x, _ = observe(env, N)
            V[T] = critic(torch.from_numpy(x).to(dev)).cpu().numpy()
        adv = np.zeros((T, N), np.float32)
        last = np.zeros(N, np.float32)
        for t in reversed(range(T)):
            nonterm = 1.0 - D[t]
            delta = R[t] + gamma * V[t + 1] * nonterm - V[t]
            last = delta + gamma * lam * nonterm * last
            adv[t] = last
        ret = adv + V[:T]
        f = lambda a: torch.from_numpy(a.reshape(T * N, *a.shape[2:])).to(dev)  # noqa: E731
        return dict(x=f(X), ap=f(AP), ab=f(AB), tp=f(TP), tb=f(TB), lp=f(LP), adv=f(adv), ret=f(ret))

    def stats():
        p = float(np.mean(recent_pair[-2000:])) if recent_pair else float("nan")
        s = float(np.mean(recent_score[-2000:])) if recent_score else float("nan")
        return p, s

    def write(phase, acc_p=0.0, acc_b=0.0, ent=0.0, kl=0.0, vloss=0.0):
        p, s = stats()
        secs = time.time() - t0
        log.write(f"{phase},{update},{secs:.0f},{total_steps},{env.episodes},{p:.4f},{s:.1f},"
                  f"{acc_p:.3f},{acc_b:.3f},{ent:.3f},{kl:.4f},{vloss:.4f}\n")
        log.flush()
        print(f"[{phase} {update:4d} {secs:5.0f}s] steps {total_steps} pairwise {p:.3f} score {s:.1f} "
              f"acc {acc_p:.2f}/{acc_b:.2f} ent {ent:.2f} kl {kl:.4f} vloss {vloss:.3f} "
              f"({total_steps / max(secs, 1):.0f} steps/s)", flush=True)

    if args.init_actor:
        # Resume: no cloning; warm the critic on the current policy first so
        # early PPO updates use sensible advantages.
        warm_end = time.time() + 60 * args.critic_warmup_minutes
        while time.time() < warm_end:
            b = rollout("policy")
            for i in range(0, T * N, 4096):
                j = torch.randperm(T * N, device=dev)[:4096]
                vl = F.mse_loss(critic(b["x"][j]), b["ret"][j])
                opt_c.zero_grad()
                vl.backward()
                opt_c.step()
            update += 1
            write("critic-warmup", vloss=vl.item())
        args.bc_minutes = 0.0

    # ---- Phase 1: behavior cloning (actor) + value pretraining (critic) ----
    # Collect teacher-driven rollouts, then fit offline with many gradient
    # steps (a few per rollout is far too few for the bid head).
    bc_end = t0 + 60 * args.bc_minutes
    buf = []
    while time.time() < bc_end:
        buf.append(rollout("teacher"))
        update += 1
        write("bc-collect")
    data = {k: torch.cat([d[k] for d in buf]) for k in ("x", "tp", "tb", "ret")} if buf else None
    n = len(data["x"]) if data else 0
    for it in range(3000 if data else 0):
        j = torch.randint(0, n, (4096,), device=dev)
        lp_, lb_ = actor(data["x"][j])
        loss = F.cross_entropy(lp_, data["tp"][j]) + F.cross_entropy(lb_, data["tb"][j])
        opt_a.zero_grad()
        loss.backward()
        opt_a.step()
        vl = F.mse_loss(critic(data["x"][j]), data["ret"][j])
        opt_c.zero_grad()
        vl.backward()
        opt_c.step()
    if data:
        with torch.no_grad():
            j = torch.randint(0, n, (20000,), device=dev)
            lp_, lb_ = actor(data["x"][j])
            acc_p = (lp_.argmax(-1) == data["tp"][j]).float().mean().item()
            acc_b = ((lb_.argmax(-1) - data["tb"][j]).abs() <= 1).float().mean().item()
        write("bc-fit", acc_p, acc_b, vloss=vl.item())
    del buf, data
    actor.export(os.path.join(args.out, "bc.rl.json"))

    # ---- Phase 2: PPO with a growing league of our own snapshots ----
    for g in opt_a.param_groups:
        g["lr"] = 3e-4
    end = t0 + 60 * args.minutes
    clip, ent_coef = 0.2, 0.003
    while time.time() < end:
        b = rollout("policy")
        adv = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8)
        kls, ents = [], []
        for epoch in range(4):
            idx = torch.randperm(T * N, device=dev)
            for i in range(0, T * N, 4096):
                j = idx[i:i + 4096]
                lp_, lb_ = actor(b["x"][j])
                dp = torch.distributions.Categorical(logits=lp_)
                db = torch.distributions.Categorical(logits=lb_)
                newlp = dp.log_prob(b["ap"][j]) + db.log_prob(b["ab"][j])
                ratio = torch.exp(newlp - b["lp"][j])
                a = adv[j]
                pl = -torch.min(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a).mean()
                ent = (dp.entropy() + db.entropy()).mean()
                loss = pl - ent_coef * ent
                opt_a.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                opt_a.step()
                vl = F.mse_loss(critic(b["x"][j]), b["ret"][j])
                opt_c.zero_grad()
                vl.backward()
                opt_c.step()
                kls.append((b["lp"][j] - newlp).mean().item())
                ents.append(ent.item())
            if np.mean(kls[-8:]) > 0.03:
                break
        update += 1
        write("ppo", ent=float(np.mean(ents)), kl=float(np.mean(kls)), vloss=vl.item())
        if update % args.snapshot_every == 0:
            path = os.path.join(args.out, f"snap_{update:04d}.rl.json")
            actor.export(path)
            torch.save(critic.state_dict(), os.path.join(args.out, "critic.pt"))
            snapshots.append(path)
            # League: GA opponents plus the latest few snapshots of ourselves.
            if args.league:
                env.set_pool(list(args.pool) + snapshots[-4:])
    actor.export(os.path.join(args.out, "final.rl.json"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
