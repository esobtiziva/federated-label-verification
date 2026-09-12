"""Automated Flower experiment runner for V3 paired experiments."""
import argparse
import csv
import socket
import subprocess
import sys
import time
from pathlib import Path


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def gamma_slug(gamma: float) -> str:
    return (f"{gamma:g}").replace(".", "p")


def run_one(outdir, iid, nclients, condition, seed, rounds, epochs,
            gamma, budget_total, noise_frac=0.30):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not Path("selected_config.json").exists():
        raise FileNotFoundError("Run python calibrate_baseline.py first")

    port = free_port()
    address = f"127.0.0.1:{port}"
    warmup = 3 if nclients == 2 else 5
    frac = 0.0 if condition == "baseline" else float(noise_frac)

    # V3: B is a GLOBAL LIFETIME budget for the whole FL run, not per round.
    per_client_lifetime = (budget_total // nclients) if condition == "crowd" else 0
    if condition == "crowd" and per_client_lifetime * nclients != budget_total:
        raise ValueError("--crowd-budget-total must be divisible by both 2 and 5")

    mode = "iid" if iid else "noniid"
    if condition == "baseline":
        csv_path = outdir / f"{mode}_{nclients}c_baseline_seed{seed}.csv"
    else:
        csv_path = outdir / f"{mode}_{nclients}c_{condition}_g{gamma_slug(gamma)}_seed{seed}.csv"

    py = sys.executable
    server_cmd = [
        py, "server_final.py", "--rounds", str(rounds),
        "--local-epochs", str(epochs), "--min-fit-clients", str(nclients),
        "--crowd-warmup", str(warmup), "--server", address, "--log", str(csv_path)
    ]
    server = subprocess.Popen(server_cmd)
    time.sleep(2.0)
    clients = []
    t0 = time.perf_counter()
    try:
        for cid in range(nclients):
            cmd = [
                py, "client_final.py", "--client-id", str(cid),
                "--num-clients", str(nclients),
                "--iid" if iid else "--non-iid", "--seed", str(seed),
                "--noise-frac", str(frac), "--noise-gamma", str(gamma),
                "--crowd-budget-lifetime", str(per_client_lifetime),
                "--server", address,
            ]
            clients.append(subprocess.Popen(cmd))
        rc = server.wait()
        if rc != 0:
            raise RuntimeError(f"Server exited with code {rc}")
        for c in clients:
            c.wait(timeout=30)
            if c.returncode not in (0, None):
                raise RuntimeError(f"Client exited with code {c.returncode}")
    finally:
        for proc in clients:
            if proc.poll() is None:
                proc.terminate()
        if server.poll() is None:
            server.terminate()

    wall = time.perf_counter() - t0
    wall_path = outdir / "wall_clock.csv"
    with open(wall_path, "a", newline="") as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(["mode", "clients", "condition", "gamma", "seed", "wall_clock_s"])
        w.writerow([mode, nclients, condition, 0.0 if condition == "baseline" else gamma, seed, wall])
    print(f"DONE {mode} {nclients}c {condition} gamma={gamma:g} seed={seed}: {wall:.3f}s")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="rerun_results_v3")
    p.add_argument("--seeds", default="1")
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--local-epochs", type=int, default=3)
    p.add_argument("--noise-gamma", type=float, default=0.5,
                   help="Gaussian SD = gamma * SD(y_train)")
    p.add_argument("--noise-frac", type=float, default=0.30)
    p.add_argument("--crowd-budget-total", type=int, default=100,
                   help="GLOBAL TOTAL review budget for the entire FL run")
    a = p.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]

    # Within each setting/seed, baseline/noise/crowd are paired by design.
    for iid in (True, False):
        for nclients in (2, 5):
            for seed in seeds:
                for condition in ("baseline", "noise", "crowd"):
                    run_one(a.outdir, iid, nclients, condition, seed,
                            a.rounds, a.local_epochs, a.noise_gamma,
                            a.crowd_budget_total, a.noise_frac)


if __name__ == "__main__":
    main()
