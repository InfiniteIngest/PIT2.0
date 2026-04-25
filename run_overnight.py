r"""
run_overnight.py -- Piticot Nash experiment orchestrator.

Replaces run_overnight.bat entirely. Pure Python -- no CMD batch quirks,
no phase ordering issues, no output buffering surprises.

HOW TO USE:
    1. Open Command Prompt and cd into that folder:
           cd C:/path/to/pit2.0
    2. Run it:
           python run_overnight.py
       All output prints to the screen AND is saved to run_overnight.log simultaneously.

TO WATCH PROGRESS from a second cmd window while it runs:
    powershell -command "Get-Content run_overnight.log -Wait"

WHAT THIS RUNS (in order, guaranteed):
    Phase 0  -- Calibration          (1 run  x  50,000 ep)  ~  5 min
    Phase 1  -- Baseline learning    (2 runs x 200,000 ep)  ~ 30 min
    Phase 2  -- Core sweep           (8 runs x 200,000 ep)  ~  4 hrs
    Phase 3  -- Reproducibility      (6 runs x 200,000 ep)  ~  3 hrs
    Phase 4  -- Dynamic moral weight (4 runs x 300,000 ep)  ~  3 hrs
    Phase 5  -- Opponent comparison  (4 runs x 200,000 ep)  ~  2 hrs
    Analysis -- Charts and tables    (once at the end)      ~  2 min

OUTPUT:
    data/runs/        -- All CSV logs, Q-tables, metadata
    data/plots/       -- All charts
    run_overnight.log -- Full log of everything printed
"""

import os
import sys
import subprocess
import time
from datetime import datetime

# ── Setup: make sure we are in the right directory ────────────────────────────

if not os.path.exists("requirements.txt") or not os.path.isdir("training"):
    print("[FAIL] Run this script from inside the piticot-nash\\ folder.")
    print(f"       Current directory: {os.getcwd()}")
    sys.exit(1)

os.makedirs("data/runs",  exist_ok=True)
os.makedirs("data/plots", exist_ok=True)

# ── Tee: write to both screen and log file simultaneously ─────────────────────

class Tee:
    """Writes every print() to both the terminal and the log file instantly."""
    def __init__(self, logfile):
        self.terminal = sys.__stdout__
        self.log = open(logfile, "w", encoding="utf-8", buffering=1)  # line-buffered

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Tee("run_overnight.log")
sys.stderr = sys.stdout  # also capture errors

# ── Helpers ───────────────────────────────────────────────────────────────────

def log(msg):
    t = datetime.now().strftime("%H:%M:%S")
    print(f"[{t}] {msg}", flush=True)

def phase(title):
    print(flush=True)
    print("=" * 62, flush=True)
    print(f"  {title}", flush=True)
    print("=" * 62, flush=True)
    print(flush=True)

def run(description, *args, **kwargs):
    """
    Run a subprocess. Streams its output line-by-line so you see progress
    immediately rather than waiting for the process to finish.
    Raises SystemExit if the process returns a non-zero exit code.
    """
    log(description)
    cmd = [sys.executable] + list(args)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        bufsize=1,
    )

    for line in process.stdout:
        print(line, end="", flush=True)

    process.wait()

    if process.returncode != 0:
        print(f"\n[FAIL] '{description}' exited with code {process.returncode}", flush=True)
        print("[FAIL] See the output above for the Python error message.", flush=True)
        print("[FAIL] Fix the issue then re-run.", flush=True)
        sys.exit(process.returncode)

# ── Main experiment sequence ──────────────────────────────────────────────────

start_time = time.time()

log(f"Starting Piticot Nash overnight run from: {os.getcwd()}")
log(f"Python: {sys.version.split()[0]}")
log("All output saved to run_overnight.log")

# ── Setup ─────────────────────────────────────────────────────────────────────

phase("SETUP -- Installing Python dependencies")
run("Installing requirements", "-m", "pip", "install", "-r", "requirements.txt", "--quiet")
log("Dependencies installed.")
log("Output directories ready.")

# ── Sanity check ──────────────────────────────────────────────────────────────

phase("SANITY CHECK -- Verifying all modules import correctly")
run(
    "Importing all project modules",
    "-c",
    "import sys; sys.path.insert(0,'.');"
    "from env.board import resolve_landing;"
    "from env.dice import DiceEngine, DiceAction;"
    "from env.piticot_env import PiticotEnv;"
    "from agents.rl_agent import QAgent;"
    "from agents.random_agent import RandomAgent, UniformRandomAgent;"
    "from reward.reward_fn import RewardConfig, compute_reward;"
    "from analysis.strategy_logger import StrategyLogger;"
    "from training.self_play import run_experiment;"
    "print('All modules imported successfully.')",
)
log("Sanity check passed.")

# ── Phase 0 -- Calibration ────────────────────────────────────────────────────

phase("PHASE 0 -- Calibration (50,000 episodes, pure random baseline)")
run("Calibration run",
    "training/self_play.py",
    "--mode", "vs_random", "--episodes", "50000",
    "--weight", "0.0", "--seed", "42", "--output", "data/runs")
log("Phase 0 complete.")

# ── Phase 1 -- Baseline learning ──────────────────────────────────────────────

phase("PHASE 1 -- Baseline learning (2 runs x 200,000 episodes)")
run("[1/2] self_play, weight=0.0, seed=42",
    "training/self_play.py",
    "--mode", "self_play", "--episodes", "200000",
    "--weight", "0.0", "--seed", "42", "--output", "data/runs")
run("[2/2] vs_random, weight=0.0, seed=42",
    "training/self_play.py",
    "--mode", "vs_random", "--episodes", "200000",
    "--weight", "0.0", "--seed", "42", "--output", "data/runs")
log("Phase 1 complete.")

# ── Phase 2 -- Core moral weight sweep ───────────────────────────────────────

phase("PHASE 2 -- Core moral weight sweep (8 runs x 200,000 episodes)")
for i, weight in enumerate(["0.0", "0.1", "0.3", "0.5", "0.7", "1.0", "1.5", "2.0"]):
    labels = ["fully amoral", "near-amoral", "low guilt", "hesitant",
              "conflicted", "moral", "strong moral", "saintly"]
    run(f"[{i+1}/8] self_play, weight={weight} ({labels[i]}), seed=42",
        "training/self_play.py",
        "--mode", "self_play", "--episodes", "200000",
        "--weight", weight, "--seed", "42", "--output", "data/runs")
log("Phase 2 complete.")

# ── Phase 3 -- Reproducibility ────────────────────────────────────────────────

phase("PHASE 3 -- Reproducibility (6 runs x 200,000 episodes)")
repro_runs = [
    ("0.0", "100"), ("0.0", "200"), ("0.0", "300"),
    ("1.0", "100"), ("1.0", "200"), ("1.0", "300"),
]
for i, (weight, seed) in enumerate(repro_runs):
    run(f"[{i+1}/6] self_play, weight={weight}, seed={seed}",
        "training/self_play.py",
        "--mode", "self_play", "--episodes", "200000",
        "--weight", weight, "--seed", seed, "--output", "data/runs")
log("Phase 3 complete.")

# ── Phase 4 -- Dynamic moral weight ──────────────────────────────────────────

phase("PHASE 4 -- Dynamic moral weight (4 runs x 300,000 episodes)")
run("Running all 4 dynamic schedules (growth, decay, shock, cynicism)",
    "phase4_dynamic.py")
log("Phase 4 complete.")

# ── Phase 5 -- Opponent type comparison ──────────────────────────────────────

phase("PHASE 5 -- Opponent type comparison (4 runs x 200,000 episodes)")
phase5_runs = [
    ("vs_random",  "0.0"), ("vs_uniform", "0.0"),
    ("vs_random",  "1.0"), ("vs_uniform", "1.0"),
]
for i, (mode, weight) in enumerate(phase5_runs):
    run(f"[{i+1}/4] {mode}, weight={weight}, seed=42",
        "training/self_play.py",
        "--mode", mode, "--episodes", "200000",
        "--weight", weight, "--seed", "42", "--output", "data/runs")
log("Phase 5 complete.")

# ── Analysis ──────────────────────────────────────────────────────────────────

phase("ANALYSIS -- Generating charts and summary tables")
run("Running analysis script",
    "notebooks/nash_analysis.py",
    "--data", "data/runs", "--output", "data/plots")
log("Analysis complete. Charts saved to data/plots/")

# ── Done ──────────────────────────────────────────────────────────────────────

elapsed = time.time() - start_time
hours   = int(elapsed // 3600)
minutes = int((elapsed % 3600) // 60)
seconds = int(elapsed % 60)

print(flush=True)
print("=" * 62, flush=True)
print("  ALL PHASES COMPLETE", flush=True)
print("=" * 62, flush=True)
print(flush=True)
print(f"  Total time : {hours}h {minutes}m {seconds}s", flush=True)
print(f"  Run data   : data/runs/", flush=True)
print(f"  Charts     : data/plots/", flush=True)
print(flush=True)
print("  Open data/plots/square24_targeting.png first.", flush=True)
print("  A plateau in the MW=0.0 line means Nash Equilibrium was found.", flush=True)
print(flush=True)
