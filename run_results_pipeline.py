import subprocess
import sys


STEPS = [
    ("sync_results.py", "sync completed results into results_cache.csv"),
    ("build_completed_round.py", "build latest_completed_round.csv"),
    ("settle_bets.py", "settle completed bets"),
    ("performance.py", "refresh performance and bankroll"),
]


def run_step(script: str, description: str) -> None:
    print(f"\n=== RUN: {script} ===")
    print(f"[info] {description}")
    result = subprocess.run([sys.executable, script], check=False)
    if result.returncode != 0:
        raise SystemExit(f"[stop] {script} failed with exit code {result.returncode}")


def main():
    for script, description in STEPS:
        run_step(script, description)

    print("\n[info] results pipeline completed successfully")


if __name__ == "__main__":
    main()
