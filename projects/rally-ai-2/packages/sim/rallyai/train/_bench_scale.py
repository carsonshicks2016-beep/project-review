"""Internal scale probe — run via ``python -m rallyai.train._bench_scale``.

Prints SyncVecEnv n∈{1,2,4,8,16} and AsyncVecEnv worker scaling against the
single-process baseline. B2 Done criterion: async/8 ≥ 5× sync/1.
"""

from __future__ import annotations

from rallyai.train.bench import bench_async, bench_sync


def main() -> None:
    print("=== SyncVecEnv ===")
    sync_rates: dict[int, float] = {}
    for n in (1, 2, 4, 8, 16):
        steps = 4000 if n <= 4 else n * 300
        r = bench_sync(n, steps, 0)
        sync_rates[n] = r["steps_per_s"]
        print(f"n={n:2d}  {r['steps_per_s']:8.1f} steps/s  wall={r['wall_s']:.2f}s")

    baseline = sync_rates[1]
    print("=== AsyncVecEnv ===")
    for workers in (1, 2, 4, 8, 12):
        steps = 8000 if workers < 8 else 20000
        async_ = bench_async(workers, steps, 0)
        ratio = async_["steps_per_s"] / baseline
        flag = " OK" if workers == 8 and ratio >= 5.0 else ""
        print(
            f"w={workers:2d}  {async_['steps_per_s']:8.1f} steps/s  "
            f"x{ratio:.2f} vs sync/1{flag}"
        )


if __name__ == "__main__":
    main()
