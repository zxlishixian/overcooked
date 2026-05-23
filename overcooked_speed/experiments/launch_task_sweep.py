"""Launch task-level shaping sweep: 72 runs distributed across 2 GPUs.
Simple sequential launcher — one Python process per GPU.
"""
import subprocess
import os
import sys

SHAPING_TYPES = [
    'self_task_progress',
    'team_task_progress',
    'teammate_task_progress',
    'task_lookahead_rule',
    'task_lola_rule',
    'team_bottleneck_progress',
]
LAMBDAS = [0.03, 0.1]
CLIPS = [3.0, 5.0]
SEEDS = [0, 1, 2]

BASE_DIR = os.path.join(os.path.dirname(__file__), '../..')

# Build all jobs
jobs = []
for st in SHAPING_TYPES:
    for lam in LAMBDAS:
        for clip in CLIPS:
            for seed in SEEDS:
                log_dir = f'logs/task_shaping_sweep/{st}_l{str(lam).replace(".","")}_c{str(clip).replace(".","")}/seed_{seed}'
                os.makedirs(os.path.join(BASE_DIR, log_dir), exist_ok=True)
                jobs.append((st, lam, clip, seed, log_dir))

print(f"Total jobs: {len(jobs)}")
sys.stdout.flush()

# Split across GPUs
gpu0_jobs = jobs[::2]
gpu1_jobs = jobs[1::2]


def run_chain(gpu_id, job_list):
    """Run jobs sequentially on a specific GPU."""
    n = len(job_list)
    for idx, (st, lam, clip, seed, log_dir) in enumerate(job_list):
        cmd = (
            f'conda run -n overcooked python overcooked_speed/experiments/run_pair.py '
            f'--layout cramped_room --algo ippo --obs_mode egocentric '
            f'--agent0 nl --agent1 nl '
            f'--shaping_type {st} '
            f'--lambda_role {lam} '
            f'--bonus_clip {clip} '
            f'--role_window 20 '
            f'--num_episodes 300 '
            f'--seed {seed} '
            f'--log_dir {log_dir}'
        )
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        tag = f"[GPU {gpu_id}] ({idx+1}/{n}) {st} lam={lam} clip={clip} seed={seed}"
        print(f"  START {tag}")
        sys.stdout.flush()

        result = subprocess.run(
            cmd, shell=True, cwd=BASE_DIR, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, timeout=600  # 10 min timeout per job
        )

        if result.returncode != 0:
            print(f"  FAIL {tag} (exit={result.returncode})")
            print(f"  Last output: {result.stdout[-300:]}")
            sys.stdout.flush()
        else:
            # Extract final line
            for line in result.stdout.strip().split('\n'):
                if 'Done.' in line:
                    print(f"  DONE {tag}  {line.strip()}")
                    break
            else:
                print(f"  DONE {tag}")
            sys.stdout.flush()


if __name__ == '__main__':
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    job_list = gpu0_jobs if gpu_id == 0 else gpu1_jobs
    print(f"GPU {gpu_id}: {len(job_list)} jobs")
    sys.stdout.flush()

    import time
    t0 = time.time()
    run_chain(gpu_id, job_list)
    elapsed = time.time() - t0
    print(f"GPU {gpu_id}: ALL {len(job_list)} JOBS COMPLETE in {elapsed/60:.1f} min")
