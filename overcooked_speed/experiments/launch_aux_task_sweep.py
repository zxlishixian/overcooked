"""Launch auxiliary task-loss IPPO sweep: 27 runs.

Usage:
    python overcooked_speed/experiments/launch_aux_task_sweep.py
    python overcooked_speed/experiments/launch_aux_task_sweep.py 0
"""
import os
import subprocess
import sys
import time

AUX_TASK_TYPES = ['self_task', 'team_task', 'teammate_task']
AUX_COEFS = [0.001, 0.003, 0.01]
SEEDS = [0, 1, 2]
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


def fmt_float(x):
    return str(x).replace('.', '')


def build_jobs():
    jobs = []
    for aux_type in AUX_TASK_TYPES:
        for coef in AUX_COEFS:
            for seed in SEEDS:
                log_dir = (
                    f'logs/aux_task_sweep/{aux_type}_c{fmt_float(coef)}/'
                    f'seed_{seed}'
                )
                os.makedirs(os.path.join(BASE_DIR, log_dir), exist_ok=True)
                jobs.append((aux_type, coef, seed, log_dir))
    return jobs


def run_chain(gpu_id, job_list):
    n = len(job_list)
    for idx, (aux_type, coef, seed, log_dir) in enumerate(job_list):
        summary_path = os.path.join(BASE_DIR, log_dir, 'summary.json')
        if os.path.exists(summary_path):
            print(f"SKIP [{idx + 1}/{n}] {aux_type} coef={coef} seed={seed} summary exists")
            sys.stdout.flush()
            continue
        cmd = [
            sys.executable, 'overcooked_speed/experiments/run_pair.py',
            '--layout', 'cramped_room',
            '--algo', 'ippo',
            '--obs_mode', 'egocentric',
            '--agent0', 'nl',
            '--agent1', 'nl',
            '--aux_task_loss',
            '--aux_task_type', aux_type,
            '--aux_coef', str(coef),
            '--aux_norm_window', '100',
            '--num_episodes', '300',
            '--seed', str(seed),
            '--log_dir', log_dir,
        ]
        env = os.environ.copy()
        if gpu_id is not None:
            env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        tag = f"[{idx + 1}/{n}] {aux_type} coef={coef} seed={seed}"
        gpu_tag = f"GPU {gpu_id}" if gpu_id is not None else "CPU/auto"
        print(f"START {gpu_tag} {tag}")
        sys.stdout.flush()

        job_t0 = time.time()
        proc = subprocess.Popen(
            cmd, cwd=BASE_DIR, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True,
        )
        deadline = job_t0 + 900
        while proc.poll() is None:
            if time.time() > deadline:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                print(f"TIMEOUT {gpu_tag} {tag}")
                sys.stdout.flush()
                break
            print(f"RUNNING {gpu_tag} {tag} elapsed={(time.time() - job_t0) / 60:.1f} min")
            sys.stdout.flush()
            time.sleep(30)

        stdout, _ = proc.communicate()
        if proc.returncode != 0:
            print(f"FAIL {gpu_tag} {tag} exit={proc.returncode}")
            print(stdout[-1000:])
        else:
            done_line = next(
                (line.strip() for line in stdout.splitlines()
                 if 'Done.' in line),
                'done',
            )
            print(f"DONE {gpu_tag} {tag} {done_line}")
        sys.stdout.flush()


def main():
    jobs = build_jobs()
    print(f"Total jobs: {len(jobs)}")
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else None
    if gpu_id is None:
        job_list = jobs
    else:
        job_list = jobs[gpu_id::2]
    t0 = time.time()
    run_chain(gpu_id, job_list)
    print(f"ALL {len(job_list)} JOBS COMPLETE in {(time.time() - t0) / 60:.1f} min")


if __name__ == '__main__':
    main()
