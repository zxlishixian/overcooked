"""Launch auxiliary task actor-loss sweep: 27 IPPO runs.

Usage:
    python overcooked_speed/experiments/launch_aux_sweep.py
    python overcooked_speed/experiments/launch_aux_sweep.py 0  # optional split
    python overcooked_speed/experiments/launch_aux_sweep.py 1
"""
import os
import subprocess
import sys
import time

AUX_TASK_TYPES = ['self_task', 'team_task', 'teammate_task']
AUX_COEFS = [0.001, 0.003, 0.01]
SEEDS = [0, 1, 2]
NUM_EPISODES = 300
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
SWEEP_DIR = 'logs/aux_task_sweep'
HEARTBEAT_SEC = 30


def coef_tag(coef):
    return str(coef).replace('.', '')


def build_jobs():
    jobs = []
    for aux_type in AUX_TASK_TYPES:
        for coef in AUX_COEFS:
            for seed in SEEDS:
                log_dir = f'{SWEEP_DIR}/{aux_type}_c{coef_tag(coef)}/seed_{seed}'
                jobs.append((aux_type, coef, seed, log_dir))
    return jobs


def run_jobs(jobs, split_label='all'):
    for idx, (aux_type, coef, seed, log_dir) in enumerate(jobs, 1):
        abs_log_dir = os.path.join(BASE_DIR, log_dir)
        os.makedirs(abs_log_dir, exist_ok=True)
        summary_path = os.path.join(abs_log_dir, 'summary.json')
        if os.path.exists(summary_path):
            print(f'SKIP [{split_label}] ({idx}/{len(jobs)}) {aux_type} coef={coef} seed={seed}', flush=True)
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
            '--num_episodes', str(NUM_EPISODES),
            '--seed', str(seed),
            '--log_dir', log_dir,
        ]
        tag = f'[{split_label}] ({idx}/{len(jobs)}) {aux_type} coef={coef} seed={seed}'
        log_path = os.path.join(abs_log_dir, 'run.log')
        print(f'START {tag}', flush=True)
        start = time.time()
        with open(log_path, 'w') as out:
            proc = subprocess.Popen(
                cmd, cwd=BASE_DIR, stdout=out, stderr=subprocess.STDOUT,
                text=True,
            )
            while proc.poll() is None:
                time.sleep(HEARTBEAT_SEC)
                elapsed = (time.time() - start) / 60.0
                print(f'RUNNING {tag} elapsed={elapsed:.1f}min', flush=True)
            rc = proc.returncode

        with open(log_path) as f:
            lines = f.readlines()
        done_line = next((line.strip() for line in lines if line.startswith('Done.')), '')
        if rc != 0:
            print(f'FAIL {tag} exit={rc}', flush=True)
            for line in lines[-20:]:
                print(line.rstrip(), flush=True)
            continue
        print(f'DONE {tag} {done_line}', flush=True)


if __name__ == '__main__':
    jobs = build_jobs()
    split_label = 'all'
    if len(sys.argv) > 1:
        split = int(sys.argv[1])
        jobs = jobs[split::2]
        split_label = str(split)
    print(f'Total jobs for this process: {len(jobs)}', flush=True)
    t0 = time.time()
    run_jobs(jobs, split_label=split_label)
    print(f'ALL COMPLETE [{split_label}] in {(time.time() - t0) / 60:.1f} min', flush=True)
