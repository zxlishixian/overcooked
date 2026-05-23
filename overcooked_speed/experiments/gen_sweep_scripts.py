"""Generate per-GPU shell scripts for the task shaping sweep."""
import os

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
BASE = '/home/lishixian/research/lola/overcooked'

jobs = []
for st in SHAPING_TYPES:
    for lam in LAMBDAS:
        for clip in CLIPS:
            for seed in SEEDS:
                log_dir = f'logs/task_shaping_sweep/{st}_l{str(lam).replace(".","")}_c{str(clip).replace(".","")}/seed_{seed}'
                jobs.append((st, lam, clip, seed, log_dir))

def gen_script(gpu_id, job_list, out_path):
    lines = ['#!/bin/bash', 'set -e', f'cd {BASE}', '']
    for i, (st, lam, clip, seed, log_dir) in enumerate(job_list):
        tag = f'[{i+1}/{len(job_list)}] {st} lam={lam} clip={clip} seed={seed}'
        lines.append(f'mkdir -p {log_dir}')
        lines.append(f'echo "START GPU{gpu_id} {tag}"')
        lines.append(
            f'CUDA_VISIBLE_DEVICES={gpu_id} conda run -n overcooked python '
            f'overcooked_speed/experiments/run_pair.py '
            f'--layout cramped_room --algo ippo --obs_mode egocentric '
            f'--agent0 nl --agent1 nl '
            f'--shaping_type {st} --lambda_role {lam} --bonus_clip {clip} '
            f'--role_window 20 --num_episodes 300 --seed {seed} '
            f'--device 0 --log_dir {log_dir} || '
            f'echo "FAIL GPU{gpu_id} {tag}"'
        )
        lines.append(f'echo "DONE  GPU{gpu_id} {tag}"')
        lines.append('')
    lines.append(f'echo "GPU{gpu_id} ALL {len(job_list)} JOBS COMPLETE"')

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))
    os.chmod(out_path, 0o755)
    print(f'Generated {out_path} with {len(job_list)} jobs')


if __name__ == '__main__':
    # Even indices → GPU 0, odd → GPU 1
    gpu0_jobs = jobs[::2]
    gpu1_jobs = jobs[1::2]

    gen_script(0, gpu0_jobs, f'{BASE}/logs/sweep_gpu0.sh')
    gen_script(1, gpu1_jobs, f'{BASE}/logs/sweep_gpu1.sh')
    print(f'\nRun with:')
    print(f'  bash {BASE}/logs/sweep_gpu0.sh &')
    print(f'  bash {BASE}/logs/sweep_gpu1.sh &')
