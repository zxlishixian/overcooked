# Overcooked Speed — Convergence & Specialization Framework

import subprocess
import os


# ── GPU utilities ──────────────────────────────────────────────────────────

def get_free_gpus(max_gpus=2, min_free_memory_mb=500, max_utilization=5):
    """Return list of free GPU indices safe to use.

    Criteria:
    - GPU utilization <= max_utilization (%)
    - Free memory >= min_free_memory_mb
    - Excludes GPUs already listed in CUDA_VISIBLE_DEVICES

    Returns at most max_gpus indices. Returns empty list if none available.
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    already_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if already_visible:
        excluded = {int(x.strip()) for x in already_visible.split(',') if x.strip()}
    else:
        excluded = set()

    free = []
    for line in result.stdout.strip().split('\n'):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 4:
            continue
        idx = int(parts[0])
        util = int(parts[1])
        mem_used = int(parts[2])
        mem_total = int(parts[3])
        free_mem = mem_total - mem_used

        if idx in excluded:
            continue
        if util <= max_utilization and free_mem >= min_free_memory_mb:
            free.append(idx)

        if len(free) >= max_gpus:
            break

    return free


# ── Custom layouts ─────────────────────────────────────────────────────────

# Custom layouts are defined here as (grid_lines, params) tuples.
# grid_lines: list of equal-length strings using Overcooked grid characters:
#   X = counter, ' ' = floor, O = onion dispenser, P = pot
#   D = dish dispenser, S = serving, 1-9 = player start positions

CUSTOM_LAYOUTS = {
    'custom_asymmetric_roles': {
        'grid': [
            "XXXXXXXX",  # counters
            "X S   PX",  # serving far-left, pot far-right
            "X 1 2  X",  # p1 left-of-center (→delivery), p2 right-of-center (→cooking)
            "X D   OX",  # dish far-left, onion far-right
            "XXXXXXXX",  # counters
        ],
        'start_all_orders': [
            {"ingredients": ["onion", "onion", "onion"]},
        ],
        'start_bonus_orders': [],
        'rew_shaping_params': None,
    },
}


def get_layout_spec(layout_name):
    """Return (is_custom, spec) for a layout name.

    If layout_name is in CUSTOM_LAYOUTS, returns (True, layout_dict).
    Otherwise returns (False, layout_name) for filesystem lookup.
    """
    if layout_name in CUSTOM_LAYOUTS:
        return (True, CUSTOM_LAYOUTS[layout_name])
    return (False, layout_name)
