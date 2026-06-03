"""Train TeammateFuturePredictor on collected trajectories.

Usage (debug):
    python overcooked_speed/experiments/train_future_predictor.py \
        --data_dirs logs/future_pred_data/normal \
        --obs_dim 520 --K 10 --H 10 \
        --epochs 3 --max_samples 10000 --output_dir logs/fp_debug

Usage (full):
    python overcooked_speed/experiments/train_future_predictor.py \
        --data_dirs logs/future_pred_data/normal logs/future_pred_data/pc_true \
        --obs_dim 520 --K 10 --H 10 \
        --epochs 50 --batch_size 128 --lr 1e-3 \
        --max_samples 500000 --output_dir logs/future_predictor
"""
import argparse
import json
import os
import sys
import time
import numpy as np
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from overcooked_speed.agents.teammate_future_predictor import (
    TeammateFuturePredictor, build_history_sequence,
    make_action_target, make_event_target,
)

N_ACTIONS = 6
N_EVENTS = 5
EVENT_KEYS = ['onion_pickup', 'potting_onion', 'dish_pickup',
              'soup_pickup', 'soup_delivery']


# ══════════════════════════════════════════════════════════════════════════════
# Index building
# ══════════════════════════════════════════════════════════════════════════════

def scan_npz_files(data_dirs):
    """Find all trajectories.npz files under given directories.

    Returns list of npz file paths.
    """
    npz_paths = []
    for d in data_dirs:
        for root, dirs, files in os.walk(d):
            for f in files:
                if f == 'trajectories.npz':
                    npz_paths.append(os.path.join(root, f))
    if not npz_paths:
        raise FileNotFoundError(f"No trajectories.npz found under {data_dirs}")
    print(f"Found {len(npz_paths)} npz files")
    return npz_paths


def preload_trajectory_metadata(npz_paths):
    """Load episode lengths from all npz files without decompressing large arrays.

    Uses zipfile to extract only the tiny episode_lengths.npy from each npz.
    Returns dict: npz_path -> {'lengths': array(E,)}
    """
    import zipfile
    import io

    meta = {}
    for npz_path in npz_paths:
        with zipfile.ZipFile(npz_path, 'r') as zf:
            lengths_bytes = zf.read('episode_lengths.npy')
        lengths = np.load(io.BytesIO(lengths_bytes))
        meta[npz_path] = {'lengths': lengths.astype(np.int32)}
    return meta


def build_split_indices(npz_paths, val_frac=0.2, seed=42, metadata=None):
    """Build train/val trajectory lists split by full trajectory.

    Returns train_traj, val_traj: lists of (npz_path, ep_idx, agent_idx, T)
    """
    rng = np.random.RandomState(seed)
    all_trajectories = []

    for npz_path in npz_paths:
        lengths = metadata[npz_path]['lengths'] if metadata else None
        if lengths is None:
            with np.load(npz_path, allow_pickle=False) as data:
                lengths = data['episode_lengths']
        E = len(lengths)
        for ep in range(E):
            T = int(lengths[ep])
            for agent in (0, 1):
                all_trajectories.append((npz_path, ep, agent, T))

    n_traj = len(all_trajectories)
    n_val = max(1, int(n_traj * val_frac))
    idx = rng.permutation(n_traj)
    val_mask = set(idx[:n_val].tolist())

    train_traj = []
    val_traj = []
    for i, traj in enumerate(all_trajectories):
        if i in val_mask:
            val_traj.append(traj)
        else:
            train_traj.append(traj)

    return train_traj, val_traj


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class PredictorDataset(Dataset):
    """Fixed sample indices for deterministic train/val evaluation.

    Samples bucketed by npz file so consecutive __getitem__ calls hit the cache.
    No sort needed — buckets are built and concatenated.
    """

    def __init__(self, trajectory_list, K, H, max_samples=None):
        self.K = K
        self.H = H

        # Shuffle trajectories so we get diverse samples when capping
        rng = np.random.RandomState(42)
        perm = rng.permutation(len(trajectory_list))

        # Bucket samples by npz_path to avoid sorting
        buckets = {}  # npz_path -> list of (ep, agent, t, T)
        total = 0

        for pi in perm:
            if max_samples and total >= max_samples:
                break
            npz_path, ep, agent, T = trajectory_list[pi]
            max_t = T - H
            if max_t <= K:
                continue
            bucket = buckets.setdefault(npz_path, [])
            for t in range(K, max_t):
                bucket.append((ep, agent, t, T))
                total += 1
                if max_samples and total >= max_samples:
                    break

        # Concatenate buckets and build final sample list with path
        self.samples = []
        for npz_path in sorted(buckets.keys()):
            for ep, agent, t, T in buckets[npz_path]:
                self.samples.append((npz_path, ep, agent, t, T))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npz_path, ep, agent, t, T = self.samples[idx]
        # Cache the last npz_path to avoid re-reading (maximized by sorted samples)
        if not hasattr(self, '_cache') or self._cache_key != npz_path:
            self._cache = dict(np.load(npz_path, allow_pickle=False))
            self._cache_key = npz_path

        data = self._cache
        prefix = f'{agent}'
        okey = f'obs_{prefix}'
        akey = f'actions_{prefix}'
        ekey = f'events_{prefix}'

        # History: t-K+1 .. t
        hist_obs = data[okey][ep, t - self.K + 1:t + 1].astype(np.float32)
        hist_act = data[akey][ep, t - self.K + 1:t + 1].astype(np.int64)
        hist_seq = build_history_sequence(hist_obs, hist_act)

        # Future: t+1 .. t+H
        fut_act = data[akey][ep, t + 1:t + self.H + 1].astype(np.int64)
        fut_evt = data[ekey][ep, t + 1:t + self.H + 1].astype(np.float32)

        action_target = make_action_target(fut_act, N_ACTIONS)
        event_target = make_event_target(fut_evt, N_EVENTS)

        return (torch.as_tensor(hist_seq, dtype=torch.float32),
                torch.as_tensor(action_target, dtype=torch.float32),
                torch.as_tensor(event_target, dtype=torch.float32))


# ══════════════════════════════════════════════════════════════════════════════
# Baselines
# ══════════════════════════════════════════════════════════════════════════════

def compute_baseline_metrics(val_dataset, K, H, n_actions=6, n_events=5, max_samples=20000):
    """Evaluate 5 baselines on the validation set.

    Returns dict of per-baseline metrics.
    """
    # Collect all targets and context from val set
    all_action_targets = []
    all_event_targets = []
    all_last_actions = []
    all_K_action_hists = []  # for recent-K histogram baseline
    all_probs_t = []  # current policy at time t

    n_eval = min(len(val_dataset), max_samples)
    for i in range(n_eval):
        npz_path, ep, agent, t, T = val_dataset.samples[i]
        # We need probs at time t and actions in the K-step window
        if not hasattr(val_dataset, '_cache') or val_dataset._cache_key != npz_path:
            val_dataset._cache = dict(np.load(npz_path, allow_pickle=False))
            val_dataset._cache_key = npz_path
        data = val_dataset._cache
        prefix = f'{agent}'
        akey = f'actions_{prefix}'
        pkey = f'probs_{prefix}'
        ekey = f'events_{prefix}'

        hist_act = data[akey][ep, t - K + 1:t + 1].astype(np.int64)
        fut_act = data[akey][ep, t + 1:t + H + 1].astype(np.int64)
        fut_evt = data[ekey][ep, t + 1:t + H + 1].astype(np.float32)

        all_action_targets.append(make_action_target(fut_act, n_actions))
        all_event_targets.append(make_event_target(fut_evt, n_events))
        all_last_actions.append(hist_act[-1])
        all_K_action_hists.append(hist_act)
        all_probs_t.append(data[pkey][ep, t].astype(np.float32))

    action_targets = np.stack(all_action_targets)
    event_targets = np.stack(all_event_targets)
    last_actions = np.array(all_last_actions)
    K_action_hists = np.array(all_K_action_hists)
    probs_t = np.stack(all_probs_t)

    eps = 1e-8
    results = {}

    # Global stats
    global_action_prior = action_targets.mean(axis=0)
    global_event_freq = event_targets.mean(axis=0)

    # 1. Uniform
    uniform_pred = np.ones(n_actions, dtype=np.float32) / n_actions
    results['uniform'] = _compute_action_metrics(
        np.tile(uniform_pred, (len(action_targets), 1)),
        action_targets, n_actions)
    results['uniform']['event_bce'] = float(
        _bce_with_logits(np.log(global_event_freq + eps) - np.log(1 - global_event_freq + eps),
                         event_targets).mean())

    # 2. Global action prior
    prior_pred = np.tile(global_action_prior, (len(action_targets), 1))
    results['global_action_prior'] = _compute_action_metrics(
        prior_pred, action_targets, n_actions)
    results['global_action_prior']['event_bce'] = results['uniform']['event_bce']

    # 3. Last action (one-hot)
    last_pred = np.zeros((len(action_targets), n_actions), dtype=np.float32)
    last_pred[np.arange(len(action_targets)), last_actions] = 1.0
    results['last_action'] = _compute_action_metrics(last_pred, action_targets, n_actions)
    results['last_action']['event_bce'] = results['uniform']['event_bce']

    # 4. Recent-K action histogram
    K_hist_pred = np.zeros((len(action_targets), n_actions), dtype=np.float32)
    for i in range(len(action_targets)):
        for a in K_action_hists[i]:
            K_hist_pred[i, a] += 1.0
    K_hist_pred /= K
    results['recent_K_histogram'] = _compute_action_metrics(K_hist_pred, action_targets, n_actions)
    results['recent_K_histogram']['event_bce'] = results['uniform']['event_bce']

    # 5. Current policy predictor (key baseline)
    results['current_policy'] = _compute_action_metrics(probs_t, action_targets, n_actions)
    results['current_policy']['event_bce'] = results['uniform']['event_bce']

    return results, global_action_prior, global_event_freq


def _bce_with_logits(logits, targets):
    """Numerically stable BCE with logits, returns per-sample loss."""
    # BCE = -[t * log(sigmoid(l)) + (1-t) * log(1 - sigmoid(l))]
    #      = l - l*t + log(1 + exp(-l))
    # Use: max(l,0) - l*t + log(1 + exp(-abs(l)))
    max_l = np.maximum(logits, 0)
    loss = max_l - logits * targets + np.log(1 + np.exp(-np.abs(logits)))
    return loss.mean(axis=-1)


def _compute_action_metrics(pred, targets, n_actions):
    eps = 1e-8
    ce = -(targets * np.log(pred + eps)).sum(axis=-1).mean()
    kl = (targets * (np.log(targets + eps) - np.log(pred + eps))).sum(axis=-1).mean()
    top1_acc = (pred.argmax(axis=-1) == targets.argmax(axis=-1)).mean()
    mse = ((pred - targets) ** 2).mean()

    per_action = {}
    for a in range(n_actions):
        per_action[f'ce_action_{a}'] = float(-(targets[:, a] * np.log(pred[:, a] + eps)).mean())
        per_action[f'kl_action_{a}'] = float((targets[:, a] * (np.log(targets[:, a] + eps) - np.log(pred[:, a] + eps))).mean())

    return {
        'ce': float(ce), 'kl': float(kl), 'top1_acc': float(top1_acc),
        'mse': float(mse), **per_action,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def file_sequential_batches(dataset, batch_size, device, shuffle=True):
    """Generator: yield all batches for ONE epoch, loading one npz file at a time.

    Each npz file is loaded once, its samples are shuffled (within file),
    and yielded as batches. Then the file is released and the next file loaded.
    Yields exactly one epoch's worth of batches, then stops.
    """
    import torch

    # Map npz_path -> list of sample indices
    file_to_indices = {}
    for i, (npz_path, ep, agent, t, T) in enumerate(dataset.samples):
        file_to_indices.setdefault(npz_path, []).append(i)

    file_paths = list(file_to_indices.keys())
    rng = np.random.RandomState()

    if shuffle:
        rng.shuffle(file_paths)

    for npz_path in file_paths:
        data = dict(np.load(npz_path, allow_pickle=False))
        indices = file_to_indices[npz_path]
        if shuffle:
            rng.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start:start + batch_size]
            batch_samples = [dataset.samples[i] for i in batch_idx]
            hist_seqs = []
            act_targets = []
            evt_targets = []

            for npz_path_b, ep, agent, t, T in batch_samples:
                prefix = f'{agent}'
                okey = f'obs_{prefix}'
                akey = f'actions_{prefix}'
                ekey = f'events_{prefix}'
                K = dataset.K
                H = dataset.H

                hist_obs = data[okey][ep, t - K + 1:t + 1].astype(np.float32)
                hist_act = data[akey][ep, t - K + 1:t + 1].astype(np.int64)
                hist_seq = build_history_sequence(hist_obs, hist_act)

                fut_act = data[akey][ep, t + 1:t + H + 1].astype(np.int64)
                fut_evt = data[ekey][ep, t + 1:t + H + 1].astype(np.float32)

                action_target = make_action_target(fut_act, N_ACTIONS)
                event_target = make_event_target(fut_evt, N_EVENTS)

                hist_seqs.append(torch.as_tensor(hist_seq, dtype=torch.float32))
                act_targets.append(torch.as_tensor(action_target, dtype=torch.float32))
                evt_targets.append(torch.as_tensor(event_target, dtype=torch.float32))

            yield (torch.stack(hist_seqs).to(device),
                   torch.stack(act_targets).to(device),
                   torch.stack(evt_targets).to(device))


def compute_pos_weight(train_dataset, K, H, max_samples=50000):
    """Compute pos_weight for BCE event loss from a subset of training data."""
    pos_counts = np.zeros(N_EVENTS)
    neg_counts = np.zeros(N_EVENTS)
    n = min(max_samples, len(train_dataset))
    indices = np.random.RandomState(42).choice(len(train_dataset), n, replace=False)
    indices = np.sort(indices)  # Sort for cache locality

    for idx in indices:
        npz_path, ep, agent, t, T = train_dataset.samples[idx]
        if not hasattr(train_dataset, '_cache') or train_dataset._cache_key != npz_path:
            train_dataset._cache = dict(np.load(npz_path, allow_pickle=False))
            train_dataset._cache_key = npz_path
        data = train_dataset._cache
        ekey = f'events_{agent}'
        fut_evt = data[ekey][ep, t + 1:t + H + 1].astype(np.float32)
        event_vec = np.any(fut_evt, axis=0).astype(np.float32)
        pos_counts += event_vec
        neg_counts += (1 - event_vec)

    pos_weight = np.clip(neg_counts / (pos_counts + 1e-8), 1.0, 20.0)
    return pos_counts, neg_counts, pos_weight


def train_predictor(train_set, val_set, obs_dim, K, H, args):
    """Train TeammateFuturePredictor and return results."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}")

    # Compute pos_weight
    pos_counts, neg_counts, pos_weight = compute_pos_weight(train_set, K, H)
    print(f"Event positive rates: {pos_counts / (pos_counts + neg_counts + 1e-8)}")
    print(f"pos_weight: {pos_weight}")

    model = TeammateFuturePredictor(
        obs_dim=obs_dim, n_actions=N_ACTIONS, n_events=N_EVENTS,
        hidden_dim=args.hidden_dim, latent_dim=args.latent_dim)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    pw_t = torch.as_tensor(pos_weight, dtype=torch.float32, device=device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pw_t)

    # File-sequential batches: each npz file loaded once per epoch
    # train_gen is created fresh per epoch inside the loop

    # Count batches for val (single pass)
    val_batches = 0
    for _ in file_sequential_batches(val_set, args.batch_size * 2, device, shuffle=False):
        val_batches += 1

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_action_loss = 0.0
        train_event_loss = 0.0
        n_batches = 0

        train_gen = file_sequential_batches(train_set, args.batch_size, device)
        for hist_seq, act_target, evt_target in train_gen:

            action_pred, event_logits, z = model(hist_seq)

            # Action loss: CE with soft target
            eps = 1e-8
            action_loss = -(act_target * torch.log(action_pred + eps)).sum(-1).mean()

            # Event loss: BCE
            event_loss = bce_loss(event_logits, evt_target)

            loss = action_loss + args.lambda_event * event_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_action_loss += action_loss.item()
            train_event_loss += event_loss.item()
            n_batches += 1

        avg_train_loss = train_loss / n_batches
        avg_train_action = train_action_loss / n_batches
        avg_train_event = train_event_loss / n_batches

        # Validation (fresh generator each epoch for single pass)
        model.eval()
        val_loss = 0.0
        val_action_loss = 0.0
        val_event_loss = 0.0
        n_val = 0
        val_gen_epoch = file_sequential_batches(val_set, args.batch_size * 2, device, shuffle=False)
        with torch.no_grad():
            for hist_seq, act_target, evt_target in val_gen_epoch:
                hist_seq = hist_seq.to(device)
                act_target = act_target.to(device)
                evt_target = evt_target.to(device)

                action_pred, event_logits, z = model(hist_seq)

                action_loss = -(act_target * torch.log(action_pred + eps)).sum(-1).mean()
                event_loss = bce_loss(event_logits, evt_target)
                loss = action_loss + args.lambda_event * event_loss

                val_loss += loss.item()
                val_action_loss += action_loss.item()
                val_event_loss += event_loss.item()
                n_val += 1

        avg_val_loss = val_loss / n_val
        avg_val_action = val_action_loss / n_val
        avg_val_event = val_event_loss / n_val

        history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'train_action_loss': avg_train_action,
            'train_event_loss': avg_train_event,
            'val_loss': avg_val_loss,
            'val_action_loss': avg_val_action,
            'val_event_loss': avg_val_event,
        })

        print(f"Epoch {epoch:3d}: train_loss={avg_train_loss:.4f} "
              f"(act={avg_train_action:.4f} evt={avg_train_event:.4f}) "
              f"val_loss={avg_val_loss:.4f} "
              f"(act={avg_val_action:.4f} evt={avg_val_event:.4f})")

        if avg_val_loss < best_val_loss - 1e-6:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    print(f"Best epoch: {best_epoch}, best val_loss: {best_val_loss:.4f}")

    # Latent diagnostics
    model.eval()
    all_z = []
    z_gen = file_sequential_batches(val_set, args.batch_size * 2, device, shuffle=False)
    with torch.no_grad():
        for hist_seq, _, _ in z_gen:
            hist_seq = hist_seq.to(device)
            z = model.encode(hist_seq)
            all_z.append(z.cpu().numpy())
    z_all = np.concatenate(all_z, axis=0)
    latent_mean = float(z_all.mean())
    latent_std = float(z_all.std())
    latent_norms = np.linalg.norm(z_all, axis=-1)
    latent_norm_mean = float(latent_norms.mean())
    latent_std_mean = float(z_all.std(axis=0).mean())

    print(f"Latent: mean={latent_mean:.4f} std={latent_std:.4f} "
          f"norm_mean={latent_norm_mean:.4f} std_mean={latent_std_mean:.6f}")

    return model, best_state, history, {
        'latent_mean': latent_mean,
        'latent_std': latent_std,
        'latent_norm_mean': latent_norm_mean,
        'latent_std_mean': latent_std_mean,
        'pos_weight': pos_weight.tolist(),
        'event_pos_rate': (pos_counts / (pos_counts + neg_counts + 1e-8)).tolist(),
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_model(model, val_set, batch_size, device):
    """Compute detailed metrics for the trained GRU model on val set."""
    model.eval()
    all_act_pred = []
    all_act_target = []
    all_evt_pred = []
    all_evt_target = []

    val_gen = file_sequential_batches(val_set, batch_size, device, shuffle=False)
    for hist_seq, act_target, evt_target in val_gen:
        hist_seq = hist_seq.to(device)
        action_pred, event_logits, _ = model(hist_seq)
        all_act_pred.append(action_pred.cpu().numpy())
        all_act_target.append(act_target.cpu().numpy())
        all_evt_pred.append(torch.sigmoid(event_logits).cpu().numpy())
        all_evt_target.append(evt_target.cpu().numpy())

    act_pred = np.concatenate(all_act_pred, axis=0)
    act_target = np.concatenate(all_act_target, axis=0)
    evt_pred = np.concatenate(all_evt_pred, axis=0)
    evt_target = np.concatenate(all_evt_target, axis=0)

    metrics = _compute_action_metrics(act_pred, act_target, N_ACTIONS)

    # Event metrics
    evt_bce = float(_bce_with_logits(
        np.log(np.clip(evt_pred, 1e-8, 1 - 1e-8)) - np.log(np.clip(1 - evt_pred, 1e-8, 1 - 1e-8)),
        evt_target).mean())
    metrics['event_bce'] = evt_bce

    for ei, ek in enumerate(EVENT_KEYS):
        precision = (evt_pred[:, ei] > 0.5).astype(float)
        tp = ((precision == 1) & (evt_target[:, ei] == 1)).sum()
        fp = ((precision == 1) & (evt_target[:, ei] == 0)).sum()
        fn = ((precision == 0) & (evt_target[:, ei] == 1)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        metrics[f'event_{ek}_f1'] = float(f1)
        metrics[f'event_{ek}_precision'] = float(prec)
        metrics[f'event_{ek}_recall'] = float(rec)

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dirs', nargs='+', required=True,
                        help='Directories containing trajectories.npz files')
    parser.add_argument('--obs_dim', type=int, required=True,
                        help='Observation dimension (e.g. 520 for ego)')
    parser.add_argument('--K', type=int, default=10,
                        help='History window size')
    parser.add_argument('--H', type=int, default=10,
                        help='Future prediction horizon')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--lambda_event', type=float, default=1.0,
                        help='Event loss weight')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Cap total train samples (debug: 10000)')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', default='logs/future_predictor')
    parser.add_argument('--no_train', action='store_true', default=False,
                        help='Skip training, only evaluate baselines')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Build index
    npz_paths = scan_npz_files(args.data_dirs)
    metadata = preload_trajectory_metadata(npz_paths)
    train_traj, val_traj = build_split_indices(npz_paths, metadata=metadata)

    # 2. Create datasets (with early capping for efficiency)
    train_max = args.max_samples if args.max_samples else None
    val_max = min(100000, args.max_samples // 5) if args.max_samples else None
    train_set = PredictorDataset(train_traj, args.K, args.H, max_samples=train_max)
    val_set = PredictorDataset(val_traj, args.K, args.H, max_samples=val_max)
    print(f"Train trajectories: {len(train_traj)}, Val trajectories: {len(val_traj)}")
    print(f"Train timestep-windows: {len(train_set)}, Val timestep-windows: {len(val_set)}")

    # 3. Baselines
    print("\n" + "=" * 60)
    print("Computing baselines...")
    baseline_metrics, global_action_prior, global_event_freq = \
        compute_baseline_metrics(val_set, args.K, args.H)
    for name, m in baseline_metrics.items():
        print(f"  {name:25s}: CE={m['ce']:.4f}  KL={m['kl']:.4f}  top1={m['top1_acc']:.4f}")

    # 4. Train
    if not args.no_train:
        print("\n" + "=" * 60)
        print("Training GRU predictor...")
        model, best_state, history, train_diag = train_predictor(
            train_set, val_set, args.obs_dim, args.K, args.H, args)

        # 5. Evaluate model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_metrics = evaluate_model(model, val_set, args.batch_size * 2, device)

        # 6. Save
        torch.save({
            'model_state_dict': best_state,
            'config': {
                'obs_dim': args.obs_dim, 'K': args.K, 'H': args.H,
                'hidden_dim': args.hidden_dim, 'latent_dim': args.latent_dim,
                'n_actions': N_ACTIONS, 'n_events': N_EVENTS,
            },
            'pos_weight': train_diag['pos_weight'],
            'train_diag': train_diag,
        }, os.path.join(args.output_dir, 'model_best.pt'))

        results = {
            'config': vars(args),
            'baselines': baseline_metrics,
            'gru_model': model_metrics,
            'train_history': history,
            'train_diag': train_diag,
            'global_action_prior': global_action_prior.tolist(),
            'global_event_freq': global_event_freq.tolist(),
        }
    else:
        results = {
            'config': vars(args),
            'baselines': baseline_metrics,
            'global_action_prior': global_action_prior.tolist(),
            'global_event_freq': global_event_freq.tolist(),
        }

    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # 7. Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY: Action Prediction")
    print(f"{'Predictor':<30} {'CE':>8} {'KL':>8} {'Top-1':>8} {'MSE':>8}")
    print("-" * 60)
    for name in ['uniform', 'global_action_prior', 'last_action',
                 'recent_K_histogram', 'current_policy']:
        m = baseline_metrics[name]
        print(f"{name:<30} {m['ce']:>8.4f} {m['kl']:>8.4f} {m['top1_acc']:>8.4f} {m['mse']:>8.4f}")
    if not args.no_train:
        m = results['gru_model']
        print(f"{'GRU (ours)':<30} {m['ce']:>8.4f} {m['kl']:>8.4f} {m['top1_acc']:>8.4f} {m['mse']:>8.4f}")

    # Success criteria
    print("\n" + "=" * 60)
    print("SUCCESS CRITERIA")
    current_policy_ce = baseline_metrics['current_policy']['ce']
    recent_K_ce = baseline_metrics['recent_K_histogram']['ce']
    if not args.no_train:
        gru_ce = model_metrics['ce']
        improved_vs_cp = (current_policy_ce - gru_ce) / current_policy_ce * 100
        improved_vs_rk = (recent_K_ce - gru_ce) / recent_K_ce * 100
        print(f"  GRU vs current_policy: {improved_vs_cp:+.1f}% (need >=5%)  "
              f"{'PASS' if improved_vs_cp >= 5 else 'FAIL'}")
        print(f"  GRU vs recent_K: {improved_vs_rk:+.1f}% (need >=5%)  "
              f"{'PASS' if improved_vs_rk >= 5 else 'FAIL'}")
        std_mean = train_diag['latent_std_mean']
        print(f"  latent_std_mean: {std_mean:.6f} (need >0.01)  "
              f"{'PASS' if std_mean > 0.01 else 'FAIL'}")
    else:
        print("  (run without --no_train to see GRU results)")

    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()
