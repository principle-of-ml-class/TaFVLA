"""Train-only sensor sampling and deterministic FP32 Euclidean K-means++."""
from __future__ import annotations

from collections import defaultdict
import random
import time

import numpy as np
import torch

from utils.live_progress import emit


def _allocate(capacities: list[int], count: int, rng: random.Random) -> list[int]:
    """Balanced quotas, redistributing capacity from short groups."""
    quotas = [0] * len(capacities)
    active = list(range(len(capacities)))
    rng.shuffle(active)
    while count:
        next_active = []
        for i in active:
            if count and quotas[i] < capacities[i]:
                quotas[i] += 1
                count -= 1
            if quotas[i] < capacities[i]:
                next_active.append(i)
        active = next_active
        if count and not active:
            raise ValueError("Insufficient distinct train windows")
    return quotas


def sample_train_windows(dataset, train_indices, sample_count: int, seed: int):
    """Balance sequences, then episodes; stratify across time inside each episode."""
    if sample_count < 1:
        raise ValueError("K-means sample count must be positive")
    rng = random.Random(seed)
    groups = defaultdict(lambda: defaultdict(list))
    identities = set()
    for i in sorted(set(train_indices)):
        if hasattr(dataset, "rows") and dataset.rows[i]["split"] != "train":
            raise ValueError("K-means initialization received a non-train window")
        ep, start = dataset._locate(i)
        sequence = (ep.sensor_type, ep.sequence_id)
        identity = (*sequence, ep.episode_index, start)
        if identity in identities:
            raise ValueError("Duplicate train window identity")
        identities.add(identity)
        groups[sequence][ep.episode_index].append((start, i))
    count = min(sample_count, len(identities))
    seqs = sorted(groups)
    quotas = _allocate([sum(map(len, groups[s].values())) for s in seqs], count, rng)
    selected, sequence_support, episode_support, gaps = [], {}, {}, []
    for seq, quota in zip(seqs, quotas, strict=True):
        episodes = sorted(groups[seq])
        ep_quotas = _allocate([len(groups[seq][e]) for e in episodes], quota, rng)
        if quota:
            sequence_support["/".join(seq)] = quota
        for ep, n in zip(episodes, ep_quotas, strict=True):
            if not n:
                continue
            windows = sorted(groups[seq][ep])
            chosen = [windows[rng.randrange(j*len(windows)//n, (j+1)*len(windows)//n)]
                      for j in range(n)]
            selected.extend(i for _, i in chosen)
            gaps.extend(b[0]-a[0] for a, b in zip(chosen, chosen[1:]))
            episode_support[f"{'/'.join(seq)}/{ep}"] = n
    return selected, {"requested_samples": sample_count, "sample_count": len(selected),
                      "sequence_count": len(sequence_support), "episode_count": len(episode_support),
                      "sequence_support": sequence_support, "episode_support": episode_support,
                      "min_window_start_gap": min(gaps) if gaps else None,
                      "mean_window_start_gap": sum(gaps)/len(gaps) if gaps else None}


def _distances(x, centers):
    return (x.square().sum(1, keepdim=True) + centers.square().sum(1)[None]
            - 2*x @ centers.T).clamp_min_(0)


def _assign(x, centers):
    # Bounded memory even for larger initialization pools.
    distances, labels = [], []
    for batch in x.split(1024):
        d, label = _distances(batch, centers).min(1)
        distances.append(d)
        labels.append(label)
    return torch.cat(distances), torch.cat(labels)


@torch.no_grad()
def kmeans_plus_plus(latents, size: int, iterations: int, seed: int, progress=None):
    """CPU FP32 keeps reductions reproducible and independent of CUDA atomics/AMP."""
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(min(previous_threads, 8))
        with torch.autocast(device_type="cpu", enabled=False):
            return _kmeans_plus_plus(latents, size, iterations, seed, progress)
    finally:
        torch.set_num_threads(previous_threads)


def _kmeans_plus_plus(latents, size: int, iterations: int, seed: int, progress=None):
    x = latents.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if x.ndim != 2 or size < 1 or iterations < 1:
        raise ValueError("K-means requires a 2D latent pool, positive size and iterations")
    x = x[torch.isfinite(x).all(1)]
    if len(x) < size:
        raise ValueError(f"K-means needs >= {size} valid latents, got {len(x)}")
    generator = torch.Generator().manual_seed(seed)
    centers = torch.empty(size, x.shape[1], dtype=torch.float32)
    chosen = torch.zeros(len(x), dtype=torch.bool)
    nearest = torch.full((len(x),), float("inf"))
    index = int(torch.randint(len(x), (), generator=generator))
    for j in range(size):
        centers[j] = x[index]
        chosen[index] = True
        torch.minimum(nearest, (x-centers[j]).square().sum(1), out=nearest)
        nearest[chosen] = 0
        if j+1 < size:
            if nearest.sum() > 0:
                index = int(torch.multinomial(nearest, 1, generator=generator))
            else:
                available = (~chosen).nonzero().flatten()
                index = int(available[torch.randint(len(available), (), generator=generator)])
        if progress and ((j+1) % 64 == 0 or j+1 == size):
            progress("kmeans++", j+1, size)
    empty_reinitializations = 0
    for iteration in range(iterations):
        _, labels = _assign(x, centers)
        counts = torch.bincount(labels, minlength=size)
        sums = torch.zeros_like(centers).index_add_(0, labels, x)
        nonempty = counts > 0
        centers[nonempty] = sums[nonempty] / counts[nonempty, None]
        empty = (~nonempty).nonzero().flatten()
        if len(empty):
            replacements = torch.randperm(len(x), generator=generator)[:len(empty)]
            centers[empty] = x[replacements]
            empty_reinitializations += len(empty)
        if progress:
            progress("lloyd", iteration+1, iterations)
    distance, labels = _assign(x, centers)
    counts = torch.bincount(labels, minlength=size)
    if not torch.isfinite(centers).all():
        raise ValueError("K-means produced non-finite centroids")
    return centers, {"valid_samples": len(x), "quantization_distance_after": float(distance.mean()),
                     "cluster_occupancy": counts.tolist(), "empty_clusters": int((counts == 0).sum()),
                     "empty_cluster_reinitializations": empty_reinitializations}


@torch.no_grad()
def initialize_codebooks(model, dataset, train_indices, options, *, resume: bool = False, batch_size: int = 256):
    method = options.get("initialization", "uniform")
    if resume or method == "uniform":
        emit("Codebook Initialization", "Completed", detail="checkpoint restored" if resume else "uniform")
        return None
    if method != "kmeans":
        raise ValueError(f"Unknown codebook initialization: {method}")
    started = time.monotonic()
    selected, report = sample_train_windows(dataset, train_indices, int(options["kmeans_samples"]),
                                            int(options["kmeans_seed"]))
    required = max(model.force_encoder.pressure.quantizer.codebook.num_embeddings,
                   model.force_encoder.force.quantizer.codebook.num_embeddings)
    if len(selected) < required:
        raise ValueError(f"K-means needs >= {required} distinct train windows, got {len(selected)}")
    emit("Codebook Initialization", total=len(selected), phase="sensor latents")
    device = next(model.force_encoder.parameters()).device
    pools = {name: [] for name in ("pressure", "force")}
    # Only sensor loading and the two MLPs; no image decoder, Sparsh, VQ or decoder.
    with torch.autocast(device_type=device.type, enabled=False):
        for start in range(0, len(selected), batch_size):
            pressure, force = [], []
            for index in selected[start:start+batch_size]:
                ep, offset = dataset._locate(index)
                f, p = dataset._sensor_rows(ep, np.arange(offset, offset+dataset.window_size, dtype=np.int64))
                pressure.append(torch.from_numpy(p))
                force.append(torch.from_numpy(f))
            p, f = model._normalize(torch.stack(pressure).to(device=device, dtype=torch.float32),
                                    torch.stack(force).to(device=device, dtype=torch.float32))
            for name, values in (("pressure", p), ("force", f)):
                branch = getattr(model.force_encoder, name)
                pools[name].append(branch.encoder(values.flatten(1)).float().cpu())
            emit("Codebook Initialization", current=min(start+batch_size, len(selected)),
                 total=len(selected), phase="sensor latents")
    # Validate both branches before copying either codebook.
    centroids = {}
    for name in pools:
        branch = getattr(model.force_encoder, name)
        x = torch.cat(pools[name])
        valid = x[torch.isfinite(x).all(1)]
        before = branch.quantizer.codebook.weight.detach().float().cpu()
        def progress(phase, current, total):
            emit("Codebook Initialization", current=current, total=total, phase=f"{name} {phase}")
        centers, stats = kmeans_plus_plus(x, len(before), int(options["kmeans_iters"]),
                                          int(options["kmeans_seed"]), progress)
        stats.update(quantization_distance_before=float(_assign(valid, before)[0].mean()),
                     codebook_norm_before=float(before.norm(dim=1).mean()),
                     codebook_norm_after=float(centers.norm(dim=1).mean()))
        report[name] = stats
        centroids[name] = centers
    for name, centers in centroids.items():
        getattr(model.force_encoder, name).quantizer.codebook.weight.copy_(centers)
    report.update(method="kmeans", seed=int(options["kmeans_seed"]), iterations=int(options["kmeans_iters"]),
                  distance_metric="squared_euclidean", elapsed_seconds=time.monotonic()-started)
    emit("Codebook Initialization", "Completed", current=len(selected), total=len(selected), detail="both codebooks ready")
    return report
