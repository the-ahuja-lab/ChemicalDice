"""
density_aware_sampling.py  v4  — with checkpointing (resume from last step)
─────────────────────────────────────────────────────────────────────────────
Checkpoints saved after each step:
  scratch/fp_uint8.mmap        → step 1 done
  scratch/umap_30d.mmap        → step 2 done
  scratch/hdbscan_labels.npy   → step 3 done

On re-run, completed steps are skipped automatically.

Install:
    pip install umap-learn pynndescent hdbscan h5py tqdm

Usage:
    python density_aware_sampling.py --target 0.10 \
        --umap_subsample 100000 --min_cluster_size 50 \
        --work_dir /path/to/scratch --out_dir /path/to/results
"""

import argparse, time, gc, os, warnings, tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch

warnings.filterwarnings("ignore")

INPUT_FILE          = "chembl_35_ecfp6.csv"
ECFP_COL            = "ECFP6"
TARGET              = 0.10
TILE_SIZE           = 30_000
UMAP_DIMS           = 30
UMAP_NEIGHBORS      = 30
UMAP_FIT_SUBSAMPLE  = 100_000
HDBSCAN_MIN_CLUSTER = 50
HDBSCAN_MIN_SAMPLES = 10
OUT_DIR             = Path(".")
WORK_DIR            = Path(tempfile.gettempdir()) / "density_work"
SEED                = 42


# ══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def ckpt(work_dir: Path, name: str) -> Path:
    return work_dir / name

def step_done(work_dir: Path, name: str) -> bool:
    p = ckpt(work_dir, name)
    if p.exists():
        print(f"  [SKIP] {name} already exists — resuming from checkpoint.")
        return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# 1.  FINGERPRINT MEMMAP
# ══════════════════════════════════════════════════════════════════════════════
def build_fp_memmap(series: pd.Series, work_dir: Path):
    path = work_dir / "fp_uint8.mmap"
    meta = work_dir / "fp_meta.npz"

    if path.exists() and meta.exists():
        m   = np.load(meta)
        N, fp_len = int(m["N"]), int(m["fp_len"])
        valid_mask = m["valid_mask"]
        fp_mm = np.memmap(path, dtype=np.uint8, mode="r", shape=(N, fp_len))
        print(f"  [SKIP] fp_uint8.mmap exists  ({N:,} × {fp_len})")
        return fp_mm, valid_mask, path

    valid_mask = series.notna() & series.astype(str).str.len().gt(0)
    n_bad = (~valid_mask).sum()
    if n_bad:
        print(f"  [WARNING] Dropping {n_bad:,} rows with null ECFP6.")
    valid  = series[valid_mask]
    N      = len(valid)
    fp_len = len(str(valid.iloc[0]).strip())

    fp_mm  = np.memmap(path, dtype=np.uint8, mode="w+", shape=(N, fp_len))
    CHUNK  = 50_000
    print(f"  Writing uint8 memmap {N:,}×{fp_len} ...")
    for s in tqdm(range(0, N, CHUNK), desc="  Parsing", unit="chunk", ncols=80):
        e = min(s + CHUNK, N)
        fp_mm[s:e] = np.vstack(
            valid.iloc[s:e].apply(
                lambda x: np.frombuffer(str(x).encode(), dtype=np.uint8) - ord("0")
            ).values
        ).astype(np.uint8)
    fp_mm.flush()
    np.savez(meta, N=N, fp_len=fp_len, valid_mask=valid_mask.values)
    print(f"  Memmap: {path.stat().st_size/1e9:.2f} GB")
    return fp_mm, valid_mask.values, path


# ══════════════════════════════════════════════════════════════════════════════
# 2.  GPU TANIMOTO
# ══════════════════════════════════════════════════════════════════════════════
def tanimoto_block(A_u8: torch.Tensor, B_u8: torch.Tensor) -> torch.Tensor:
    A = A_u8.float(); B = B_u8.float()
    inter = torch.mm(A, B.T)
    union = torch.clamp(
        A.sum(1, keepdim=True) + B.sum(1, keepdim=True).T - inter, min=1e-8)
    return inter / union


# ══════════════════════════════════════════════════════════════════════════════
# 3.  UMAP  (CPU, jaccard, fit-on-subsample + transform-rest)
# ══════════════════════════════════════════════════════════════════════════════
def run_umap(fp_mm, n_components, n_neighbors, fit_subsample,
             work_dir: Path, seed: int):
    try:
        import umap as umap_lib
    except ImportError:
        raise ImportError("pip install umap-learn pynndescent")

    N        = fp_mm.shape[0]
    emb_path = work_dir / f"umap_{n_components}d.mmap"
    meta     = work_dir / "umap_meta.npz"

    if emb_path.exists() and meta.exists():
        m      = np.load(meta)
        emb_mm = np.memmap(emb_path, dtype=np.float32, mode="r",
                           shape=(N, n_components))
        print(f"  [SKIP] umap_{n_components}d.mmap exists  shape={emb_mm.shape}")
        return emb_mm, emb_path
    exit()
    fit_n   = min(fit_subsample, N)
    rng     = np.random.default_rng(seed)
    fit_idx = np.sort(rng.choice(N, fit_n, replace=False))
    print(f"  Fitting UMAP on {fit_n:,} mols  "
          f"(n_components={n_components}, n_neighbors={n_neighbors}, "
          f"metric=jaccard) ...")

    fp_sub  = np.array(fp_mm[fit_idx], dtype=np.float32)
    t0      = time.time()
    reducer = umap_lib.UMAP(
        n_components=n_components, n_neighbors=n_neighbors,
        metric="jaccard", random_state=seed,
        verbose=True, low_memory=True, n_jobs=-1,
    )
    emb_sub = reducer.fit_transform(fp_sub).astype(np.float32)
    print(f"  Fit done in {time.time()-t0:.0f}s")
    del fp_sub; gc.collect()

    emb_mm = np.memmap(emb_path, dtype=np.float32, mode="w+",
                       shape=(N, n_components))
    emb_mm[fit_idx] = emb_sub
    del emb_sub; gc.collect()

    rest_idx = np.setdiff1d(np.arange(N), fit_idx)
    if len(rest_idx) > 0:
        print(f"  Transforming {len(rest_idx):,} remaining mols ...")
        BATCH = 50_000
        for s in tqdm(range(0, len(rest_idx), BATCH),
                      desc="  Transform", unit="batch", ncols=80):
            e     = min(s + BATCH, len(rest_idx))
            idx_b = rest_idx[s:e]
            fp_b  = np.array(fp_mm[idx_b], dtype=np.float32)
            emb_b = reducer.transform(fp_b).astype(np.float32)
            # Replace any NaNs produced by transform with zeros
            nan_rows = ~np.isfinite(emb_b).all(axis=1)
            if nan_rows.any():
                emb_b[nan_rows] = 0.0
            emb_mm[idx_b] = emb_b
            del fp_b, emb_b; gc.collect()

    emb_mm.flush()
    np.savez(meta, done=True)
    print(f"  Saved → {emb_path}  ({emb_path.stat().st_size/1e6:.0f} MB)")
    return emb_mm, emb_path


# ══════════════════════════════════════════════════════════════════════════════
# 4.  HDBSCAN  (CPU, multi-core, disk-cached)
# ══════════════════════════════════════════════════════════════════════════════
def run_hdbscan(emb_mm, min_cluster_size, min_samples, work_dir: Path):
    labels_path  = work_dir / "hdbscan_labels.npy"
    probs_path   = work_dir / "hdbscan_probs.npy"
    persist_path = work_dir / "hdbscan_persistence.npy"

    if labels_path.exists() and probs_path.exists():
        labels      = np.load(labels_path)
        probs       = np.load(probs_path)
        persistence = np.load(persist_path) if persist_path.exists() else None
        N           = len(labels)
        n_cl        = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise     = (labels == -1).sum()
        print(f"  [SKIP] HDBSCAN labels loaded  "
              f"clusters={n_cl:,}  noise={n_noise:,} ({n_noise/N*100:.1f}%)")
        return labels, probs, persistence
    exit()
    try:
        import hdbscan as hdbscan_lib
    except ImportError:
        raise ImportError(
            "\n\n  hdbscan not installed!\n"
            "  Fix:  pip install hdbscan\n"
            "  Then re-run — UMAP step will be skipped automatically.\n"
        )

    emb = np.array(emb_mm, dtype=np.float32)
    N   = emb.shape[0]

    # ── Sanitise NaNs/Infs from UMAP transform failures ───────────────────────
    bad_mask = ~np.isfinite(emb).all(axis=1)
    n_bad    = bad_mask.sum()
    if n_bad > 0:
        print(f"  [WARNING] {n_bad:,} rows have NaN/Inf in embedding "
              f"({n_bad/N*100:.2f}%) — replacing with column means.")
        col_means = np.nanmean(emb, axis=0)
        col_means = np.where(np.isfinite(col_means), col_means, 0.0)
        emb[bad_mask] = col_means
    # ─────────────────────────────────────────────────────────────────────────

    print(f"  HDBSCAN {N:,} pts  "
          f"min_cluster={min_cluster_size}  min_samples={min_samples} ...")
    t0 = time.time()

    cl = hdbscan_lib.HDBSCAN(
        min_cluster_size = min_cluster_size,
        min_samples      = min_samples,
        core_dist_n_jobs = -1,
        prediction_data  = True,
        memory           = str(work_dir / "hdbscan_cache"),
    )
    cl.fit(emb)
    labels      = cl.labels_.astype(np.int32)
    probs       = cl.probabilities_.astype(np.float32)
    persistence = cl.cluster_persistence_
    del emb; gc.collect()

    np.save(labels_path,  labels)
    np.save(probs_path,   probs)
    if persistence is not None:
        np.save(persist_path, persistence)

    n_cl    = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"  Done in {time.time()-t0:.0f}s  "
          f"clusters={n_cl:,}  noise={n_noise:,} ({n_noise/N*100:.1f}%)")
    if n_cl > 0:
        sizes = np.array([np.sum(labels == c)
                          for c in range(n_cl)], dtype=np.int64)
        print(f"  Cluster sizes: min={sizes.min()}  "
              f"median={np.median(sizes):.0f}  max={sizes.max():,}")
    return labels, probs, persistence


# ══════════════════════════════════════════════════════════════════════════════
# 5.  MAXMIN
# ══════════════════════════════════════════════════════════════════════════════
def maxmin_pick(fp_mm, candidate_indices: list, n_pick: int,
                device, tile_size: int, seed: int = 42) -> list:
    candidates = np.array(candidate_indices, dtype=np.int32)
    M = len(candidates)
    if n_pick >= M:
        return candidates.tolist()
    rng      = np.random.default_rng(seed)
    sel      = [int(rng.integers(M))]
    min_dist = np.ones(M, dtype=np.float32)
    for _ in range(1, n_pick):
        g = int(candidates[sel[-1]])
        last = torch.from_numpy(
            np.array(fp_mm[g:g+1], dtype=np.uint8)).to(device)
        for cs in range(0, M, tile_size):
            ce  = min(cs + tile_size, M)
            B   = torch.from_numpy(
                np.array(fp_mm[candidates[cs:ce]], dtype=np.uint8)).to(device)
            with torch.no_grad():
                sim = tanimoto_block(last, B).squeeze(0)
            np.minimum(min_dist[cs:ce], 1.0 - sim.cpu().numpy(),
                       out=min_dist[cs:ce])
            del B, sim
        for s in sel: min_dist[s] = -1.0
        nxt = int(np.argmax(min_dist))
        for s in sel: min_dist[s] = 0.0
        sel.append(nxt)
        del last
    return candidates[sel].tolist()


# ══════════════════════════════════════════════════════════════════════════════
# 6.  DENSITY-AWARE SAMPLING
# ══════════════════════════════════════════════════════════════════════════════
def density_aware_sample(labels, probs, fp_mm, n_target,
                          device, tile_size, work_dir: Path, seed=42):
    sel_path = work_dir / "selected_indices.npy"
    if sel_path.exists():
        selected = np.load(sel_path).tolist()
        print(f"  [SKIP] selected_indices.npy exists  ({len(selected):,} mols)")
        return selected

    rng      = np.random.default_rng(seed)
    N        = len(labels)
    selected = []

    # Noise — rarest scaffolds, highest priority
    noise_idx    = np.where(labels == -1)[0]
    noise_budget = min(len(noise_idx), max(1, n_target // 5))
    print(f"  Noise points : {len(noise_idx):,}  budget={noise_budget:,}")

    MAXMIN_NOISE_LIMIT = 10_000   # MaxMin only if manageable; else stratified random
    if len(noise_idx) <= noise_budget:
        selected.extend(noise_idx.tolist())
        print(f"  → All noise included")
    elif noise_budget <= MAXMIN_NOISE_LIMIT:
        print(f"  → MaxMin picking {noise_budget:,} from noise ...")
        selected.extend(
            maxmin_pick(fp_mm, noise_idx.tolist(), noise_budget,
                        device, tile_size, seed))
    else:
        # Too many noise points for MaxMin — use stratified random sampling.
        # Stratify by Tanimoto bit-count (molecular size proxy) to maintain
        # diversity without the O(N²) cost.
        print(f"  → Noise too large for MaxMin ({len(noise_idx):,}) — "
              f"stratified random sampling {noise_budget:,} ...")
        bit_counts  = fp_mm[noise_idx].sum(axis=1).astype(np.int32)
        # 20 strata by bit-count (molecular complexity)
        n_strata    = 20
        strata_edges = np.percentile(bit_counts,
                                     np.linspace(0, 100, n_strata + 1))
        strata_edges[-1] += 1  # include max
        noise_picked = []
        for i in range(n_strata):
            lo, hi   = strata_edges[i], strata_edges[i + 1]
            in_strat = noise_idx[(bit_counts >= lo) & (bit_counts < hi)]
            if len(in_strat) == 0:
                continue
            q = max(1, int(round(len(in_strat) / len(noise_idx) * noise_budget)))
            q = min(q, len(in_strat))
            chosen = rng.choice(in_strat, size=q, replace=False)
            noise_picked.extend(chosen.tolist())
        # Trim to exact budget
        if len(noise_picked) > noise_budget:
            noise_picked = noise_picked[:noise_budget]
        selected.extend(noise_picked)
        print(f"  → Stratified noise done: {len(noise_picked):,} selected")

    print(f"  After noise  : {len(selected):,} selected")

    n_remaining = n_target - len(selected)
    cluster_ids = sorted(set(labels[labels >= 0]))
    n_clusters  = len(cluster_ids)
    if n_clusters == 0 or n_remaining <= 0:
        return selected

    # log-weighted quotas
    sizes  = np.array([np.sum(labels == c) for c in cluster_ids], dtype=np.float64)
    log_w  = np.log1p(sizes)
    quotas = np.round(log_w / log_w.sum() * n_remaining).astype(int)
    diff   = n_remaining - int(quotas.sum())
    quotas[np.argmax(log_w)] += diff

    print(f"  Clusters : {n_clusters:,}  "
          f"size {sizes.min():.0f}–{sizes.max():.0f}  "
          f"quota {quotas.min()}–{quotas.max()}")

    for cid, quota in tqdm(list(zip(cluster_ids, quotas)),
                            desc="  Density-sample", unit="cluster",
                            ncols=80, miniters=100):
        members = np.where(labels == cid)[0]
        q       = min(int(quota), len(members))
        if q <= 0: continue
        if q >= len(members):
            selected.extend(members.tolist()); continue
        top = members[np.argsort(-probs[members])]
        if q <= 3:
            selected.extend(top[:q].tolist())
        else:
            n_cands = min(len(members), q * 3)
            selected.extend(
                maxmin_pick(fp_mm, top[:n_cands].tolist(),
                            q, device, tile_size, seed + cid))

    # Dedup + exact trim/pad
    seen = set()
    selected = [x for x in selected if not (x in seen or seen.add(x))]
    if len(selected) > n_target:
        selected = selected[:n_target]
    elif len(selected) < n_target:
        pad = list(set(range(N)) - set(selected))
        rng.shuffle(pad)
        selected.extend(pad[:n_target - len(selected)])

    np.save(sel_path, np.array(selected, dtype=np.int32))
    print(f"  Saved selected_indices.npy  ({len(selected):,})")
    return selected


# ══════════════════════════════════════════════════════════════════════════════
# 7.  REPORT
# ══════════════════════════════════════════════════════════════════════════════
def save_report(labels, probs, persistence, selected, out_dir):
    sel_set = set(selected)
    rows = []
    for cid in tqdm(sorted(set(labels)), desc="  Report", ncols=80):
        members   = np.where(labels == cid)[0]
        n_sampled = sum(1 for m in members if m in sel_set)
        rows.append(dict(
            cluster_id           = int(cid),
            cluster_type         = "noise" if cid == -1 else "cluster",
            size                 = len(members),
            n_sampled            = n_sampled,
            sample_rate          = round(n_sampled / len(members), 4),
            mean_membership_prob = round(float(probs[members].mean()), 4),
            persistence = (float(persistence[cid])
                          if persistence is not None and cid >= 0
                          and cid < len(persistence) else float("nan")),
        ))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "cluster_report_hdbscan.csv", index=False)
    cl = df[df.cluster_type == "cluster"]
    print(f"\n  Clusters     : {len(cl):,}")
    print(f"  Noise points : {(labels==-1).sum():,}")
    if len(cl):
        print(f"  Size range   : {cl['size'].min()} – {cl['size'].max():,}")
        print(f"  Sample rates : {cl.sample_rate.min():.3f} – "
              f"{cl.sample_rate.max():.3f}")
    print(f"  Report → cluster_report_hdbscan.csv")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",            default=INPUT_FILE)
    parser.add_argument("--target",           type=float, default=TARGET)
    parser.add_argument("--tile_size",        type=int,   default=TILE_SIZE)
    parser.add_argument("--umap_dims",        type=int,   default=UMAP_DIMS)
    parser.add_argument("--umap_neighbors",   type=int,   default=UMAP_NEIGHBORS)
    parser.add_argument("--umap_subsample",   type=int,   default=UMAP_FIT_SUBSAMPLE)
    parser.add_argument("--min_cluster_size", type=int,   default=HDBSCAN_MIN_CLUSTER)
    parser.add_argument("--min_samples",      type=int,   default=HDBSCAN_MIN_SAMPLES)
    parser.add_argument("--work_dir",         default=str(WORK_DIR))
    parser.add_argument("--out_dir",          default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir  = Path(args.out_dir);  out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(args.work_dir); work_dir.mkdir(parents=True, exist_ok=True)
    # global WORK_DIR; WORK_DIR = work_dir

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for MaxMin Tanimoto tiles.")
    device   = torch.device("cuda")
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"\n  GPU     : {torch.cuda.get_device_name(0)}  ({vram_gb:.1f} GB)")
    print(f"  Scratch : {work_dir}  (checkpoints saved here — safe to re-run)")

    # 1. Load
    print(f"\n[1/5] Fingerprints ...")
    df           = pd.read_csv(args.input)
    fp_mm, vm, _ = build_fp_memmap(df[ECFP_COL], work_dir)
    df           = df[vm].reset_index(drop=True)
    N            = fp_mm.shape[0]
    n_target     = max(1, int(round(N * args.target)))
    print(f"  N={N:,}  target={n_target:,} ({args.target*100:.1f}%)")

    # 2. UMAP
    print(f"\n[2/5] UMAP ...")
    emb_mm, _ = run_umap(fp_mm, args.umap_dims, args.umap_neighbors,
                          args.umap_subsample, work_dir, SEED)

    # 3. HDBSCAN
    print(f"\n[3/5] HDBSCAN ...")
    labels, probs, persistence = run_hdbscan(
        emb_mm, args.min_cluster_size, args.min_samples, work_dir)
    del emb_mm; gc.collect()

    # Save labels CSV to out_dir if not already there
    lcsv = out_dir / "hdbscan_labels.csv"
    if not lcsv.exists():
        id_col = df["id"].values if "id" in df.columns else np.arange(N)
        pd.DataFrame({"id" if "id" in df.columns else "mol_idx": id_col,
                      "hdbscan_label": labels,
                      "membership_prob": probs,
                      }).to_csv(lcsv, index=False)
        print(f"  Labels CSV → hdbscan_labels.csv")

    # 4. Sampling
    print(f"\n[4/5] Density-aware sampling ...")
    selected = density_aware_sample(
        labels, probs, fp_mm, n_target,
        device, args.tile_size, work_dir, SEED)

    # 5. Save
    print(f"\n[5/5] Saving representative set ...")
    repr_df = df.iloc[selected].copy()
    out_p   = out_dir / f"representative_{int(args.target*100)}pct_hdbscan.csv"
    repr_df.to_csv(out_p, index=False)
    save_report(labels, probs, persistence, selected, out_dir)

    print(f"\n{'═'*60}")
    print(f"  Input              : {N:,}")
    print(f"  Representative set : {len(repr_df):,}  ({len(repr_df)/N*100:.2f}%)")
    print(f"  Target             : {args.target*100:.1f}%")
    print(f"  Saved → {out_p.name}")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
