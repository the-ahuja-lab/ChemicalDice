"""
shannon_entropy.py
───────────────────
Compares Shannon Entropy of the ORIGINAL ChEMBL dataset vs the
REPRESENTATIVE 10% subset across multiple chemical descriptors.

Shannon Entropy measures information diversity:
  H = -Σ p(x) * log2(p(x))
  High H → diverse, spread-out distribution
  Low H  → redundant, concentrated distribution

A good representative subset should have H_repr ≈ H_original
(same diversity in fewer molecules).

Entropy computed over:
  1. Bit-count buckets  (molecular complexity)
  2. Per-bit entropy    (each ECFP6 bit independently)
     → mean, std, min, max across 2048 bits
  3. Scaffold diversity (Bemis-Murcko scaffolds via RDKit, if available)
  4. Tanimoto pairwise distance distribution (random sample)

Outputs:
  shannon_entropy_report.csv    — per-descriptor entropy comparison
  shannon_entropy_plots.png     — 4-panel visual comparison

Usage:
    python shannon_entropy.py \
        --original  chembl_35_ecfp6.csv \
        --repr      results/representative_10pct_hdbscan.csv \
        --fp_col    ECFP6 \
        --out_dir   results/
"""

import argparse, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")

# ── Style ─────────────────────────────────────────────────────────────────────
BG       = "#0a0e17"
PANEL_BG = "#111827"
GRID_COL = "#1f2937"
TEXT     = "#f0f4f8"
COL_ORIG = "#58a6ff"   # blue  — original
COL_REPR = "#f78166"   # coral — representative
MUTED    = "#6b7280"


def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=TEXT, labelsize=8.5)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_COL)
    ax.grid(True, color=GRID_COL, linewidth=0.35, alpha=0.5)
    if title:  ax.set_title(title, fontsize=11, fontweight="bold",
                             color=TEXT, pad=9)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9)


# ══════════════════════════════════════════════════════════════════════════════
# FINGERPRINT PARSING
# ══════════════════════════════════════════════════════════════════════════════
def parse_fp(series: pd.Series, label: str) -> np.ndarray:
    valid = series.dropna()
    n_bad = len(series) - len(valid)
    if n_bad:
        print(f"  [{label}] Dropping {n_bad:,} null rows")
    print(f"  [{label}] Parsing {len(valid):,} fingerprints ...")
    CHUNK = 50_000
    rows  = []
    for s in range(0, len(valid), CHUNK):
        e = min(s + CHUNK, len(valid))
        rows.append(np.vstack(
            valid.iloc[s:e].apply(
                lambda x: np.frombuffer(str(x).encode(),
                                        dtype=np.uint8) - ord("0")
            ).values
        ).astype(np.uint8))
    return np.vstack(rows)


# ══════════════════════════════════════════════════════════════════════════════
# ENTROPY CALCULATIONS
# ══════════════════════════════════════════════════════════════════════════════
def shannon_entropy_1d(counts: np.ndarray) -> float:
    """Shannon entropy in bits from a count array."""
    counts = counts[counts > 0].astype(np.float64)
    p      = counts / counts.sum()
    return float(-np.sum(p * np.log2(p)))


def max_entropy_1d(n_bins: int) -> float:
    """Maximum possible entropy for n_bins (uniform distribution)."""
    return np.log2(n_bins) if n_bins > 1 else 0.0


def per_bit_entropy(fp: np.ndarray) -> np.ndarray:
    """
    Shannon entropy of each bit position independently.
    H_bit = -p1*log2(p1) - p0*log2(p0)
    Returns array of shape (fp_len,) with entropy in [0, 1] bits.
    """
    N    = fp.shape[0]
    p1   = fp.mean(axis=0).astype(np.float64)          # fraction of 1s per bit
    p0   = 1.0 - p1
    # Avoid log(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        h  = np.where(p1 > 0, -p1 * np.log2(p1), 0.0)
        h += np.where(p0 > 0, -p0 * np.log2(p0), 0.0)
    return h.astype(np.float32)


def bit_count_entropy(fp: np.ndarray, n_bins: int = 20) -> tuple:
    """
    Entropy of the bit-count distribution (molecular complexity proxy).
    Returns (entropy, bin_edges, counts).
    """
    bit_counts = fp.sum(axis=1).astype(np.int32)
    counts, edges = np.histogram(bit_counts, bins=n_bins)
    return shannon_entropy_1d(counts), edges, counts, bit_counts


def tanimoto_distance_entropy(fp: np.ndarray,
                               n_sample: int = 5000,
                               n_bins: int = 50,
                               seed: int = 42) -> tuple:
    """
    Entropy of the pairwise Tanimoto DISTANCE distribution.
    Samples n_sample molecules, computes all-pairs distances.
    Higher entropy → distances more uniformly spread → more diverse.
    Returns (entropy, bin_edges, counts).
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(fp), min(n_sample, len(fp)), replace=False)
    sub = fp[idx].astype(np.float32)

    inter = sub @ sub.T
    bits  = sub.sum(axis=1, keepdims=True)
    union = np.clip(bits + bits.T - inter, 1e-8, None)
    sim   = inter / union
    # Upper triangle distances only (exclude self)
    dist  = 1.0 - sim[np.triu_indices(len(sub), k=1)]

    counts, edges = np.histogram(dist, bins=n_bins, range=(0.0, 1.0))
    return shannon_entropy_1d(counts), edges, counts, dist


def bit_frequency_entropy(fp: np.ndarray, n_bins: int = 50) -> tuple:
    """
    Entropy of the per-bit frequency distribution.
    p_bit = fraction of molecules that have bit i set.
    Histogram of these 2048 frequencies → entropy.
    High H → bits are spread across many different frequencies (diverse).
    Low H  → many bits have same frequency (uniform/redundant).
    """
    freqs         = fp.mean(axis=0)
    counts, edges = np.histogram(freqs, bins=n_bins, range=(0.0, 1.0))
    return shannon_entropy_1d(counts), edges, counts, freqs


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════
def plot_comparison(orig_data: dict, repr_data: dict,
                    entropy_report: pd.DataFrame, out_path: Path, dpi: int):
    fig = plt.figure(figsize=(20, 15), facecolor=BG)
    fig.suptitle(
        "Shannon Entropy: Original ChEMBL vs Representative 10% Subset",
        fontsize=14, fontweight="bold", color=TEXT, y=0.985
    )
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           hspace=0.42, wspace=0.32,
                           left=0.06, right=0.97,
                           top=0.94, bottom=0.07)

    ax_bar  = fig.add_subplot(gs[0, :])   # full top row: summary bar chart
    ax_bc   = fig.add_subplot(gs[1, 0])   # bit-count distribution
    ax_tan  = fig.add_subplot(gs[1, 1])   # tanimoto distances
    ax_freq = fig.add_subplot(gs[1, 2])   # bit-frequency distribution

    # ── Top: entropy summary bar chart ────────────────────────────────────────
    metrics   = entropy_report["metric"].tolist()
    h_orig    = entropy_report["H_original"].tolist()
    h_repr    = entropy_report["H_representative"].tolist()
    retention = entropy_report["retention_pct"].tolist()

    x     = np.arange(len(metrics))
    width = 0.35
    bars1 = ax_bar.bar(x - width/2, h_orig, width,
                       color=COL_ORIG, alpha=0.85,
                       label="Original", edgecolor=BG, linewidth=0.5)
    bars2 = ax_bar.bar(x + width/2, h_repr, width,
                       color=COL_REPR, alpha=0.85,
                       label="Representative 10%",
                       edgecolor=BG, linewidth=0.5)

    # Annotate retention %
    for i, (b1, b2, ret) in enumerate(zip(bars1, bars2, retention)):
        col   = "#3fb950" if ret >= 95 else ("#f59e0b" if ret >= 85 else "#f85149")
        ymax  = max(b1.get_height(), b2.get_height())
        ax_bar.text(i, ymax + 0.05, f"{ret:.1f}%",
                    ha="center", va="bottom", fontsize=8.5,
                    color=col, fontweight="bold")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(metrics, fontsize=9.5, color=TEXT)
    style_ax(ax_bar,
             title="Shannon Entropy Comparison  "
                   "(annotation = entropy retention %)",
             ylabel="Shannon Entropy (bits)")
    ax_bar.legend(fontsize=9, facecolor=PANEL_BG, labelcolor=TEXT,
                  framealpha=0.85, edgecolor=GRID_COL)

    # Horizontal reference line: max entropy per metric
    ax_bar.axhline(0, color=GRID_COL, lw=0.5)

    # ── Bit-count distribution ─────────────────────────────────────────────────
    _, o_edges, o_counts, _ = orig_data["bit_count"]
    _, r_edges, r_counts, _ = repr_data["bit_count"]
    o_norm = o_counts / o_counts.sum()
    r_norm = r_counts / r_counts.sum()
    mid    = (o_edges[:-1] + o_edges[1:]) / 2
    ax_bc.fill_between(mid, 0, o_norm, alpha=0.4, color=COL_ORIG,
                        step="mid", label="Original")
    ax_bc.fill_between(mid, 0, r_norm, alpha=0.6, color=COL_REPR,
                        step="mid", label="Representative")
    ax_bc.step(mid, o_norm, color=COL_ORIG, lw=1.5, where="mid")
    ax_bc.step(mid, r_norm, color=COL_REPR, lw=1.5, where="mid",
               linestyle="--")
    ho = orig_data["bit_count"][0]
    hr = repr_data["bit_count"][0]
    style_ax(ax_bc,
             title=f"Bit-Count Distribution\n"
                   f"H_orig={ho:.3f}  H_repr={hr:.3f}  "
                   f"retention={hr/ho*100:.1f}%",
             xlabel="Number of set bits", ylabel="Probability")
    ax_bc.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT,
                  framealpha=0.85, edgecolor=GRID_COL)

    # ── Tanimoto distance distribution ────────────────────────────────────────
    _, o_edges, o_counts, _ = orig_data["tanimoto"]
    _, r_edges, r_counts, _ = repr_data["tanimoto"]
    o_norm = o_counts / o_counts.sum()
    r_norm = r_counts / r_counts.sum()
    mid    = (o_edges[:-1] + o_edges[1:]) / 2
    ax_tan.fill_between(mid, 0, o_norm, alpha=0.4, color=COL_ORIG, label="Original")
    ax_tan.fill_between(mid, 0, r_norm, alpha=0.6, color=COL_REPR,
                         label="Representative")
    ax_tan.plot(mid, o_norm, color=COL_ORIG, lw=1.5)
    ax_tan.plot(mid, r_norm, color=COL_REPR, lw=1.5, linestyle="--")
    ho = orig_data["tanimoto"][0]
    hr = repr_data["tanimoto"][0]
    style_ax(ax_tan,
             title=f"Pairwise Tanimoto Distance\n"
                   f"H_orig={ho:.3f}  H_repr={hr:.3f}  "
                   f"retention={hr/ho*100:.1f}%",
             xlabel="Tanimoto distance (1 − similarity)",
             ylabel="Probability")
    ax_tan.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT,
                   framealpha=0.85, edgecolor=GRID_COL)

    # ── Bit-frequency distribution ─────────────────────────────────────────────
    _, o_edges, o_counts, o_freqs = orig_data["bit_freq"]
    _, r_edges, r_counts, r_freqs = repr_data["bit_freq"]
    o_norm = o_counts / o_counts.sum()
    r_norm = r_counts / r_counts.sum()
    mid    = (o_edges[:-1] + o_edges[1:]) / 2
    ax_freq.fill_between(mid, 0, o_norm, alpha=0.4, color=COL_ORIG, label="Original")
    ax_freq.fill_between(mid, 0, r_norm, alpha=0.6, color=COL_REPR,
                          label="Representative")
    ax_freq.plot(mid, o_norm, color=COL_ORIG, lw=1.5)
    ax_freq.plot(mid, r_norm, color=COL_REPR, lw=1.5, linestyle="--")
    ho = orig_data["bit_freq"][0]
    hr = repr_data["bit_freq"][0]
    style_ax(ax_freq,
             title=f"Bit-Frequency Distribution\n"
                   f"H_orig={ho:.3f}  H_repr={hr:.3f}  "
                   f"retention={hr/ho*100:.1f}%",
             xlabel="Fraction of molecules with bit set",
             ylabel="Probability (over 2048 bits)")
    ax_freq.legend(fontsize=8, facecolor=PANEL_BG, labelcolor=TEXT,
                    framealpha=0.85, edgecolor=GRID_COL)

    # Footer
    n_orig = len(orig_data["bit_count"][3])
    n_repr = len(repr_data["bit_count"][3])
    fig.text(0.5, 0.005,
             f"Original: {n_orig:,} molecules  |  "
             f"Representative: {n_repr:,} molecules "
             f"({n_repr/n_orig*100:.2f}%)  |  "
             "Retention % = H_repr / H_orig × 100",
             ha="center", color=MUTED, fontsize=8.5)

    print(f"  Saving → {out_path} ...")
    fig.savefig(out_path, dpi=dpi, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  Done  ({out_path.stat().st_size/1e6:.1f} MB)")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True,
                        help="Full dataset CSV  (e.g. chembl_35_ecfp6.csv)")
    parser.add_argument("--repr",     required=True,
                        help="Representative subset CSV")
    parser.add_argument("--fp_col",   default="ECFP6")
    parser.add_argument("--n_bins",   type=int, default=20)
    parser.add_argument("--tan_sample", type=int, default=5000,
                        help="Molecules to use for pairwise Tanimoto (default 5000)")
    parser.add_argument("--out_dir",  default=".")
    parser.add_argument("--dpi",      type=int, default=200)
    args = parser.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n[1/3] Loading datasets ...")
    df_orig = pd.read_csv(args.original)
    df_repr = pd.read_csv(args.repr)
    print(f"  Original     : {len(df_orig):,} molecules")
    print(f"  Representative: {len(df_repr):,} molecules "
          f"({len(df_repr)/len(df_orig)*100:.2f}%)")

    fp_orig = parse_fp(df_orig[args.fp_col], "original")
    fp_repr = parse_fp(df_repr[args.fp_col], "representative")

    # ── Entropy calculations ──────────────────────────────────────────────────
    print("\n[2/3] Computing Shannon entropy metrics ...")

    metrics = {}
    for name, fp in [("original", fp_orig), ("repr", fp_repr)]:
        print(f"\n  [{name}]")
        t0 = time.time()

        # 1. Bit-count distribution entropy
        bc = bit_count_entropy(fp, n_bins=args.n_bins)
        print(f"    Bit-count entropy       : {bc[0]:.4f} bits  "
              f"(max={max_entropy_1d(args.n_bins):.4f})")

        # 2. Per-bit entropy (mean, std, fraction of informative bits)
        pb      = per_bit_entropy(fp)
        pb_mean = pb.mean()
        pb_frac = (pb > 0.05).mean()   # fraction of bits with H > 0.05
        print(f"    Mean per-bit entropy    : {pb_mean:.4f} bits  "
              f"(frac informative={pb_frac*100:.1f}%)")

        # 3. Tanimoto distance entropy
        print(f"    Computing Tanimoto distances (sample={args.tan_sample:,}) ...")
        tan = tanimoto_distance_entropy(fp, n_sample=args.tan_sample)
        print(f"    Tanimoto dist entropy   : {tan[0]:.4f} bits  "
              f"(max={max_entropy_1d(50):.4f})")

        # 4. Bit-frequency distribution entropy
        bf = bit_frequency_entropy(fp, n_bins=50)
        print(f"    Bit-freq entropy        : {bf[0]:.4f} bits")

        metrics[name] = {
            "bit_count" : bc,
            "per_bit"   : (pb_mean, pb, pb_frac),
            "tanimoto"  : tan,
            "bit_freq"  : bf,
        }
        print(f"    Done in {time.time()-t0:.1f}s")

    # ── Build report ──────────────────────────────────────────────────────────
    o, r = metrics["original"], metrics["repr"]
    rows = [
        dict(metric="Bit-count dist.",
             H_original    = round(o["bit_count"][0], 4),
             H_representative = round(r["bit_count"][0], 4),
             H_max         = round(max_entropy_1d(args.n_bins), 4)),
        dict(metric="Mean per-bit",
             H_original    = round(float(o["per_bit"][0]), 4),
             H_representative = round(float(r["per_bit"][0]), 4),
             H_max         = 1.0),
        dict(metric="Tanimoto dist.",
             H_original    = round(o["tanimoto"][0], 4),
             H_representative = round(r["tanimoto"][0], 4),
             H_max         = round(max_entropy_1d(50), 4)),
        dict(metric="Bit-freq dist.",
             H_original    = round(o["bit_freq"][0], 4),
             H_representative = round(r["bit_freq"][0], 4),
             H_max         = round(max_entropy_1d(50), 4)),
        dict(metric="Frac. informative bits",
             H_original    = round(float(o["per_bit"][2]), 4),
             H_representative = round(float(r["per_bit"][2]), 4),
             H_max         = 1.0),
    ]
    report = pd.DataFrame(rows)
    report["retention_pct"] = (
        report["H_representative"] / report["H_original"] * 100
    ).round(2)
    report["H_max_utilisation_orig"] = (
        report["H_original"] / report["H_max"] * 100
    ).round(2)
    report["H_max_utilisation_repr"] = (
        report["H_representative"] / report["H_max"] * 100
    ).round(2)

    report_path = out_dir / "shannon_entropy_report.csv"
    report.to_csv(report_path, index=False)

    print("\n  ── Shannon Entropy Report ───────────────────────────────────")
    print(report.to_string(index=False))
    print(f"\n  Saved → {report_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    print("\n[3/3] Generating plots ...")
    plot_comparison(
        orig_data = {k: o[k] for k in ["bit_count","tanimoto","bit_freq"]},
        repr_data = {k: r[k] for k in ["bit_count","tanimoto","bit_freq"]},
        entropy_report = report[report.metric != "Frac. informative bits"],
        out_path  = out_dir / "shannon_entropy_plots.png",
        dpi       = args.dpi
    )

    # ── Per-bit entropy comparison ────────────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(16, 5), facecolor=BG)
    ax.set_facecolor(PANEL_BG)
    bits = np.arange(fp_orig.shape[1])
    pb_o = o["per_bit"][1]
    pb_r = r["per_bit"][1]
    ax.fill_between(bits, 0, pb_o, alpha=0.4, color=COL_ORIG, label="Original")
    ax.fill_between(bits, 0, pb_r, alpha=0.55, color=COL_REPR,
                     label="Representative")
    ax.plot(bits, pb_o, color=COL_ORIG, lw=0.6, alpha=0.8)
    ax.plot(bits, pb_r, color=COL_REPR, lw=0.6, alpha=0.8, linestyle="--")
    ax.axhline(1.0, color=TEXT, lw=0.8, linestyle=":", alpha=0.4,
               label="Max entropy = 1.0 bit")
    style_ax(ax,
             title="Per-Bit Shannon Entropy  (2048 ECFP6 bits)",
             xlabel="Bit position", ylabel="Shannon entropy (bits)")
    ax.legend(fontsize=9, facecolor=PANEL_BG, labelcolor=TEXT,
              framealpha=0.85, edgecolor=GRID_COL)
    fig2.tight_layout()
    pb_path = out_dir / "per_bit_entropy.png"
    fig2.savefig(pb_path, dpi=args.dpi, facecolor=BG, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Saved → {pb_path}")

    print(f"\n{'═'*60}")
    print(f"  Outputs in : {out_dir.resolve()}")
    print(f"  shannon_entropy_report.csv")
    print(f"  shannon_entropy_plots.png  (4-panel)")
    print(f"  per_bit_entropy.png        (2048 bits)")
    print(f"{'═'*60}")


if __name__ == "__main__":
    main()
