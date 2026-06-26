"""
extract_embeddings.py
=====================
Extracts molecular embeddings from five models (Uni-Mol, MolT5, Atomas, MolCA, CDI)
for a single input CSV file and saves one Parquet file per model.

Usage
-----
    python -m ChemicalDice.sota_pipeline.extract_embeddings \
        --input_csv   ./datasets/mydata.csv \
        --output_dir  ./parquets \
        --atomas_ckpt checkpoints/atomas_pretrained.ckpt \
        --molca_ckpt  checkpoints/stage1.ckpt \
        --batch_size  64 \
        --gpu_fraction 0.6
"""

import sys
import os
import argparse
import gc
import logging
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup — adjust if your repo clones live elsewhere
# ---------------------------------------------------------------------------
sys.path.append(os.path.abspath("./Atomas"))
sys.path.append(os.path.abspath("./MolCA"))

from transformers import T5Tokenizer, T5EncoderModel
from unimol_tools import UniMolRepr
from models.atomas import Atomas
from model.blip2_stage1 import Blip2Stage1
from model.blip2_opt import smiles2data
from torch_geometric.data import Batch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def setup_device(gpu_fraction: float = 0.6) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_per_process_memory_fraction(
            gpu_fraction, device=0
        )
        props = torch.cuda.get_device_properties(0)
        reserved_gb = props.total_memory * gpu_fraction / 1e9
        log.info(
            f"GPU: {props.name}  |  "
            f"Total VRAM: {props.total_memory/1e9:.1f} GB  |  "
            f"Reserved for this process: {reserved_gb:.1f} GB "
            f"({gpu_fraction*100:.0f}%)"
        )
    else:
        device = torch.device("cpu")
        log.warning("CUDA not available — running on CPU (will be slow).")
    return device


def free_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def load_molt5(device: torch.device):
    log.info("Loading MolT5 …")
    tokenizer = T5Tokenizer.from_pretrained(
        "laituan245/molt5-base", model_max_length=512
    )
    model = T5EncoderModel.from_pretrained("laituan245/molt5-base")
    model.eval().to(device)
    return tokenizer, model


def load_atomas(ckpt_path: str, device: torch.device):
    log.info("Loading Atomas …")
    atomas_args = argparse.Namespace(
        max_lenth=512,
        queue_size=13200,
        momentum=0.995,
        data_dir="./data",
        model_size="large",
        task="genmol",
        temp_dir="./temp",
        version="v1",
        start_valid_epoch=0,
    )
    model = Atomas.load_from_checkpoint(
        ckpt_path, args=atomas_args, map_location="cpu"
    )
    model.eval().to(device)
    return model


def load_molca(ckpt_path: str, device: torch.device):
    log.info("Loading MolCA …")
    molca_args = argparse.Namespace(
        num_query_token=8,
        cross_attention_freq=2,
        drop_ratio=0,
        tune_gnn=False,
        devices="cpu",
        init_checkpoint=ckpt_path,
        rerank_cand_num=128,
        gtm=True,
        lm=True,
        bert_name="scibert",
        temperature=0.07,
        gin_num_layers=5,
        gin_hidden_dim=300,
        projection_dim=256,
    )
    model = Blip2Stage1.load_from_checkpoint(
        ckpt_path, device=device, args=molca_args, map_location="cpu"
    )
    model.eval().to(device)
    return model


def extract_unimol(
    smiles_list: List[str], ids: List[str], batch_size: int = 128
) -> tuple:
    log.info("Extracting Uni-Mol embeddings …")
    extractor = UniMolRepr(data_type="molecule", remove_hs=False, model_name="unimolv1")
    all_reprs, good_ids = [], []

    for i in range(0, len(smiles_list), batch_size):
        chunk_smi = smiles_list[i : i + batch_size]
        chunk_ids = ids[i : i + batch_size]
        try:
            repr_dict = extractor.get_repr(chunk_smi, return_atomic_reprs=False)
            all_reprs.append(np.array(repr_dict["cls_repr"]))
            good_ids.extend(chunk_ids)
        except Exception as batch_err:
            log.warning(f"  Uni-Mol batch {i}–{i+len(chunk_smi)} failed ({batch_err}); retrying per-molecule …")
            for smi, mol_id in zip(chunk_smi, chunk_ids):
                try:
                    r = extractor.get_repr([smi], return_atomic_reprs=False)
                    all_reprs.append(np.array(r["cls_repr"]))
                    good_ids.append(mol_id)
                except Exception as e:
                    log.warning(f"    Skipped {mol_id}: {e}")
        log.info(f"  Uni-Mol: {min(i+batch_size, len(smiles_list))}/{len(smiles_list)}  (kept {len(good_ids)})")

    del extractor
    free_gpu_memory()
    return np.vstack(all_reprs), good_ids


def extract_molt5(
    smiles_list: List[str],
    ids: List[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int = 64,
) -> tuple:
    log.info("Extracting MolT5 embeddings …")
    all_embs, good_ids = [], []

    def _run_batch(smi_batch):
        enc = tokenizer(smi_batch, padding=True, truncation=True,
                        max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask)
        tok_emb = out.last_hidden_state.float()
        mask    = enc.attention_mask.unsqueeze(-1).float()
        pooled  = (tok_emb * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        result  = pooled.cpu().numpy()
        if np.isnan(result).any():
            raise ValueError("NaN detected in MolT5 output")
        return result

    for i in range(0, len(smiles_list), batch_size):
        chunk_smi = smiles_list[i : i + batch_size]
        chunk_ids = ids[i : i + batch_size]
        try:
            pooled = _run_batch(chunk_smi)
            all_embs.append(pooled)
            good_ids.extend(chunk_ids)
        except Exception as batch_err:
            log.warning(f"  MolT5 batch {i}–{i+len(chunk_smi)} failed ({batch_err}); retrying per-molecule …")
            for smi, mol_id in zip(chunk_smi, chunk_ids):
                try:
                    result = _run_batch([smi])
                    all_embs.append(result)
                    good_ids.append(mol_id)
                except Exception as e:
                    log.warning(f"    Skipped {mol_id}: {e}")
        log.info(f"  MolT5: {min(i+batch_size, len(smiles_list))}/{len(smiles_list)}  (kept {len(good_ids)})")

    return np.vstack(all_embs), good_ids


def extract_atomas(
    smiles_list: List[str],
    ids: List[str],
    model,
    device: torch.device,
    batch_size: int = 64,
) -> tuple:
    log.info("Extracting Atomas embeddings …")
    all_embs, good_ids = [], []

    for i, (smiles, mol_id) in enumerate(zip(smiles_list, ids)):
        try:
            with torch.no_grad():
                enc = model.tokenizer(
                    smiles, padding="max_length", max_length=512,
                    truncation=True, return_tensors="pt"
                ).to(model.molt5.device)
                mask = enc.attention_mask.unsqueeze(-1).float()
                feat = model.get_smiles_feat(smiles)

                pooled = (feat * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                pooled = pooled.squeeze(0).cpu().float().numpy()

            if np.isnan(pooled).any():
                raise ValueError("NaN in Atomas output after masked pooling")
            all_embs.append(pooled)
            good_ids.append(mol_id)
        except Exception as e:
            log.warning(f"    Atomas skipped {mol_id}: {e}")
        if (i + 1) % batch_size == 0 or (i + 1) == len(smiles_list):
            log.info(f"  Atomas: {i+1}/{len(smiles_list)}  (kept {len(good_ids)})")

    return np.vstack(all_embs), good_ids


def extract_molca(
    smiles_list: List[str],
    ids: List[str],
    model,
    device: torch.device,
    batch_size: int = 32,
) -> tuple:
    log.info("Extracting MolCA embeddings …")
    all_embs, good_ids = [], []
    qformer = model.blip2qformer

    def _run_graphs(graph_batch):
        with torch.no_grad():
            ge, gm = qformer.graph_encoder(graph_batch)
            ge = qformer.ln_graph(ge)
            qt = qformer.query_tokens.expand(ge.shape[0], -1, -1)
            qo = qformer.Qformer.bert(
                query_embeds=qt, encoder_hidden_states=ge,
                encoder_attention_mask=gm, return_dict=True,
            )
        result = qo.last_hidden_state.mean(dim=1).cpu().float().numpy()
        if np.isnan(result).any():
            raise ValueError("NaN detected in MolCA output")
        return result

    for i in range(0, len(smiles_list), batch_size):
        chunk_smi = smiles_list[i : i + batch_size]
        chunk_ids = ids[i : i + batch_size]
        try:
            batch = Batch.from_data_list([smiles2data(s) for s in chunk_smi]).to(device)
            pooled = _run_graphs(batch)
            del batch
            all_embs.append(pooled)
            good_ids.extend(chunk_ids)
        except Exception as batch_err:
            log.warning(f"  MolCA batch {i}–{i+len(chunk_smi)} failed ({batch_err}); retrying per-molecule …")
            for smi, mol_id in zip(chunk_smi, chunk_ids):
                try:
                    g = Batch.from_data_list([smiles2data(smi)]).to(device)
                    result = _run_graphs(g)
                    all_embs.append(result)
                    good_ids.append(mol_id)
                except Exception as e:
                    log.warning(f"    Skipped {mol_id}: {e}")
        log.info(f"  MolCA: {min(i+batch_size, len(smiles_list))}/{len(smiles_list)}  (kept {len(good_ids)})")

    return np.vstack(all_embs), good_ids


def extract_cdi(
    smiles_list: List[str],
    ids: List[str],
) -> tuple:
    import tempfile
    from ChemicalDice import smiles_to_embeddings as cdi_api

    CDI_BATCH_LIMIT = 20000

    log.info("Extracting CDI (ChemicalDice) embeddings …")
    log.info(f"  Processing in batches of {CDI_BATCH_LIMIT} (API hard limit).")

    smiles_to_id = dict(zip(smiles_list, ids))
    all_result_dfs = []
    feat_cols = None

    for batch_start in range(0, len(smiles_list), CDI_BATCH_LIMIT):
        batch_smi = smiles_list[batch_start : batch_start + CDI_BATCH_LIMIT]
        batch_ids = ids[batch_start : batch_start + CDI_BATCH_LIMIT]
        batch_num = batch_start // CDI_BATCH_LIMIT + 1
        n_batches = (len(smiles_list) + CDI_BATCH_LIMIT - 1) // CDI_BATCH_LIMIT

        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, prefix="cdi_input_"
        )
        tmp_path = tmp.name
        tmp.close()

        batch_df = pd.DataFrame({"SMILES": batch_smi})
        batch_df.to_csv(tmp_path, index=False)

        try:
            result_df = cdi_api.collect_features_from_csv(
                filepath=tmp_path,
                convert_to_canonical=False,
            )
        except Exception as e:
            log.warning(f"  CDI batch {batch_num}/{n_batches} failed ({e}) — retrying per-molecule …")
            result_df = None
            per_mol_dfs = []
            for smi in batch_smi:
                t2 = tempfile.NamedTemporaryFile(
                    mode="w", suffix=".csv", delete=False, prefix="cdi_mol_"
                )
                t2_path = t2.name
                t2.close()
                pd.DataFrame({"SMILES": [smi]}).to_csv(t2_path, index=False)
                try:
                    r = cdi_api.collect_features_from_csv(
                        filepath=t2_path, convert_to_canonical=False
                    )
                    if r is not None and len(r) > 0:
                        per_mol_dfs.append(r)
                except Exception as mol_e:
                    log.warning(f"    CDI skipped {smi[:40]}...: {mol_e}")
                finally:
                    try:
                        os.remove(t2_path)
                    except OSError:
                        pass
            if per_mol_dfs:
                result_df = pd.concat(per_mol_dfs, ignore_index=True)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

        if result_df is None or len(result_df) == 0:
            log.warning(f"  CDI batch {batch_num}/{n_batches}: no results returned.")
            continue

        smiles_col_out = result_df.columns[0]
        if feat_cols is None:
            feat_cols = result_df.columns[1:].tolist()

        result_df = result_df.copy()
        result_df["_cdi_id"] = result_df[smiles_col_out].map(smiles_to_id)
        result_df = result_df.dropna(subset=["_cdi_id"])
        all_result_dfs.append(result_df)

        log.info(
            f"  CDI batch {batch_num}/{n_batches}: "
            f"{len(result_df)}/{len(batch_smi)} molecules OK  "
            f"(total so far: {sum(len(d) for d in all_result_dfs)})"
        )

    if not all_result_dfs:
        raise ValueError("CDI API returned no results for any batch.")

    combined = pd.concat(all_result_dfs, ignore_index=True)
    good_ids = combined["_cdi_id"].astype(str).tolist()
    embs = combined[feat_cols].values.astype(np.float32)

    nan_count = int(np.isnan(embs).sum())
    if nan_count > 0:
        log.warning(f"  CDI: {nan_count} NaN values in output — affected rows will be dropped.")
        valid_mask = ~np.isnan(embs).any(axis=1)
        embs = embs[valid_mask]
        good_ids = [gid for gid, ok in zip(good_ids, valid_mask) if ok]

    log.info(f"  CDI: kept {len(good_ids)}/{len(smiles_list)} molecules  "
             f"(dim={embs.shape[1]})")
    return embs, good_ids


def load_csv(csv_path: Path, smiles_col: str = "SMILES", id_col: str = "CID"):
    df = pd.read_csv(csv_path)

    for col, flag in [(smiles_col, "--smiles_col"), (id_col, "--id_col")]:
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found in {csv_path.name}. "
                f"Available: {list(df.columns)}. Use {flag} to specify the correct one."
            )

    n_before = len(df)
    df = df.dropna(subset=[smiles_col, id_col])

    df = df.drop_duplicates(subset=[id_col, smiles_col])

    smiles = df[smiles_col].astype(str).tolist()
    ids    = df[id_col].astype(str).tolist()

    return smiles, ids


def output_path_for(stem: str, output_dir: Path, model_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{stem}_{model_name}.parquet"


def save_parquet(embs: np.ndarray, ids: List[str], out_path: Path):
    if len(ids) == 0 or embs.shape[0] == 0:
        raise ValueError("Nothing to save — embeddings array is empty.")
    if embs.shape[0] != len(ids):
        raise ValueError(f"Shape mismatch: {embs.shape[0]} embeddings but {len(ids)} ids.")

    n_dims = embs.shape[1]
    df = pd.DataFrame(
        embs.astype(np.float32),
        columns=list(range(n_dims)),
    )
    df.insert(0, "id", ids)
    df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
    log.info(f"    → Saved {out_path}  ({embs.shape[0]} rows × {n_dims} dims)")


def run_pipeline(args: argparse.Namespace):
    t0_total = time.time()

    device = setup_device(args.gpu_fraction)

    csv_path   = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    stem       = csv_path.stem

    log.info("=" * 60)
    log.info(f"Input : {csv_path}")
    log.info(f"Output: {output_dir}/  (stem: {stem})")
    log.info("=" * 60)

    try:
        smiles_list, ids = load_csv(csv_path, args.smiles_col, args.id_col)
    except Exception as e:
        log.error(f"Failed to load input CSV: {e}")
        return

    if not smiles_list:
        log.error("No valid molecules found in input CSV. Exiting.")
        return

    log.info(f"Loaded {len(smiles_list)} unique molecules.")

    def needs_run(model_name: str) -> bool:
        out = output_path_for(stem, output_dir, model_name)
        if out.exists() and not args.overwrite:
            log.info(f"  [{model_name}] {out.name} already exists — skipping.")
            return False
        return True

    if getattr(args, "skip_cdi", False):
        log.info("  [cdi] Skipped (--skip_cdi passed).")
    elif needs_run("cdi"):
        try:
            embs, out_ids = extract_cdi(smiles_list, ids)
            save_parquet(embs, out_ids, output_path_for(stem, output_dir, "cdi"))
            del embs
        except Exception as e:
            log.error(f"  CDI failed: {e}")

    log.info("=" * 60)
    log.info(f"All done in {(time.time()-t0_total)/60:.1f} min.")
    log.info(f"Parquet files saved under: {output_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract molecular embeddings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--input_csv", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./parquets")
    p.add_argument("--atomas_ckpt", type=str, default="checkpoints/atomas_pretrained.ckpt")
    p.add_argument("--molca_ckpt", type=str, default="checkpoints/stage1.ckpt")
    p.add_argument("--smiles_col", type=str, default="SMILES")
    p.add_argument("--id_col", type=str, default="CID")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gpu_fraction", type=float, default=0.6)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip_cdi", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
