# Combined Embedding Pipelines Guide

This repository contains multiple pipelines to compute embeddings and descriptors for molecules. Below is the unified guidance on how to run them, including the required instructions and example code snippets where applicable.

## General Workflow
For each pipeline, the workflow generally consists of:
1. Activating the relevant environment.
2. Running a bash script to launch the embedding/descriptor generation script (`run_*.sh`). Output is typically logged in a `logs_new/` directory.
3. Merging the output chunks into a single `.npz` file (`merge_*_chunks.py`).
4. Running a final script (`final.py`) to process outputs, check for missing molecule IDs, and filter the final embeddings.

---

## 1. AlvaDesc

### Requirements
- Python 3.x
- `alvadescpy` package (`pip install alvadescpy`)
- Licensed alvaDesc installation (e.g., set path/license via python script or automatically)
- `alvadesc` conda environment

### Instructions
```bash
# 1. Activate the conda environment:
conda activate alvadesc

# 2. Run the pipeline (logs directed to logs_new_with_nan/):
bash run_alvadesc_new.sh

# 3. Merge output chunks:
python merge_alvadesc_chunks.py

# 4. Post-processing / analysis:
python final.py
```

### Example Usage (Python Code)
```python
from alvadescpy import CONFIG

# If needed to set paths explicitly
CONFIG['alvadesc_path'] = '/home/suvenduk/Yasser/alvadesc/alvadesc_extracted/usr/bin/alvaDescCLI'
CONFIG['license_path'] = '/path/to/alvaDesc.lic'

from alvadescpy import alvadesc
# Calculate all descriptors for a SMILES string
descriptors = alvadesc(ismiles='CCC', descriptors='ALL')

# Get a dictionary with labels
descriptors = alvadesc(ismiles='CCC', descriptors='ALL', labels=True)
```

---

## 2. CLAMP

### Requirements
- Python 3.x
- CLAMP package installed via pip
- Dependencies listed in `env.yml` (PyTorch, RDKit, etc.)

### Instructions
```bash
# 1. Set up and compute embeddings:
bash run_clamp.sh

# 2. Merge chunked embeddings:
python merge_clamp_chunks.py

# 3. Check for missing IDs and filter final embeddings:
python final.py
```

---

## 3. Graphormer

### Model Details
Uses the pretrained model `clefourrier/graphormer-base-pcqm4mv1` from Hugging Face. Downloaded automatically if not present locally.

### Requirements
- Python 3.x
- Dependencies installed in the `graphormer_hf` conda environment.

### Instructions
```bash
# 1. Setup and generate embeddings (logs directed to logs_new1/):
bash run_graphormer_embeddings.sh

# 2. Merge chunked embeddings:
python merge_graphormer_chunks.py

# 3. Check for missing IDs and filter final embeddings:
python final.py
```

---

## 4. MolFormer

### Model Details
Uses the pretrained model `ibm/MoLFormer-XL-both-10pct` from Hugging Face by default. Downloaded automatically. Model can be changed via the `--model` argument to the python script.

### Requirements
- Python 3.x
- Dependencies installed in the `MolTran_CUDA11` conda environment.

### Instructions
```bash
# 1. Setup and generate embeddings:
bash run_molformer_embeddings.sh

# 2. Merge chunked embeddings:
python merge_molformer_chunks.py

# 3. Check for missing IDs and filter final embeddings:
python final.py
```

---

## 5. AQME (Quantum Descriptors)

### Model Details
Uses the AQME package for conformer search and quantum descriptor calculations.

### Requirements
- Python 3.x
- Dependencies and AQME package installed in the `aqme` conda environment.

### Instructions
```bash
# 1. Compute AQME quantum descriptors:
bash run_aqme_pipeline.sh

# 2. Merge chunked descriptors:
python merge_aqme_chunks.py

# 3. Filter outputs or check logs (use your own script or logs here).
```

---

## 6. VideoMol

### Model Details
Uses a ViT-based FramePredictor model with weights loaded from `ckpts/VideoMol_vit_small_patch16_224.pth`. Ensure the checkpoint file exists.

### Requirements
- Python 3.x
- Dependencies installed in the `videomol` conda environment.

### Instructions
```bash
# 1. Compute embeddings with precomputed video frames:
bash run_videomol_pipeline.sh

# 2. Merge chunked embeddings:
python merge_videomol_chunks.py

# 3. Check for missing IDs and filter final embeddings:
python final.py
```
