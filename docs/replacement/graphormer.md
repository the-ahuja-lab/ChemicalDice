# How to Run

1. To compute embeddings for a set of molecules, run:
	bash run_graphormer_embeddings.sh
	(This script sets up the environment and runs graphormer_code.py on your input CSV. Output is logged in logs_new1/)

2. After embeddings are generated in chunks, use merge_graphormer_chunks.py to combine them into a single file.

3. If you want to check for missing IDs or filter the final embeddings, use final.py.

## Model Used

The script uses the pretrained model 'clefourrier/graphormer-base-pcqm4mv1' from HuggingFace. The model will be downloaded automatically if not present locally.


# Graphormer Embedding Pipeline (Internal)

This README is for my own internal use, in case I want to run the Graphormer embedding pipeline in the future.

## scripts/ directory (my workflow)
- `graphormer_code.py`: Main script to compute Graphormer embeddings for a list of molecules from a CSV file. Handles chunking, logging, and saving results in `embeddings_new1/`.
- `merge_graphormer_chunks.py`: Merges all chunked `.npz` embedding files into a single final `.npz` file.
- `final.py`: Checks for missing molecule IDs in the embeddings, writes missing IDs to a log, and saves a filtered final file if needed.
- `run_graphormer_embeddings.sh`: Bash script to activate the environment and launch `graphormer_code.py` in the background, logging output to `logs_new1/`.

## How I Run Things

- I install all requirements using the conda environment specified for Graphormer (see the main repo or my own environment).
- To compute embeddings for a set of molecules, I use the `run_graphormer_embeddings.sh` script, which sets up the environment and runs `graphormer_code.py` on my input CSV.
- After embeddings are generated in chunks, I use `merge_graphormer_chunks.py` to combine them into a single file.
- If I want to check for missing IDs or filter the final embeddings, I use `final.py`.

## Requirements
- Python 3.x
- All dependencies should be installed in the `graphormer_hf` conda environment (see main repo or my own notes).

## Notes
- This README is for my own reference. Not intended for external use or support.