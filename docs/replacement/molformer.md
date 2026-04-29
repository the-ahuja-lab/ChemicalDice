# How to Run

1. To compute embeddings for a set of molecules, run:
   bash run_molformer_embeddings.sh
   (This script sets up the environment and runs molformer_embeddings.py on your input CSV. Output is logged in logs_new/)

2. After embeddings are generated in chunks, use merge_molformer_chunks.py to combine them into a single file.

3. If you want to check for missing IDs or filter the final embeddings, use final.py.

## scripts/ directory (my workflow)
- `molformer_embeddings.py`: Main script to compute MolFormer embeddings for a list of molecules from a CSV file. Handles chunking, logging, and saving results in `embeddings_new/`.
- `merge_molformer_chunks.py`: Merges all chunked `.npz` embedding files into a single final `.npz` file.
- `final.py`: Checks for missing molecule IDs in the embeddings, writes missing IDs to a log, and saves a filtered final file if needed.
- `run_molformer_embeddings.sh`: Bash script to activate the environment and launch `molformer_embeddings.py` in the background, logging output to `logs_new/`.

## Model Used

The script uses the pretrained model 'ibm/MoLFormer-XL-both-10pct' from HuggingFace by default. The model will be downloaded automatically if not present locally. You can change the model by passing the --model argument to molformer_embeddings.py.

## Requirements
- Python 3.x
- All dependencies should be installed in the MolTran_CUDA11 conda environment (see my own notes).

## Notes
- This README is for my own reference. Not intended for external use or support.