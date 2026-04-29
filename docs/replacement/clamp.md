# How to Run

1. To compute embeddings for a set of molecules, run:
	bash run_clamp.sh
	(This script sets up the environment and runs clamp_embeddings.py on your input CSV. Output is logged in logs_new/)

2. After embeddings are generated in chunks, use merge_clamp_chunks.py to combine them into a single file.

3. If you want to check for missing IDs or filter the final embeddings, use final.py.


### scripts/ directory (my own workflow)
- `clamp_embeddings.py`: Main script to compute CLAMP embeddings for a list of molecules from a CSV file. Handles chunking, logging, and saving results in `embeddings_new/`.
- `merge_clamp_chunks.py`: Merges all chunked `.npz` embedding files into a single final `.npz` file.
- `final.py`: Checks for missing molecule IDs in the embeddings, writes missing IDs to a log, and saves a filtered final file if needed.
- `run_clamp.sh`: Bash script to activate the environment and launch `clamp_embeddings.py` in the background, logging output to `logs_new/`.

## How I Run Things

- I install clamp using pip (from PyPI or GitHub).
- To compute embeddings for a set of molecules, I use the `run_clamp.sh` script, which sets up the environment and runs `clamp_embeddings.py` on my input CSV.
- After embeddings are generated in chunks, I use `merge_clamp_chunks.py` to combine them into a single file.
- If I want to check for missing IDs or filter the final embeddings, I use `final.py`.


## Requirements
- Python 3.x
- All dependencies are listed in `env.yml` (includes PyTorch, RDKit, etc.)
