# How to Run

1. To compute AQME quantum descriptors for a set of molecules, run:
   bash run_aqme_pipeline.sh
   (This script sets up the environment and runs aqme_embeddings.py on your input CSV. Output is logged in logs_new/)

2. After embeddings are generated in chunks, use merge_aqme_chunks.py to combine them into a single file.

3. If you want to check for missing IDs or filter the final embeddings, use your own script or check logs.

## scripts/ directory (my workflow)
- aqme_embeddings.py: Main script to compute AQME quantum descriptors for a list of molecules. Handles chunking, logging, and saving results in embeddings_new/.
- merge_aqme_chunks.py: (Empty or placeholder) Intended to merge all chunked .npz embedding files into a single final .npz file.
- run_aqme_pipeline.sh: Bash script to activate the environment and launch aqme_embeddings.py in the background, logging output to logs_new/.

## Model/Software Used

The script uses the AQME package (https://github.com/ivanslapnicar/aqme) for conformer search and quantum descriptor calculation. Make sure AQME and all dependencies are installed in the aqme conda environment.

## Requirements
- Python 3.x
- All dependencies should be installed in the aqme conda environment (see my own notes).

## Notes
- This README is for my own reference. Not intended for external use or support.
