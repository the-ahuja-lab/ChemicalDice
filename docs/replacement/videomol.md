# How to Run

1. To compute embeddings for a set of molecules, run:
   bash run_videomol_pipeline.sh
   (This script sets up the environment and runs videomol_end2end.py on your input CSV and frame data. Output is logged in logs_new/)

2. After embeddings are generated in chunks, use merge_videomol_chunks.py to combine them into a single file.

3. If you want to check for missing IDs or filter the final embeddings, use final.py.

## script/ directory (my workflow)
- videomol_end2end.py: Main script to compute VideoMol embeddings for a list of molecules with precomputed video frames. Handles chunking, logging, and saving results in embeddings_new/.
- merge_videomol_chunks.py: Merges all chunked .npz embedding files into a single final .npz file.
- final.py: Checks for missing molecule IDs in the embeddings, writes missing IDs to a log, and saves a filtered final file if needed.
- run_videomol_pipeline.sh: Bash script to activate the environment and launch videomol_end2end.py in the background, logging output to logs_new/.

## Model Used

The script uses a ViT-based FramePredictor model with weights loaded from 'ckpts/VideoMol_vit_small_patch16_224.pth'. Make sure this checkpoint file is present. The model is defined in model/base/predictor.py.

## Requirements
- Python 3.x
- All dependencies should be installed in the videomol conda environment (see my own notes).

## Notes
- This README is for my own reference. Not intended for external use or support.
