"""
Command Line Interface mapping the ChemicalDice tiers directly to terminal utilities.
Leverages pure standard library `argparse` to remain compliant with Tier 1 constraints.
"""

import argparse
import logging
import sys

# Configure root cli logger gracefully
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("cdi-cli")

def main():
    parser = argparse.ArgumentParser(
        description="ChemicalDice Integrator CLI: Unifying Heterogeneous Chemical Representations."
    )
    subparsers = parser.add_subparsers(dest="command", help="Available CDI deployment strategies")

    # --- 1. Data Retrieval (Tier 1) ---
    parser_pull = subparsers.add_parser("fetch", help="Stream CDI features from an input CSV containing SMILES.")
    parser_pull.add_argument("--input", required=True, help="Input CSV file with a 'SMILES' column")
    parser_pull.add_argument("--output", default="cdi_features.csv", help="Output path to save mapped features")
    parser_pull.add_argument("--canonicalize", action="store_true", help="Force RDKit canonicalization locally")

    # --- 2. Local Descriptor Generation ---
    parser_compute = subparsers.add_parser("compute", help="Calculate molecular descriptors from SMILES.")
    parser_compute.add_argument("remaining", nargs=argparse.REMAINDER, help="Arguments passed to the calculation script")

    parser_convert = subparsers.add_parser("convert", help="Convert descriptor CSVs into optimized HDF5 structures.")
    parser_convert.add_argument("remaining", nargs=argparse.REMAINDER, help="Arguments passed to the conversion script")

    # --- 3. Model Training (Tier 2) ---
    parser_train_ae = subparsers.add_parser("train-basic", help="Train the Unsupervised Multi-Modal Autoencoder.")
    parser_train_ae.add_argument("--data-files", nargs='+', required=True, help="List of HDF5 input manifolds")
    parser_train_ae.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser_train_ae.add_argument("--output-model", default="cdi_model.pt", help="Path to save the trained model weights")
    parser_train_ae.add_argument("--output-embeddings", default="cdi_embeddings.h5", help="Path to save the training set embeddings")
    parser_train_ae.add_argument("--bottleneck", type=int, default=8192, help="The latent space dimension (e.g., 8192, 1024, 512)")
    
    parser_train_mb = subparsers.add_parser("train-gen", help="Fine-Tune HuggingFace Mamba Network against CDI loss maps.")
    parser_train_mb.add_argument("--smiles-csv", required=True, help="Raw SMILES token topology")
    parser_train_mb.add_argument("--target-h5", required=True, help="Target Regression array map (8192-D)")
    parser_train_mb.add_argument("--mamba-dir", default=None, help="Local directory hosting smi_ssed (defaults to ~/.chemicaldice/materials.smi_ssed)")
    parser_train_mb.add_argument("--target-dim", type=int, default=8192, help="Target dimension to match the basic model bottleneck")

    # --- 5. Setup & Initialization ---
    parser_setup = subparsers.add_parser("setup", help="Automatically install and configure Mamba, MOPAC, and SMI-SSED dependencies.")
    parser_setup.add_argument("part", choices=["gen", "mopac", "all"], default="all", nargs="?",
                              help="Which part of the setup to run: 'gen' (Generalized CDI), 'mopac' (3D descriptors), or 'all'.")

    # --- 4. Deployment (Tier 3) ---
    parser_serve = subparsers.add_parser("serve", help="Deploy the active Inference ASGI layer on Uvicorn.")
    parser_serve.add_argument("--host", default="0.0.0.0")
    parser_serve.add_argument("--port", type=int, default=8001)

    # --- 5. Analysis & Benchmarking ---
    parser_benchmark = subparsers.add_parser("benchmark", help="Run the Generic SOTA Benchmark Pipeline (inspect, label, classify, plots).")
    parser_benchmark.add_argument("remaining", nargs=argparse.REMAINDER, help="Arguments passed to the SOTA orchestrator")

    parser_cluster = subparsers.add_parser("cluster", help="Run the Density-Aware Sampling module to create a representative subset of ChEMBL.")
    parser_cluster.add_argument("remaining", nargs=argparse.REMAINDER, help="Arguments passed to the clustering script")

    parser_ood = subparsers.add_parser("ood", help="Perform Out-of-Distribution (OOD) analysis using Scaffold vs Random splits.")
    parser_ood.add_argument("remaining", nargs=argparse.REMAINDER, help="Arguments passed to the OOD analysis script")

    parser_ldc = subparsers.add_parser("ldc", help="Perform Low Data Condition (LDC) analysis across multiple training fractions.")
    parser_ldc.add_argument("remaining", nargs=argparse.REMAINDER, help="Arguments passed to the LDC analysis script")

    args = parser.parse_args()

    # Route execution blocks via explicit local scope imports 
    # to fiercely protect the optional dependency isolations!

    if args.command == "fetch":
        from ChemicalDice.core.api_client import collect_features_from_csv
        logger.info(f"Initiating remote stream sequence mapped to: {args.input}")
        df = collect_features_from_csv(args.input, convert_to_canonical=args.canonicalize)
        if df is not None:
            df.to_csv(args.output, index=False)
            logger.info(f"Target matrices compiled optimally at: {args.output}")

    elif args.command == "compute":
        from ChemicalDice.descriptors.calculate import main as compute_main
        sys.argv = [sys.argv[0]] + args.remaining
        compute_main()

    elif args.command == "convert":
        from ChemicalDice.descriptors.convert import main as convert_main
        sys.argv = [sys.argv[0]] + args.remaining
        convert_main()

    elif args.command == "train-basic":
        try:
            from ChemicalDice.training.basic_model import train_basic_cdi
            train_basic_cdi(
                args.data_files, 
                num_epochs=args.epochs, 
                model_path=args.output_model, 
                embedding_path=args.output_embeddings,
                bottleneck=args.bottleneck
            )
        except ImportError:
            logger.error("Tier 2 components missing. Install using: pip install ChemicalDice[training]")
            sys.exit(1)

    elif args.command == "train-gen":
        try:
            from ChemicalDice.training.gen_model import train_generalised_cdi
            train_generalised_cdi(args.smiles_csv, args.target_h5, args.mamba_dir, target_dim=args.target_dim)
        except ImportError:
            logger.error("Tier 2 components missing. Install using: pip install ChemicalDice[training]")
            sys.exit(1)

    elif args.command == "setup":
        from ChemicalDice.setup.install import run_setup
        logger.info(f"Initializing ChemicalDice environment configuration for: {args.part}...")
        run_setup(part=args.part)

    elif args.command == "serve":
        try:
            import uvicorn
        except ImportError:
            logger.error("Tier 3 ASGI host missing. Install using: pip install ChemicalDice[deployment]")
            sys.exit(1)
        
        logger.info("Transferring execution pipeline to Uvicorn Worker Threading...")
        uvicorn.run("ChemicalDice.deployment.api:app", host=args.host, port=args.port, log_level="info")

    elif args.command == "benchmark":
        from ChemicalDice.sota_pipeline.Evaluate import benchmark as sota_benchmark, make_default_config
        cfg = make_default_config()
        # Simple routing for CLI args to config
        if len(args.remaining) >= 1:
            cfg["results_dir"] = args.remaining[0]
        if len(args.remaining) >= 2:
            cfg["datasets"] = [args.remaining[1]]
        if len(args.remaining) >= 3:
            cfg["label_col"] = args.remaining[2]
        sota_benchmark(cfg)

    elif args.command == "cluster":
        from ChemicalDice.clustering.density_aware_sampling import main as cluster_main
        sys.argv = [sys.argv[0]] + args.remaining
        cluster_main()

    elif args.command == "ood":
        from ChemicalDice.experiments.ood_analysis import main as ood_main
        sys.argv = [sys.argv[0]] + args.remaining
        ood_main()

    elif args.command == "ldc":
        from ChemicalDice.experiments.ldc_analysis import main as ldc_main
        sys.argv = [sys.argv[0]] + args.remaining
        ldc_main()

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
