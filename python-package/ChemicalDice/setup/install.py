import subprocess
import sys
import platform
import logging
import os
import site
from pathlib import Path

# ---------------- Logging ----------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("ChemicalDice-Installer")

BASE_DIR = Path.home() / ".chemicaldice"
MAMBA_DIR = BASE_DIR / "mamba"
SMI_DIR = BASE_DIR / "materials.smi_ssed"

# ---------------- Utils ----------------
def run(cmd, cwd=None):
    try:
        log.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        log.error(f"Command failed: {e}")
        raise


# ---------------- VERSION UTILS ----------------
def get_closest_version(target_str, available_strs):
    """
    Finds the exact match or the closest previous version.
    Parses strings like '1.13' or '118' into tuples/ints for accurate math comparison.
    """
    def parse_ver(v):
        return tuple(map(int, v.split('.'))) if '.' in v else int(v)

    avail_parsed = [parse_ver(v) for v in available_strs]
    target_parsed = parse_ver(target_str)

    # 1. Exact match
    if target_parsed in avail_parsed:
        return available_strs[avail_parsed.index(target_parsed)]

    # 2. Find previous version (largest version <= target)
    smaller_or_equal = [v for v in avail_parsed if v <= target_parsed]
    if smaller_or_equal:
        best_match = max(smaller_or_equal)
        return available_strs[avail_parsed.index(best_match)]

    # 3. If all available are newer than target, fall back to the absolute minimum available
    return available_strs[avail_parsed.index(min(avail_parsed))]


# ---------------- GPU CHECK ----------------
def get_cuda_info():
    try:
        import torch
        if not torch.cuda.is_available():
            return None, None, False
        return torch.version.cuda, torch.__version__, torch._C._GLIBCXX_USE_CXX11_ABI
    except Exception:
        return None, None, False


# ---------------- MAMBA ----------------
def install_mamba():
    log.info("🔧 Installing Mamba (Part of Generalized CDI)...")

    if platform.system() != "Linux":
        log.warning("Mamba supported only on Linux → skipping")
        return

    cuda, torch_v, abi = get_cuda_info()

    if cuda is None:
        log.warning("CUDA not available → skipping Mamba")
        return

    # Extract versions for matching
    cuda_v = cuda.replace(".", "")
    torch_v_short = ".".join(torch_v.split(".")[:2])
    abi_tag_str = "TRUE" if abi else "FALSE"

    # Supported versions from the Mamba v1.2.0 GitHub releases
    supported_cuda = ["118", "122"]
    supported_torch = ["1.12", "1.13", "2.0", "2.1", "2.2", "2.3"]

    # Dynamic fallback matching
    cuda_tag_val = get_closest_version(cuda_v, supported_cuda)
    torch_tag_val = get_closest_version(torch_v_short, supported_torch)

    cuda_tag = "cu" + cuda_tag_val
    torch_tag = "torch" + torch_tag_val

    py_major, py_minor = sys.version_info.major, sys.version_info.minor
    py_v = f"cp{py_major}{py_minor}"
    # Python 3.7 uses the 'm' suffix in its ABI tag based on the wheel list
    py_suffix = "cp37m" if py_v == "cp37" else py_v

    wheel_url = (
        f"https://github.com/state-spaces/mamba/releases/download/v1.2.0/"
        f"mamba_ssm-1.2.0+{cuda_tag}{torch_tag}cxx11abi{abi_tag_str}-{py_v}-{py_suffix}-linux_x86_64.whl"
    )

    try:
        log.info(f"🚀 Detected System: PyTorch {torch_v_short}, CUDA {cuda}, Python {py_major}.{py_minor}")
        log.info(f"🚀 Selected Wheel Configuration: Torch {torch_tag_val}, CUDA {cuda_tag_val}")
        log.info(f"🚀 Attempting to install wheel: {wheel_url}")
        run([sys.executable, "-m", "pip", "install", wheel_url])
        log.info("✅ Mamba installed via wheel")
    except Exception as e:
        log.warning(f"⚠️ Wheel installation failed: {e}")
        log.warning("⚠️ Falling back to standard pip install (Source/PyPI build)...")
        try:
            run([sys.executable, "-m", "pip", "install", "causal-conv1d", "mamba-ssm"])
            log.info("✅ Mamba installed via PyPI/Source")
        except Exception as e2:
            log.error(f"❌ Mamba installation failed completely: {e2}")
            log.error("💡 Hint: Ensure your environment has CUDA Toolkit and a C++ compiler installed.")


# ---------------- SMI-SSED ----------------
def install_smi_ssed():
    log.info("🔧 Setting up SMI-SSED (Part of Generalized CDI)...")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        log.error("huggingface_hub missing. Install via pip install huggingface-hub")
        return

    if not SMI_DIR.exists():
        run(["git", "clone", "https://huggingface.co/ibm-research/materials.smi_ssed", str(SMI_DIR)])

    model_path = SMI_DIR / "smi_ssed/inference/smi_ssed/smi_ssed_130.pt"

    if not model_path.exists():
        log.info("Downloading SMI-SSED model weights...")
        hf_hub_download(
            repo_id="ibm-research/materials.smi_ssed",
            filename="smi_ssed_130.pt",
            local_dir=str(model_path.parent),
            local_dir_use_symlinks=False
        )

    patch_file = SMI_DIR / "smi_ssed/inference/smi_ssed/load.py"
    if patch_file.exists():
        text = patch_file.read_text()
        if "weights_only=False" not in text:
            text = text.replace("torch.load(model_path)", "torch.load(model_path, weights_only=False)")
            patch_file.write_text(text)

    try:
        site_packages = site.getsitepackages()[0]
        pth = Path(site_packages) / "smi_ssed.pth"
        pth.write_text(str(SMI_DIR / "smi_ssed/inference"))
        log.info(f"✅ SMI-SSED path linked in {pth}")
    except Exception:
        log.warning("⚠️ Could not create .pth file for SMI-SSED.")

    log.info("✅ SMI-SSED ready")


# ---------------- MOPAC ----------------
def install_mopac():
    log.info("🔧 Setting up MOPAC binaries...")

    if os.name == "nt":
        log.warning("Windows: manual MOPAC install required.")
        return

    mopac_dir = BASE_DIR / "mopac"
    if mopac_dir.exists():
        log.info("MOPAC already present")
        return

    try:
        log.info("Downloading MOPAC 22.1.1...")
        tar_path = BASE_DIR / "mopac.tar.gz"
        run(["wget", "https://github.com/openmopac/mopac/releases/download/v22.1.1/mopac-22.1.1-linux.tar.gz", "-O", str(tar_path)])
        run(["tar", "-xzf", str(tar_path)], cwd=BASE_DIR)
        
        extracted = BASE_DIR / "mopac-22.1.1-linux"
        if extracted.exists():
            extracted.rename(mopac_dir)
        log.info("✅ MOPAC installed successfully")
    except Exception as e:
        log.error(f"❌ MOPAC installation failed: {e}")


# ---------------- MAIN ----------------
def run_setup(part="all"):
    """
    Modular CDI setup.
    :param part: 'gen', 'mopac', or 'all'
    """
    BASE_DIR.mkdir(exist_ok=True)
    log.info(f"🚀 Initializing CDI Setup Part: {part}")

    if part in ["gen", "all"]:
        install_mamba()
        install_smi_ssed()

    if part in ["mopac", "all"]:
        install_mopac()

    log.info(f"🎉 Setup for '{part}' completed successfully")