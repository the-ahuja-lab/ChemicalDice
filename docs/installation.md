# ⚙️ Installation 

**ChemicalDice Integrator (CDI)** depends on advanced scientific and deep learning libraries such as **RDKit**, **PyTorch**, and **HDF5**.
To avoid dependency conflicts and unnecessary overhead, CDI is designed with a **modular installation approach**.

> [!TIP]
> Install only what you need based on your use case.

---

## 🛠️ Environment Setup

To setup an environment to run **ChemicalDice**, you can install Miniconda using the following commands:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Follow the prompts after the above commands to install Conda. Create a separate environment named `chemicaldice` using the `conda create` command:

```bash
conda create -n chemicaldice python=3.9
conda activate chemicaldice
```

## 📦 Install Packages

To use the ChemicalDice package, you need to install it along with its dependencies. You can install ChemicalDice and its dependencies using the following commands:

---


## 🧩 Minimal Installation

**Purpose:**
Lightweight usage for:

* Remote API inference
* Structural validation
* Schema-level operations

No heavy deep learning frameworks are required.

```bash
pip install ChemicalDice
pip install numpy pandas rdkit tqdm requests 
pip install scikit-learn xgboost lightgbm
```

**Installs:**

* `numpy`
* `pandas`
* `rdkit`
* `tqdm`
* `requests`
* `scikit-learn`
* `xgboost`
* `lightgbm`

---

## 🧬 Descriptor Generation

**Purpose:**
Compute external modal descriptors (e.g., Mordred, Quantum, Grover) directly from SMILES strings using the CDI feature extraction modules.

```bash
pip install ChemicalDice[descriptors]
```

**Installs (in addition to minimal setup):**

* `mordred`
* `transformers`
* `tokenizers`
* `networkx`

> [!NOTE]
> Some descriptors like **MOPAC** or **3D-MORSE** require pre-compiled binaries. You can automatically configure them by running:
> ```bash
> cdi setup mopac
> ```

---

## 🧠 Training Environment

**Purpose:**
Enables deep model optimization through two primary frameworks:

* **CDI-Basic**: Multi-modal representation learning and unsupervised autoencoder dimensionality reduction.
* **CDI-Generalised**: Supervised SMILES modeling utilizing **Mamba (smi_ssed)** structured state-spaces.

```bash
pip install ChemicalDice[training]
```

**Installs (in addition to minimal setup):**

* `torch`
* `h5py`
* `huggingface-hub`
* `smi_ssed`

> [!NOTE]
> Initializing the Mamba kernels and model weights requires a one-time configuration:
> ```bash
> cdi setup gen
> ```

---

## 🚀 Deployment Server

**Purpose:**
Production-ready setup for:

* High-throughput inference
* Asynchronous API serving
* SMILES-to-embedding translation

```bash
pip install ChemicalDice[deployment]
```

**Installs (in addition to minimal setup):**

* `fastapi`
* `uvicorn`
* `pydantic`
* `torch`
* `smi_ssed`

---

## ⚙️ Environment Configuration (`cdi setup`)

After installing the Python package, you must configure the external environment parts to enable 3D descriptors and deep learning architectures.

| Command | Target | Description |
| :--- | :--- | :--- |
| `cdi setup gen` | **Generalized CDI** | Installs Mamba kernels and fetches SMI-SSED model weights from HuggingFace. |
| `cdi setup mopac` | **Quantum Descriptors** | Downloads and configures MOPAC and 3D-MORSE binaries. |
| `cdi setup all` | **Full Environment** | Runs both of the above (default). |

> [!IMPORTANT]
> The `cdi setup gen` command requires a working CUDA installation and `git` to clone the weight repositories.
