# 🧬 **ChemicalDice Integrator (CDI)**  
**CDI (ChemicalDice Integrator)** is an advanced **deep-learning framework** built to unify diverse chemical representations into a single, information-rich latent space. It integrates six complementary molecular embeddings from **ChemicalDice** into a consolidated vector, optimized for downstream cheminformatics and bioinformatics tasks.

---

<div align="center">
  <img src="Images/CDI.png" alt="ChemicalDice Integrator Overview" width="750">
</div>

<p align="center">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg">
  <img src="https://img.shields.io/badge/docs-passing-green">
  <img src="https://img.shields.io/badge/python-3.9+-blue">
  <a href="https://github.com/the-ahuja-lab/ChemicalDiceIntegrator">
    <img src="https://img.shields.io/badge/Code-Source-black">
  </a>
</p>

---

## ⚗️ **Overview**

CDI extends the **ChemicalDice** featurization ecosystem by performing unsupervised integration of **six distinct molecular embeddings**:

- 🧬 **Quantum Descriptors**  
- ⚗️ **Bioactivity Signatures**  
- 💬 **Language Model Embeddings**  
- 🌐 **Graph-Derived Representations**  
- ⚖️ **Physicochemical Profiles**  
- 🖼️ **2D Molecular Image Features**  

Each compound’s six feature types are combined to create a **single latent embedding** that captures chemical, structural, and biological semantics. These embeddings can be directly used for tasks such as **QSAR modeling**, **virtual screening**, **drug-target interaction prediction**, and **bioactivity clustering**.

---

## ⚙️ **Implementation Details**

| Parameter | Description |
|------------|--------------|
| Framework | PyTorch / TensorFlow |
| Loss Function | Mean Squared Error + Reconstruction Error |
| Activation | ReLU |
| Latent Dimension | 1024 |
| Input Features | 6 precomputed embeddings per molecule |
| Output | Unified latent embedding (`.npy` or `.csv`) |

---

## 📦 **Installation**

```bash
git clone https://github.com/the-ahuja-lab/ChemicalDiceIntegrator.git
cd ChemicalDiceIntegrator
pip install -r requirements.txt
```

---

## 🚀 **Usage Example**

```python
from CDI import ChemicalDiceIntegrator

# Load six input embeddings for each molecule
# Example: quantum, bioactivity, language, graph, physicochemical, and 2D features

integrator = ChemicalDiceIntegrator()
super_embeddings = integrator.fit_transform(six_feature_matrix)
```

**Output:**
```
Molecule_ID | Super_Embedding_Vector (1024 dims)
-----------------------------------------------
MOL_001     | [0.0123, 0.4421, 0.2235, ...]
MOL_002     | [0.1032, 0.5124, 0.1346, ...]
```

---

## 📊 **Applications**

- 🧩 Unified embedding generation for **QSAR / virtual screening**  
- 🧠 Latent-space mapping for **deorphanization** and **bioactivity clustering**  
- ⚗️ Foundation for **Chemical Foundation Models**  
- 🔬 Enables **cross-modal integration** of text, graph, and physicochemical data**  

---

## 🧠 **Citation**

If you use **ChemicalDice Integrator (CDI)** in your research, please cite:

> *ChemicalDice Integrator (CDI): An Evolutionary-Guided Deep Learning Framework for Unified Molecular Embedding Integration*  
> Mudit Gupta, The Ahuja Lab, 2024.  
> [GitHub Repository](https://github.com/the-ahuja-lab/ChemicalDiceIntegrator)
