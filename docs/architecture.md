# Architecture of ChemicalDice Integrator (CDI)

The Chemical Dice Integrator (CDI) is a hierarchical, multimodal deep learning framework designed to unify diverse molecular representations into a single, high-information latent space. The architecture is split into two primary operational modes: **CDI-Basic** for unsupervised integration and **CDI-Generalised** for scalable, sequence-based inference.

---

## 1. CDI-Basic: Multimodal Fusion Engine

CDI-Basic is the foundational engine that performs the unsupervised integration of six orthogonal molecular modalities. It utilizes a **two-tiered hierarchical autoencoder** architecture to learn deep semantic relationships between different views of a molecule.

### Tier 1: Semantic Commonality Autoencoders (SCA)
In the first tier, CDI employs six dedicated autoencoders. Each SCA is responsible for learning inter-modality dependencies through a **Leave-One-Out (LOO)** reconstruction objective:

*   **Input**: Concatenated latent information from five modalities.
*   **Target**: Reconstruction of the sixth (omitted) modality.
*   **Outcome**: This process enforces the discovery of shared inter-modality information, yielding six latent subspaces that capture the "semantic overlap" between feature domains.

### Tier 2: Super-Embedding Autoencoder (SEA)
The outputs from the six SCAs are concatenated and fed into the Super-Embedding Autoencoder.

*   **Function**: Compress the consolidated latent vectors into a single, 8192-dimensional **Super Embedding**.
*   **Objective**: Minimize global reconstruction error and ensure a cohesive, unified representation that preserves the unique characteristics of each source modality while capturing their synergy.

---

## 2. CDI-Generalised: Sequence-to-Embedding Mapping

While CDI-Basic provides the "gold standard" integrated embedding, it requires all six source features to be computed beforehand—a significant computational bottleneck. **CDI-Generalised** solves this by providing a direct pathway from chemical structure to the integrated space.

### Mamba State-Space Model (SSM) Core
CDI-Generalised leverages a modern **Mamba SSM** architecture (based on the SMI-SSED framework) to map raw SMILES strings directly into the 8192-D CDI space.

*   **Efficiency**: Bypasses the need to generate descriptor-heavy inputs (like MOPAC or Mordred) during inference.
*   **Training**: The model is trained using a supervised regression objective to ensure both geometric and angular alignment with the pre-trained CDI-Basic targets.
*   **Scalability**: Enables high-throughput molecular screening with the latency of a single model while maintaining the breadth of the original multimodal ensemble.

---

## 3. The Six Orthogonal Modalities

CDI integrates information from across the chemical intelligence spectrum:

1.  **Quantum-Mechanical (MOPAC)**: Captures electronic properties and orbital landscapes.
2.  **Topological/Graph-Based (GROVER)**: Maps atomic connections and structural motifs.
3.  **Linguistic/Language-Based (ChemBERTa)**: Extracts transformer-based semantics from SMILES syntax.
4.  **Biological/Bioactivity (Signaturizer)**: Represents pre-trained bioactivity signatures and profiles.
5.  **Visual/Image-Based (ImageMol)**: Learns 2D topological encodings via visual ResNet features.
6.  **Physicochemical (Mordred)**: Provides deterministic mathematical descriptors defining geometry and chemistry.

---

## 4. Mathematical Formulation & Loss Functions

The hierarchical training protocol of ChemicalDice is governed by specific objective functions that ensure high-fidelity representation learning.

### CDI-Basic Optimization
The total loss for the fusion engine combines the reconstruction and alignment objectives of both tiers:

$$
\mathcal{L}_{total} = \frac{1}{6} \sum_{j=1}^{6} \left( \mathcal{L}_{RE}(\mathbf{a}_j^i, \tilde{\mathbf{a}}_j^i) + \mathcal{L}_{MSE}(\mathbf{p}_j^i, \mathbf{d}_j^i) \right) + \mathcal{L}_{SEA}
$$

*   $\mathcal{L}_{RE}$: Tier 1 reconstruction error for the leave-one-out modality.
*   $\mathcal{L}_{MSE}$: Mean Squared Error for semantic alignment between latent spaces.
*   $\mathcal{L}_{SEA}$: Global reconstruction loss for the Tier 2 super-embedding.

### CDI-Generalised Optimization
The Mamba-based mapping is optimized to align the sequence-derived embeddings with the established CDI manifold:

$$
\mathcal{L}_{SSM} = \frac{1}{N \times D} \sum_{i=1}^{N} \sum_{j=1}^{D} (E_{target, ij} - E_{pred, ij})^2
$$

*   $N$: Batch size.
*   $D$: Embedding dimension (fixed at 8192).
*   $E_{target}$: Gold-standard embeddings from CDI-Basic.
*   $E_{pred}$: Predicted embeddings from the Mamba encoder.

---

## Technical Specifications Summary

| Component | Architecture | Dimensionality | Primary Objective |
| :--- | :--- | :--- | :--- |
| **CDI-Basic** | Two-Tiered Autoencoder | 8192-D | Multimodal Representation Fusion |
| **CDI-Generalised** | Mamba State-Space Model | 8192-D | Sequence-to-Latent Mapping |
| **Training Scale** | ~1.09 Billion Parameters | - | Global Semantic Integration |
| **Inference Mode** | Sequence-based (SMILES) | - | High-throughput Scalability |
