# Descriptors

ChemicalDice integrates a diverse range of molecular representations—from graph-level neural embeddings to quantum mechanical descriptors—to build a comprehensive latent space. To ensure high-performance I/O and support for massive chemical libraries, the framework utilizes optimized, chunked HDF5 structures for all descriptor storage and retrieval.

## Source Descriptors
The six source descriptor representations built natively into the training execution system are:

1. **Grover**: Dense graph neural network parameters mapping atomic connections.
2. **Chemberta**: Deep language transformer contexts extrapolated from base SMILES.
3. **Signaturizer**: Pre-trained mappings representing explicit bioactivity profiles.
4. **ImageMol**: 2D topological mapping visual encodings trained via ResNets.
5. **Mordred**: Pure deterministic mathematical rules defining physicochemical geometry.
6. **MOPAC**: Quantum descriptors detailing molecular properties.

## HDF5 Data Standards

Prior to executing integrators, datasets must be structurally fused:

1. **Global Intersection**: A central script `get_common_ids()` iterates every CSV in $O(n)$ chunked time filtering down to a pure mathematical intersection map where every single structural ID strictly exists cleanly across all $N$ datasets.
2. **Matrix Packing**: Every CSV is translated continuously into `.h5` datasets with the dataset hook defined as `/data`.
3. **Continuous Imputation**: The `mordred` pipeline applies real-time row-based KNN imputation `k=5` dynamically against NaN properties avoiding structural data leakages natively.

---

# Installation

To set up **ChemicalDice** descriptor generation, you can natively install the package with the optional `[descriptors]` dependency group and let the automated setup tool handle external binaries.

```bash
# 1. Install ChemicalDice with descriptor dependencies (mordred, transformers, signaturizer, etc.)
pip install .[descriptors]

# 2. Run the automated setup tool to configure external binaries
cdi setup
```

The `cdi setup` command intelligently probes your environment. Because you installed the `[descriptors]` group, it will automatically download, compile, and configure **MOPAC** and **3DMORSE** into your `~/.chemicaldice` directory natively based on your OS.

---

# Calculating Descriptors

Generate foundational descriptor representations directly from SMILES using the natively integrated package module.

### CLI Execution
Run the calculation engine using the `cdi` utility. You can specify a single descriptor or a list of them.

```bash
# Calculate specific descriptors
cdi compute --input chembl_35.csv --descriptors mordred mopac
```

### Python API
```python
from ChemicalDice.descriptors.calculate import calculate_descriptors

calculate_descriptors(
    input_file="chembl_35.csv",
    output_dir="Chemicaldice_data",
    descriptors=["mordred", "mopac"]
)
```

### Supported Descriptors
* `mopac` (Quantum)
* `grover` (Graph)
* `imagemol` (Image)
* `chemberta` (LLM)
* `signaturizer` (Bioactivity)
* `mordred` (Physico-chemical)
* `all` (Calculates all of the above)

---

## Matrix Conversion (HDF5)

To prepare for high-performance I/O integration, transform the raw CSVs into optimized `.h5` files.

### CLI Execution
```bash
# Convert CSVs to optimized HDF5 structures
cdi convert --input_dir Chemicaldice_data --output_dir Chemicaldice_data
```

### Python API
```python
from ChemicalDice.descriptors.convert import convert_to_hdf5

convert_to_hdf5(
    input_dir="Chemicaldice_data",
    output_dir="Chemicaldice_data",
    chunk_size=10000,
    knn_neighbors=5
)
```

This applies chunked processing to handle massive datasets, performs KNN-based imputation for missing data (e.g., in `mordred.csv`), and enforces structural schema consistency across all representations.

**Output:** Descriptor HDF5 Files optimized for fast I/O ingestion by the integrators.


---

## Technical Descriptor Details

### 1. Grover (Graph-level Representation)
**Grover** uses a massive message-passing neural network architecture (MPNN) to extract deep topological features. It captures intricate graph structures by learning atomic environments and bond relationships across millions of molecules, providing a robust geometric foundation for downstream tasks.

### 2. ChemBERTa (Chemical Language Model)
**ChemBERTa** leverages transformer-based architectures similar to BERT but specifically trained on over 77 million SMILES strings. By learning the "grammar" of chemical structures, it provides contextualized embeddings that represent both local functional groups and global molecular properties.

### 3. Signaturizer (Bioactivity Signatures)
**Signaturizer** maps molecules into a biological activity space. It uses pre-trained models to predict the "signature" of a molecule across multiple bioactivity categories (e.g., target binding, metabolic pathways, and cellular responses), bridging the gap between chemical structure and biological effect.

### 4. ImageMol (Visual Molecular Encodings)
**ImageMol** treats molecular structures as 2D images. Using deep convolutional neural networks (CNNs), it learns to extract visual features from molecular diagrams. This approach captures structural motifs and symmetry in a way that is complementary to traditional graph or string-based methods.

### 5. Mordred (Physicochemical Rules)
**Mordred** is a comprehensive descriptor calculator that applies over 1,600 deterministic mathematical rules to a molecular structure. It covers a vast range of features, including molecular weight, logP, polar surface area, and complex topological indices (like the Wiener index), providing a detailed physicochemical profile.

### 6. MOPAC (Quantum Mechanical Descriptors)
**MOPAC** uses semi-empirical quantum mechanical methods to calculate electronic and energetic properties. It provides critical data such as the energy of the Highest Occupied Molecular Orbital (HOMO), Lowest Unoccupied Molecular Orbital (LUMO), dipole moments, and partial atomic charges, which are essential for understanding reactivity and binding affinity.
