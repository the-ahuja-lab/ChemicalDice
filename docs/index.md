# Chemical Dice Integrator

**ChemicalDice Integrator (CDI)** is a high-performance deep learning framework designed to unify heterogeneous chemical representations into a single, high information rich latent space. By fusing six complementary molecular embeddings, CDI produces a consolidated vector optimized for large-scale cheminformatics, bioinformatics, and AI-driven molecular discovery tasks.

---

## Modular Architecture
We understand the computational burden inherent to scientific computing. CDI is strictly engineered into three installable modules utilizing powerful lazy-loading techniques:

### 1. Minimal 🟢
Run real-time API client hooks effortlessly without PyTorch or massive computational graphs. Designed purely for extracting feature-sets off our cloud clusters.

### 2. **[Training](training.md)** 🟠
Abstracts deep HDF5 preprocessing and multi-modal Autoencoder/Mamba network alignments into object-oriented classes. Complete dataset optimization out-of-core.

### 3. Deployment 🔴
Harnesses highly scalable ASGI (FastAPI) servers to host structural generalizations asynchronously, matching the execution performance of enterprise deployments.

---

## Next Steps
- Head to the **[Installation Matrix](installation.md)** to correctly configure your setup.
- Jump straight into the **[Quickstart](quickstart.md)** payload.
- Learn about the **[Training Specifics](training.md)** for Basic and Generalised models.
- Review our multi-agent **[Architecture](architecture.md)** diagrams mapping Autoencoders to State-Spaces natively.
