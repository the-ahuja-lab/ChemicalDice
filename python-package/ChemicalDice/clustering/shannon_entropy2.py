from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import entropy
import collections
import pandas as pd


df = pd.read_csv("representative_10pct_hdbscan.csv")



smiles_list = df['Canonical_SMILES']
# Assuming 'smiles_list' is your list of SMILES strings
mols = [Chem.MolFromSmiles(s) for s in smiles_list if Chem.MolFromSmiles(s) is not None]
scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]

# Count the frequency of each distinct scaffold
counts = collections.Counter(scaffolds)

# Calculate probabilities
total_molecules = len(scaffolds)
probabilities = [count / total_molecules for count in counts.values()]

# Calculate Shannon Entropy (base 2)
shannon_entropy = entropy(probabilities, base=2)
print(f"Scaffold Shannon Entropy: {shannon_entropy:.4f}")
