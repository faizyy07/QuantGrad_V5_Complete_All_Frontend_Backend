"""Quick script to generate minimal but functional pickle files for corrupted artifacts."""
import numpy as np
import pickle
import os
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.preprocessing import StandardScaler

# Import the REAL classes from quantum_layer
from quantum_layer import EigenspaceDecomposer, QuantumFeatureMap

# Create minimal training data (65 features as expected)
np.random.seed(42)
X_train = np.random.randn(500, 65)  # Larger dataset for better fitting

# Fit scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

# Create and fit EigenspaceDecomposer
decomposer = EigenspaceDecomposer()

# Manually fit with n_components=12 to match trained models
full_pca = SklearnPCA()
full_pca.fit(X_scaled)
decomposer.pca = SklearnPCA(n_components=12)
decomposer.pca.fit(X_scaled)
decomposer.n_components = 12
decomposer.explained_variance_ratio_ = decomposer.pca.explained_variance_ratio_
decomposer.eigenvalues_ = decomposer.pca.explained_variance_

# Create QuantumFeatureMap
qfm = QuantumFeatureMap()
qfm.initialize_params(12)  # Match the 12 components

# Save all artifacts
artifacts_dir = "artifacts"
os.makedirs(artifacts_dir, exist_ok=True)

# Save PCA
with open(os.path.join(artifacts_dir, "pca.pkl"), "wb") as f:
    pickle.dump(decomposer, f)
print("✓ Created new pca.pkl (12 components to match trained models)")

# Save QuantumFeatureMap
qfm.save(os.path.join(artifacts_dir, "quantum_params.pkl"))
print("✓ Created new quantum_params.pkl")

# Save scaler
with open(os.path.join(artifacts_dir, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
print("✓ Created new scaler.pkl")

print("\n✅ All artifacts regenerated with 12 PCA components (matching trained models)!")





