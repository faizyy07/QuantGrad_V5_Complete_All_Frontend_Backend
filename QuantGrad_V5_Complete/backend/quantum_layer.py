"""
quantum_layer.py — clean rewrite with robust Qiskit imports
"""

import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Qiskit imports ────────────────────────────────────────────────────
QISKIT_AVAILABLE = False

try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    QISKIT_AVAILABLE = True
    print("[quantum] Qiskit loaded.")
except Exception as e:
    print(f"[quantum] Qiskit unavailable: {e} — using classical fallback.")

ARTIFACTS_DIR      = os.path.join(os.path.dirname(__file__), "artifacts")
VARIANCE_THRESHOLD = 0.95
# Statevector simulation allocates 2**n complex amplitudes. Keep the default
# local cap low enough for ordinary developer laptops; larger PCA spaces use
# the deterministic classical feature-map fallback instead.
MAX_STATEVECTOR_QUBITS = 12


class EigenspaceDecomposer:
    def __init__(self, variance_threshold=VARIANCE_THRESHOLD):
        self.variance_threshold        = variance_threshold
        self.pca                       = None
        self.n_components              = None
        self.explained_variance_ratio_ = None
        self.eigenvalues_              = None

    def fit(self, X_2d):
        full_pca = PCA()
        full_pca.fit(X_2d)
        cumvar            = np.cumsum(full_pca.explained_variance_ratio_)
        self.n_components = int(np.searchsorted(cumvar, self.variance_threshold) + 1)
        self.n_components = min(self.n_components, X_2d.shape[1])
        print(f"[quantum] PCA: {self.n_components} components → "
              f"{cumvar[self.n_components-1]*100:.1f}% variance")
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X_2d)
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.eigenvalues_              = self.pca.explained_variance_
        return self

    def transform(self, X_2d):
        return self.pca.transform(X_2d)

    def fit_transform(self, X_2d):
        self.fit(X_2d)
        return self.transform(X_2d)

    def save(self, path=None):
        path = path or os.path.join(ARTIFACTS_DIR, "pca.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[quantum] PCA saved → {path}")

    @staticmethod
    def load(path=None):
        path = path or os.path.join(ARTIFACTS_DIR, "pca.pkl")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        print(f"[quantum] PCA loaded ({obj.n_components} components)")
        return obj


class QuantumFeatureMap:
    def __init__(self, n_qubits=None):
        self.n_qubits         = n_qubits
        self.trainable_params = None
        self.use_statevector  = False

    def initialize_params(self, n_qubits):
        self.n_qubits         = n_qubits
        self.trainable_params = np.random.uniform(0, 2 * np.pi, n_qubits)
        self.use_statevector = bool(QISKIT_AVAILABLE and n_qubits <= MAX_STATEVECTOR_QUBITS)
        if self.use_statevector:
            print(f"[quantum] VQC: {n_qubits} qubits (statevector mode)")
        else:
            reason = "Qiskit unavailable" if not QISKIT_AVAILABLE else f"{n_qubits} qubits exceeds local cap of {MAX_STATEVECTOR_QUBITS}"
            print(f"[quantum] VQC: {n_qubits} qubits ({reason}; classical fallback)")

    def _encode_angles(self, v):
        v = np.clip(v, -1.0, 1.0)
        m = np.abs(v).max()
        if m > 0:
            v = v / m
        return np.arccos(v)

    def _run_circuit(self, encoding_angles, trainable_angles):
        n  = self.n_qubits
        qc = QuantumCircuit(n)
        for i in range(n):
            qc.ry(float(encoding_angles[i]), i)
        for i in range(n - 1):
            qc.cx(i, i + 1)
        for i in range(n):
            qc.ry(float(trainable_angles[i]), i)
        sv    = Statevector(qc)
        probs = sv.probabilities_dict()
        exp   = np.zeros(n)
        for bs, p in probs.items():
            bits = [int(b) for b in reversed(bs)]
            for q in range(n):
                if q < len(bits):
                    exp[q] += p * (1 - 2 * bits[q])
        return exp

    def transform_single(self, pca_vector):
        if not self.use_statevector:
            return self._fallback(pca_vector)
        try:
            angles = self._encode_angles(pca_vector[:self.n_qubits])
            return self._run_circuit(angles, self.trainable_params)
        except Exception as e:
            return self._fallback(pca_vector)

    def transform_batch(self, pca_matrix, verbose=False):
        n  = pca_matrix.shape[0]
        out = np.zeros((n, self.n_qubits))
        for i in range(n):
            if verbose and i % 200 == 0:
                print(f"[quantum] {i}/{n}...")
            out[i] = self.transform_single(pca_matrix[i])
        return out

    def parameter_shift_gradient(self, pca_vector, target):
        n      = self.n_qubits
        grads  = np.zeros(n)
        orig   = self.trainable_params.copy()
        angles = self._encode_angles(pca_vector[:n])
        for i in range(n):
            pp = orig.copy(); pp[i] += np.pi / 2
            pm = orig.copy(); pm[i] -= np.pi / 2
            lp = np.mean((self._run_circuit(angles, pp) - target) ** 2)
            lm = np.mean((self._run_circuit(angles, pm) - target) ** 2)
            grads[i] = (lp - lm) / 2
        self.trainable_params = orig
        return grads

    def _fallback(self, pca_vector):
        v   = np.clip(pca_vector[:self.n_qubits], -1, 1)
        out = np.cos(np.arccos(v))
        for i in range(len(v) - 1):
            out[i] *= np.cos(v[i] * v[i + 1])
        return out

    def save(self, path=None):
        path = path or os.path.join(ARTIFACTS_DIR, "quantum_params.pkl")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"n_qubits": self.n_qubits,
                         "trainable_params": self.trainable_params,
                         "qiskit_available": QISKIT_AVAILABLE,
                         "use_statevector": self.use_statevector}, f)
        print(f"[quantum] Quantum params saved → {path}")

    @staticmethod
    def load(path=None):
        path = path or os.path.join(ARTIFACTS_DIR, "quantum_params.pkl")
        with open(path, "rb") as f:
            d = pickle.load(f)
        q = QuantumFeatureMap(n_qubits=d["n_qubits"])
        q.trainable_params = d["trainable_params"]
        q.use_statevector = bool(d.get(
            "use_statevector",
            QISKIT_AVAILABLE and q.n_qubits <= MAX_STATEVECTOR_QUBITS,
        ))
        print(f"[quantum] Quantum params loaded ({d['n_qubits']} qubits)")
        return q


class QuantumMarketClassifier:
    """Small supervised variational classifier using a Qiskit statevector."""

    def __init__(self, n_qubits=4, n_classes=3, seed=42):
        self.n_qubits = n_qubits
        self.n_classes = n_classes
        self.seed = seed
        self.scaler = StandardScaler()
        self.decomposer = PCA(n_components=n_qubits)
        self.trainable_params = np.random.default_rng(seed).uniform(
            -0.1, 0.1, size=n_qubits * 2
        )
        self.is_fitted = False

    def _class_probabilities(self, encoded_features, parameters):
        if not QISKIT_AVAILABLE:
            raise RuntimeError("Qiskit is required to train the quantum classifier")
        circuit = QuantumCircuit(self.n_qubits)
        for qubit, value in enumerate(encoded_features):
            circuit.ry(float(np.pi * np.tanh(value)), qubit)
        for qubit in range(self.n_qubits - 1):
            circuit.cx(qubit, qubit + 1)
        for layer in range(2):
            for qubit in range(self.n_qubits):
                circuit.ry(float(parameters[layer * self.n_qubits + qubit]), qubit)
            for qubit in range(self.n_qubits):
                circuit.cx(qubit, (qubit + 1) % self.n_qubits)

        state_probabilities = Statevector(circuit).probabilities()
        class_probabilities = np.zeros(self.n_classes, dtype=np.float64)
        np.add.at(
            class_probabilities,
            np.arange(len(state_probabilities)) % self.n_classes,
            state_probabilities,
        )
        return class_probabilities / max(class_probabilities.sum(), 1e-12)

    def fit(self, X, y, epochs=4, batch_size=16, learning_rate=0.08, max_samples=96):
        if not QISKIT_AVAILABLE:
            raise RuntimeError("Qiskit is required to train the quantum classifier")
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32).reshape(-1)
        if X.ndim != 2 or len(X) != len(y) or X.shape[1] < self.n_qubits:
            raise ValueError("Quantum classifier expects aligned 2D features and labels")
        if len(X) < self.n_qubits or np.any((y < 0) | (y >= self.n_classes)):
            raise ValueError("Quantum classifier received too few samples or invalid labels")

        X_scaled = self.scaler.fit_transform(X)
        X_encoded = self.decomposer.fit_transform(X_scaled)
        rng = np.random.default_rng(self.seed)
        per_class = max(1, max_samples // max(1, len(np.unique(y))))
        selected = []
        for class_id in np.unique(y):
            class_indices = np.flatnonzero(y == class_id)
            if len(class_indices) > per_class:
                positions = np.linspace(0, len(class_indices) - 1, per_class, dtype=int)
                class_indices = class_indices[positions]
            selected.extend(class_indices.tolist())
        selected = np.asarray(selected, dtype=np.int32)
        X_encoded, y = X_encoded[selected], y[selected]

        for _ in range(epochs):
            order = rng.permutation(len(y))
            for start in range(0, len(order), batch_size):
                batch = order[start:start + batch_size]
                gradient = np.zeros_like(self.trainable_params)
                for sample_index in batch:
                    features = X_encoded[sample_index]
                    target = y[sample_index]
                    base = self._class_probabilities(features, self.trainable_params)
                    target_probability = max(base[target], 1e-8)
                    for parameter_index in range(len(self.trainable_params)):
                        plus = self.trainable_params.copy()
                        minus = self.trainable_params.copy()
                        plus[parameter_index] += np.pi / 2
                        minus[parameter_index] -= np.pi / 2
                        probability_gradient = 0.5 * (
                            self._class_probabilities(features, plus)[target]
                            - self._class_probabilities(features, minus)[target]
                        )
                        gradient[parameter_index] -= probability_gradient / target_probability
                self.trainable_params -= learning_rate * gradient / len(batch)

        self.training_samples = int(len(y))
        self.trained_epochs = int(epochs)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        if not self.is_fitted:
            raise RuntimeError("Quantum classifier has not been fitted")
        X = np.asarray(X, dtype=np.float64)
        X_encoded = self.decomposer.transform(self.scaler.transform(X))
        return np.asarray([
            self._class_probabilities(row, self.trainable_params)
            for row in X_encoded
        ])

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)


class QuantumPreprocessor:
    def __init__(self):
        self.scaler     = StandardScaler()
        self.decomposer = EigenspaceDecomposer()
        self.qfm        = QuantumFeatureMap()
        self.is_fitted  = False

    def fit(self, X_2d, n_quantum_train_samples=200):
        print("[quantum] 1/3 StandardScaler...")
        X_sc   = self.scaler.fit_transform(X_2d)
        print("[quantum] 2/3 PCA...")
        X_pca  = self.decomposer.fit_transform(X_sc)
        n_comp = self.decomposer.n_components
        print(f"[quantum] 3/3 VQC ({n_comp} qubits)...")
        self.qfm.initialize_params(n_comp)
        if self.qfm.use_statevector:
            n = min(n_quantum_train_samples, len(X_pca))
            print(f"[quantum] Training VQC on {n} samples...")
            self._train_vqc(X_pca[:n])
        else:
            print("[quantum] Skipping VQC train (safe classical fallback).")
        self.is_fitted = True
        return self

    def _train_vqc(self, X_pca, lr=0.05, epochs=8):
        n = len(X_pca)
        for ep in range(epochs):
            loss = 0
            for i in range(n):
                out    = self.qfm.transform_single(X_pca[i])
                target = np.clip(X_pca[i][:self.qfm.n_qubits], -1, 1)
                loss  += np.mean((out - target) ** 2)
                self.qfm.trainable_params -= lr * self.qfm.parameter_shift_gradient(X_pca[i], target)
            if (ep + 1) % 2 == 0:
                print(f"[quantum]   epoch {ep+1}/{epochs} loss {loss/n:.6f}")

    def transform(self, X_2d, verbose=False):
        X_sc  = self.scaler.transform(X_2d)
        X_pca = self.decomposer.transform(X_sc)
        return self.qfm.transform_batch(X_pca, verbose=verbose)

    def transform_windows(self, X_3d, verbose=False):
        n, w, f = X_3d.shape
        return self.transform(X_3d.reshape(-1, f), verbose).reshape(n, w, -1)

    def pca_only_transform_windows(self, X_3d):
        n, w, f = X_3d.shape
        X_sc    = self.scaler.transform(X_3d.reshape(-1, f))
        X_pc    = self.decomposer.transform(X_sc)
        return X_pc.reshape(n, w, -1)

    @property
    def n_qubits(self):
        return self.qfm.n_qubits or 0

    def get_eigenvalue_summary(self):
        if not self.is_fitted:
            return {}
        return {
            "n_components":             self.decomposer.n_components,
            "eigenvalues":              self.decomposer.eigenvalues_.tolist(),
            "explained_variance_ratio": self.decomposer.explained_variance_ratio_.tolist(),
            "cumulative_variance":      np.cumsum(
                self.decomposer.explained_variance_ratio_).tolist(),
        }

    def save(self, artifact_dir=None):
        d = artifact_dir or ARTIFACTS_DIR
        os.makedirs(d, exist_ok=True)
        self.decomposer.save(os.path.join(d, "pca.pkl"))
        self.qfm.save(os.path.join(d, "quantum_params.pkl"))
        with open(os.path.join(d, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)
        print(f"[quantum] All artifacts saved → {d}")

    @staticmethod
    def load(artifact_dir=None):
        d  = artifact_dir or ARTIFACTS_DIR
        qp = QuantumPreprocessor()
        qp.decomposer = EigenspaceDecomposer.load(os.path.join(d, "pca.pkl"))
        qp.qfm        = QuantumFeatureMap.load(os.path.join(d, "quantum_params.pkl"))
        with open(os.path.join(d, "scaler.pkl"), "rb") as f:
            qp.scaler = pickle.load(f)
        qp.is_fitted = True
        return qp

    @staticmethod
    def load_decomposer(artifact_dir=None):
        d = artifact_dir or ARTIFACTS_DIR
        return EigenspaceDecomposer.load(os.path.join(d, "pca.pkl"))

    @staticmethod
    def load_qfm(artifact_dir=None):
        d = artifact_dir or ARTIFACTS_DIR
        return QuantumFeatureMap.load(os.path.join(d, "quantum_params.pkl"))

    @staticmethod
    def load_scaler(artifact_dir=None):
        d = artifact_dir or ARTIFACTS_DIR
        with open(os.path.join(d, "scaler.pkl"), "rb") as f:
            return pickle.load(f)