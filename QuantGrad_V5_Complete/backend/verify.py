"""
verify.py
=========
Run this after installing requirements to confirm all dependencies work.
  python verify.py
"""

import sys
print("=" * 55)
print("QuantGrad — Environment Verification")
print("=" * 55)

errors = []

def check(name, fn):
    try:
        result = fn()
        print(f"  OK  {name}: {result}")
    except Exception as e:
        print(f"  FAIL {name}: {e}")
        errors.append(name)

print("\n[Python]")
check("Python version", lambda: sys.version.split()[0])

print("\n[Core ML]")
check("TensorFlow", lambda: __import__("tensorflow").__version__)
check("Scikit-learn", lambda: __import__("sklearn").__version__)
check("Imbalanced-learn", lambda: (
    __import__("imblearn.over_sampling", fromlist=["SMOTE"]),
    "OK"
)[1])
check("NumPy", lambda: __import__("numpy").__version__)
check("Pandas", lambda: __import__("pandas").__version__)
check("Scipy", lambda: __import__("scipy").__version__)

print("\n[Quantum]")
check("Qiskit", lambda: __import__("qiskit").__version__)
check("Qiskit Aer", lambda: (
    __import__("qiskit_aer", fromlist=["AerSimulator"]).AerSimulator(),
    "AerSimulator OK"
)[1])

print("\n[Data, API & Terminal]")
check("FastAPI",    lambda: __import__("fastapi").__version__)
check("Uvicorn",    lambda: __import__("uvicorn").__version__)
check("yfinance",   lambda: __import__("yfinance").__version__)
check("requests",   lambda: __import__("requests").__version__)
check("Matplotlib", lambda: __import__("matplotlib").__version__)
check("Seaborn",    lambda: __import__("seaborn").__version__)
check("Joblib",     lambda: __import__("joblib").__version__)
check("dotenv",     lambda: (__import__("dotenv"), "OK")[1])

print("\n[Quick Quantum Test]")
def _qiskit_test():
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    import numpy as np
    qc = QuantumCircuit(2)
    qc.ry(np.pi / 4, 0)
    qc.cx(0, 1)
    sv = Statevector(qc)
    probs = sv.probabilities()
    return f"2-qubit circuit OK, probs sum={probs.sum():.4f}"
check("Qiskit circuit execution", _qiskit_test)

print("\n[File Structure]")
import os
files = [
    "feature_engine.py",
    "quantum_layer.py",
    "trainer_v3.py",
    "market_structure.py",
    "macro_fetcher.py",
    "server.py",
    "requirements.txt",
    ".env.example",
]
for f in files:
    exists = os.path.exists(f)
    print(f"  {'OK' if exists else 'MISSING'}  {f}")
    if not exists:
        errors.append(f)

print("\n" + "=" * 55)
if errors:
    print(f"FAILED: {len(errors)} issue(s) found:")
    for e in errors:
        print(f"  - {e}")
    print("\nFix the issues above, then re-run: python verify.py")
else:
    print("All checks passed. You are ready to run QuantGrad.")
    print("\nNext steps:")
    print("  1. Optional: copy .env.example to .env and set FRED_API_KEY")
    print("  2. python macro_fetcher.py    (fetch macro data)")
    print("  3. python trainer_v3.py --quick (quick test train)")
    print("  4. python server.py  (launch terminal at http://localhost:8000)")
print("=" * 55)
