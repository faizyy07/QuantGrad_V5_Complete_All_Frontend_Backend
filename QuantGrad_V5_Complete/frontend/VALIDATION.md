# QuantGrad V5 Validation Notes

The V5 terminal was validated after adding the typed bridge from the Node service to the local Python model API. The decision ledger retains its layout while the Python API is starting or unavailable, and presents an intentional unavailable state instead of unmounting or showing the previous configuration placeholder.

| Check | Result |
|---|---|
| Automated tests | 9 passed across 3 test files |
| TypeScript check | Passed with `tsc --noEmit` |
| Production build | Passed with Vite and the Node server bundle |
| Browser rendering | Terminal remains rendered with an explicit local-model status state |

The bridge maps `signal_label`, `confidence`, `risk_level`, `trend`, `structure`, and `adx` from the Python `/api/analyze` response when trained artifacts are available. When the API cannot be reached or artifacts are missing, the terminal shows a readable state and leaves all order execution manual.
