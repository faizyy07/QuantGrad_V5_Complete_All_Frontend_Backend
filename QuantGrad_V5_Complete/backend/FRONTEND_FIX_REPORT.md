# QuantGrad Frontend Rebuild and Stability Fix

## What was corrected

The original upload did not include the React source directory referenced by its Vite configuration, so the dashboard failure could not be repaired in place. The project now includes a complete `client/` source tree and a production `web/` build that the supplied Python server serves directly.

The rebuilt dashboard avoids the one-second disappearance failure mode by treating API calls as an optional data enhancement rather than the condition for rendering the page. The terminal mounts from stable local preview data. If `/api/status` or `/api/analyze` is unavailable, malformed, or returns the expected `503` while trained artifacts are missing, the error is caught, a clear preview/offline state is shown, and the dashboard remains mounted and interactive.

## Run locally

| Task | Command |
|---|---|
| Install JavaScript dependencies | `npm install` or `pnpm install` |
| Build the frontend into `web/` | `npm run build` or `pnpm build` |
| Run the Python application | `python server.py` |
| Development frontend with API proxy | Run `python server.py`, then `npm run dev` |

Open `http://127.0.0.1:8000` when using the Python server, or the Vite address shown by `npm run dev` in development.

## Verification completed

The delivered repository was type-checked with `tsc --noEmit`, production-built with Vite, and served through the supplied Python API server. The root document and compiled JavaScript returned HTTP `200`. The backend correctly returned its artifact-missing status and the expected `503` analysis response; the frontend was designed and visually verified to display its preview state instead of unmounting in that case.

## Design implementation

The interface synthesizes the supplied Figma references into a Cobalt Signal Desk: a command-center layout with a watchlist and model state at left, a dominant market evidence workspace in the center, and a decision ledger at right. Its brand mark and visual motifs are chart-derived confidence topology, not generic decorative AI imagery. See `ideas.md` and `reference-layout-notes.md` for the selected system and source-reference observations.
