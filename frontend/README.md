# Ô Ăn Quan Dashboard

Next.js 16 (App Router) + Tailwind CSS 4 + shadcn/ui dashboard for the
[Ô Ăn Quan study](../README.md).

| Page | What it shows |
|---|---|
| `/` | Totals, Table I (persona scores by phase), Table II (win/draw by model), planning depth, Table III (reasoning types), recent games |
| `/games` | Every saved game, with search and filters |
| `/games/[id]` | Move-by-move replay in 2D or 3D: board, reasoning, steps, score over time |
| `/play` | Start a game (random or LLM agents) and watch it live |
| `/prompts` | Persona and rule prompts given to the agents |

Every chart has a **Table** tab with the same numbers.

### 3D board

`src/components/game/board-3d/` renders the board with React Three Fiber. The
characters are built from three.js primitives (no model files): peasants (*dân*)
wear a conical hat (*nón lá*), mandarins (*quan*) wear a robe in their team colour
and a winged hat (*mũ cánh chuồn*).

- Stepping forward one move replays it piece by piece: pick up, sow cell by cell,
  capture (captured pieces jump into the capturing team's pot). The move is
  re-simulated by `src/lib/simulate.ts`, a port of the backend rules that is checked
  against the recorded board; if they ever disagree the board just snaps.
- Hover a cell to see its contents (its characters cheer), click a character to make it
  jump and talk, drag to orbit, scroll to zoom, 1×/2×/4× animation speed.
- three.js is only downloaded when the 3D view is opened.

## Run

The dashboard needs the FastAPI backend (`backend/src/o_an_quan/api`):

```bash
# from the repository root, with the Python environment active
python -m o_an_quan.api            # http://127.0.0.1:8000

# in this folder
npm install
npm run dev                        # http://localhost:3000
```

Requests to `/api/*` are proxied to the backend (see `next.config.ts`).
Set `BACKEND_URL` if the API runs somewhere other than `http://127.0.0.1:8000`.

## Scripts

```bash
npm run dev     # development server
npm run build   # production build
npm run lint    # ESLint
```
