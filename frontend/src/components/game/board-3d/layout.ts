/**
 * World layout of the 3D board. Seen from above (camera on team B's side),
 * team A's row is the far row (-z) and B's row the near row (+z), so the
 * backend's clockwise order QA -> A1..A5 -> QB -> B5..B1 runs clockwise,
 * matching the 2D board.
 */
import type { Team } from "@/lib/types"

export type Vec3 = [number, number, number]

export const CELL = 1.2
export const BOARD_TOP = 0.12 // y of the playing surface
const PEASANT_SPACING = 0.27
const LAYER_HEIGHT = 0.44

export const QUAN_RADIUS = CELL
export const QUAN_EDGE = 2.5 * CELL // x where the row of small cells ends

export function cellCenter(pos: string): Vec3 {
  if (pos === "QA") return [-QUAN_EDGE - 0.5, BOARD_TOP, 0]
  if (pos === "QB") return [QUAN_EDGE + 0.5, BOARD_TOP, 0]
  const i = Number(pos[1]) - 1
  const x = (i - 2) * CELL
  return [x, BOARD_TOP, pos[0] === "A" ? -CELL / 2 : CELL / 2]
}

/** Slots inside a Quan cell, away from the mandarin standing in the middle. */
const QUAN_SLOTS: [number, number][] = [
  [0, -0.62], [0, 0.62], [-0.28, -0.42], [-0.28, 0.42], [0.28, -0.42], [0.28, 0.42],
  [0, -0.88], [0, 0.88], [-0.3, -0.72], [-0.3, 0.72], [0.3, -0.72], [0.3, 0.72],
  [-0.38, -0.16], [-0.38, 0.16], [0.36, -0.16], [0.36, 0.16],
]

/** Fixed slot of the n-th peasant in a cell (does not depend on the cell's total). */
export function peasantSlot(pos: string, index: number): Vec3 {
  const [cx, cy, cz] = cellCenter(pos)
  if (pos === "QA" || pos === "QB") {
    const [dx, dz] = QUAN_SLOTS[index % QUAN_SLOTS.length]
    const layer = Math.floor(index / QUAN_SLOTS.length)
    const mirror = pos === "QA" ? 1 : -1
    return [cx + dx * mirror, cy + layer * LAYER_HEIGHT, cz + dz]
  }
  const perLayer = 16
  const local = index % perLayer
  const layer = Math.floor(index / perLayer)
  const col = local % 4
  const row = Math.floor(local / 4)
  // Offset odd layers so stacked peasants are visible
  const shift = layer % 2 ? PEASANT_SPACING / 2 : 0
  return [
    cx + (col - 1.5) * PEASANT_SPACING + shift,
    cy + layer * LAYER_HEIGHT,
    cz + (row - 1.5) * PEASANT_SPACING + shift,
  ]
}

export function mandarinSlot(pos: string): Vec3 {
  const [cx, cy, cz] = cellCenter(pos)
  return [cx, cy, cz]
}

/** Where picked-up peasants float while they are being sown. */
export function handSlot(pos: string, index: number): Vec3 {
  const [cx, , cz] = cellCenter(pos)
  const angle = index * 2.4
  const radius = 0.1 * Math.sqrt(index)
  return [cx + Math.cos(angle) * radius, 1.25 + index * 0.015, cz + Math.sin(angle) * radius]
}

/** Score pots behind each team's row. */
export function potCenter(team: Team): Vec3 {
  return [0, BOARD_TOP, team === "A" ? -2.35 : 2.35]
}

export function potSlot(team: Team, seed: number): Vec3 {
  const [x, , z] = potCenter(team)
  const angle = seed * 2.399
  const radius = 0.18 * ((seed % 5) / 5)
  return [x + Math.cos(angle) * radius, 0.3, z + Math.sin(angle) * radius]
}
