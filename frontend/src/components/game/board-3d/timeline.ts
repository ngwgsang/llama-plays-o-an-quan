/**
 * Turn a simulated move (pickup / drop / capture events) into per-piece hops
 * that the scene interpolates every frame.
 */
import { BOARD_ORDER } from "@/lib/format"
import type { MoveEvent, Piece, PieceBoard } from "@/lib/simulate"
import { handSlot, mandarinSlot, peasantSlot, potSlot, type Vec3 } from "./layout"

export type Hop = {
  t0: number
  t1: number
  from: Vec3
  to: Vec3
  /** Height of the jump arc */
  arc: number
  /** Scale animation: grow in (restore) or shrink out (captured) */
  fade?: "in" | "out"
}

export type Timeline = {
  tracks: Map<string, Hop[]>
  duration: number
  /** Cells to pulse while the move plays, with the time they become active */
  marks: { pos: string; kind: "pickup" | "capture" | "blocked"; t: number }[]
  /** Set by the scene on the first frame */
  startedAt?: number
}

/** Resting position of every piece on a board. */
export function restingPositions(board: PieceBoard) {
  const positions = new Map<string, Vec3>()
  for (const pos of BOARD_ORDER) {
    let peasant = 0
    for (const piece of board[pos]) {
      positions.set(piece.id, piece.kind === "mandarin" ? mandarinSlot(pos) : peasantSlot(pos, peasant++))
    }
  }
  return positions
}

const PICKUP = 0.34
const PICKUP_STAGGER = 0.035
const DROP = 0.3
const CAPTURE = 0.65
const CAPTURE_STAGGER = 0.05
const RESTORE = 0.5
const SETTLE = 0.3
const PAUSE = 0.25
/** Long sowing chains are sped up so one move never plays longer than this (seconds) */
const MAX_DURATION = 10

export function buildTimeline(start: PieceBoard, events: MoveEvent[], end: PieceBoard, speed = 1): Timeline {
  const current = restingPositions(start)
  const peasantsIn: Record<string, number> = Object.fromEntries(
    BOARD_ORDER.map((pos) => [pos, start[pos].filter((p: Piece) => p.kind === "peasant").length]),
  )
  const tracks = new Map<string, Hop[]>()
  const marks: Timeline["marks"] = []
  let hand: string[] = []
  let T = 0
  let captured = 0

  const hop = (id: string, t0: number, duration: number, to: Vec3, arc: number, fade?: Hop["fade"]) => {
    const from = current.get(id) ?? to
    const list = tracks.get(id) ?? []
    list.push({ t0, t1: t0 + duration, from, to, arc, fade })
    tracks.set(id, list)
    current.set(id, to)
  }

  for (const event of events) {
    switch (event.type) {
      case "restore":
        event.pieces.forEach(({ pos, id }) => {
          const slot = peasantSlot(pos, peasantsIn[pos]++)
          current.set(id, [slot[0], slot[1] + 1.2, slot[2]])
          hop(id, T, RESTORE, slot, 0, "in")
        })
        T += RESTORE + PAUSE
        break
      case "pickup":
        marks.push({ pos: event.pos, kind: "pickup", t: T })
        event.ids.forEach((id, i) => hop(id, T + i * PICKUP_STAGGER, PICKUP, handSlot(event.pos, i), 0.25))
        peasantsIn[event.pos] = 0
        hand = [...event.ids]
        T += PICKUP + event.ids.length * PICKUP_STAGGER
        break
      case "drop": {
        const [dropped, ...rest] = hand
        hop(dropped, T, DROP, peasantSlot(event.pos, peasantsIn[event.pos]++), 0.35)
        rest.forEach((id, i) => hop(id, T, DROP, handSlot(event.pos, i), 0.08))
        hand = rest
        T += DROP
        break
      }
      case "capture":
        T += PAUSE / 2
        marks.push({ pos: event.pos, kind: "capture", t: T })
        event.ids.forEach((id, i) => hop(id, T + i * CAPTURE_STAGGER, CAPTURE, potSlot(event.team, captured++), 1.1, "out"))
        peasantsIn[event.pos] = 0
        T += CAPTURE + event.ids.length * CAPTURE_STAGGER
        break
      case "end":
        if (event.pos) marks.push({ pos: event.pos, kind: "blocked", t: T })
        T += PAUSE
        break
    }
  }

  // Close the gaps left by moved pieces
  const resting = restingPositions(end)
  let settled = false
  for (const [id, to] of resting) {
    const from = current.get(id)
    if (from && Math.hypot(from[0] - to[0], from[1] - to[1], from[2] - to[2]) > 0.01) {
      hop(id, T, SETTLE, to, 0.05)
      settled = true
    }
  }
  if (settled) T += SETTLE

  // Apply playback speed
  speed = Math.max(speed, T / MAX_DURATION)
  for (const list of tracks.values()) {
    for (const h of list) {
      h.t0 /= speed
      h.t1 /= speed
    }
  }
  for (const mark of marks) mark.t /= speed
  return { tracks, duration: T / speed, marks }
}
