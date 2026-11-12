"use client"

import { useMemo, useRef, useState } from "react"
import { Canvas, useFrame } from "@react-three/fiber"
import { OrbitControls } from "@react-three/drei"
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib"
import * as THREE from "three"
import { Rotate3dIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { BOARD_ORDER } from "@/lib/format"
import { countBoard, piecesFromBoard, sameBoard, simulateTurn, type Piece, type PieceBoard } from "@/lib/simulate"
import type { Board, Score, Team, Turn } from "@/lib/types"
import { Mandarin, Peasant, TEAM_COLORS } from "./characters"
import { BOARD_TOP, CELL, cellCenter, potCenter, QUAN_EDGE, QUAN_RADIUS, type Vec3 } from "./layout"
import { buildTimeline, restingPositions, type Timeline } from "./timeline"

export type Board3DProps = {
  /** Board after the current step */
  board: Board
  /** Board before the current step, and the turn that led from it to `board` */
  prevBoard?: Board
  turn?: Turn
  step: number
  score: Score
  /** Score before the current step, shown while the move is animating */
  prevScore?: Score
  highlight?: { start?: string; team?: Team; captured?: string[] }
  /** Called once the move for `step` has finished animating (or was shown without animation) */
  onAnimationEnd?: (step: number) => void
}

type View = {
  step: number
  key: string
  pieces: PieceBoard
  render: Piece[]
  rest: Map<string, Vec3>
  timeline: Timeline | null
}

const boardKey = (board: Board) => BOARD_ORDER.map((p) => `${board[p].peasants}${board[p].mandarin ? "m" : ""}`).join(",")
const flatten = (pieces: PieceBoard) => BOARD_ORDER.flatMap((pos) => pieces[pos])

/**
 * Moving forward by exactly one step animates the move from the previous
 * pieces; any other change (slider jump, new game) rebuilds the board.
 */
function nextView(previous: View | null, props: Board3DProps, speed: number): View {
  const key = `${props.step}:${boardKey(props.board)}`
  const { turn, prevBoard } = props
  if (
    previous &&
    turn &&
    prevBoard &&
    props.step === previous.step + 1 &&
    sameBoard(countBoard(previous.pieces), prevBoard)
  ) {
    const sim = simulateTurn(previous.pieces, turn)
    if (sim) {
      const render = new Map<string, Piece>()
      for (const piece of [...flatten(previous.pieces), ...flatten(sim.board)]) render.set(piece.id, piece)
      return {
        step: props.step,
        key,
        pieces: sim.board,
        render: [...render.values()],
        rest: restingPositions(sim.board),
        timeline: buildTimeline(previous.pieces, sim.events, sim.board, speed),
      }
    }
  }
  const pieces = piecesFromBoard(props.board)
  return { step: props.step, key, pieces, render: flatten(pieces), rest: restingPositions(pieces), timeline: null }
}

/** Animation clock shared by the animator and the cell overlays (mutated every frame, never rendered). */
class AnimationClock {
  private view: View | null = null
  private startedAt = 0
  elapsed = Infinity

  /** Restart when a new view arrives; returns seconds since its animation started. */
  tick(view: View, now: number) {
    if (this.view !== view) {
      this.view = view
      this.startedAt = now
    }
    this.elapsed = view.timeline ? now - this.startedAt : Infinity
    return this.elapsed
  }
}

const easeInOut = (p: number) => (p < 0.5 ? 2 * p * p : 1 - (-2 * p + 2) ** 2 / 2)

/** Moves every piece along its hops; positions are written straight to the three.js groups. */
function Animator({
  view,
  groups,
  clock,
  onAnimationEnd,
}: {
  view: View
  groups: Map<string, THREE.Group>
  clock: AnimationClock
  onAnimationEnd?: (step: number) => void
}) {
  const notified = useRef<View | null>(null)

  useFrame((state) => {
    const elapsed = clock.tick(view, state.clock.elapsedTime)
    const timeline = view.timeline

    for (const piece of view.render) {
      const group = groups.get(piece.id)
      if (!group) continue
      const hops = timeline?.tracks.get(piece.id)
      let x: number, y: number, z: number
      let scale = 1
      let flying = false
      let heading: number | null = null

      if (hops?.length) {
        let hop = hops[0]
        for (const h of hops) if (h.t0 <= elapsed) hop = h
        if (elapsed < hop.t0) {
          ;[x, y, z] = hop.from
          if (hop.fade === "in") scale = 0
        } else if (elapsed < hop.t1) {
          const p = (elapsed - hop.t0) / (hop.t1 - hop.t0)
          const e = easeInOut(p)
          x = hop.from[0] + (hop.to[0] - hop.from[0]) * e
          y = hop.from[1] + (hop.to[1] - hop.from[1]) * e + hop.arc * 4 * p * (1 - p)
          z = hop.from[2] + (hop.to[2] - hop.from[2]) * e
          if (hop.fade === "in") scale = p
          if (hop.fade === "out") scale = 1 - THREE.MathUtils.smoothstep(p, 0.65, 1)
          flying = hop.arc > 0.1
          const dx = hop.to[0] - hop.from[0]
          const dz = hop.to[2] - hop.from[2]
          if (Math.hypot(dx, dz) > 0.05) heading = Math.atan2(dx, dz)
        } else {
          ;[x, y, z] = hop.to
          if (hop.fade === "out") scale = 0
        }
      } else {
        const rest = view.rest.get(piece.id)
        if (!rest) {
          group.visible = false
          continue
        }
        ;[x, y, z] = rest
      }

      group.position.set(x, y, z)
      group.scale.setScalar(Math.max(scale, 0.0001))
      group.visible = scale > 0.001
      group.userData.flying = flying
      // Face the direction of travel, then turn back towards the camera
      const target = heading ?? 0
      let delta = target - group.rotation.y
      delta = Math.atan2(Math.sin(delta), Math.cos(delta))
      group.rotation.y += delta * 0.18
    }

    if (notified.current !== view && (!timeline || elapsed >= timeline.duration)) {
      notified.current = view
      onAnimationEnd?.(view.step)
    }
  })

  return null
}

/** Coloured overlay on a cell: hover, start of the move, captures (pulses while the move plays). */
function CellOverlay({
  pos,
  hovered,
  start,
  captured,
  pulses,
  clock,
}: {
  pos: string
  hovered: boolean
  start?: Team
  captured: boolean
  pulses: { kind: string; t: number }[]
  clock: AnimationClock
}) {
  const material = useRef<THREE.MeshBasicMaterial>(null)
  const isQuan = pos === "QA" || pos === "QB"
  const [cx, , cz] = cellCenter(pos)
  const x = isQuan ? (pos === "QA" ? -QUAN_EDGE : QUAN_EDGE) : cx

  useFrame(() => {
    const m = material.current
    if (!m) return
    const elapsed = clock.elapsed
    const pulse = pulses.find((p) => elapsed >= p.t && elapsed < p.t + 0.8)
    if (pulse) {
      const k = 1 - (elapsed - pulse.t) / 0.8
      m.color.set(pulse.kind === "capture" ? "#e34948" : pulse.kind === "blocked" ? "#eda100" : "#ffffff")
      m.opacity = 0.15 + 0.45 * k
    } else if (start) {
      m.color.set(TEAM_COLORS[start])
      m.opacity = hovered ? 0.45 : 0.32
    } else if (captured) {
      m.color.set("#e34948")
      m.opacity = hovered ? 0.3 : 0.18
    } else {
      m.color.set("#ffffff")
      m.opacity = hovered ? 0.22 : 0
    }
  })

  return (
    <mesh
      position={[x, BOARD_TOP + 0.004, isQuan ? 0 : cz]}
      rotation-x={-Math.PI / 2}
      rotation-z={pos === "QA" ? Math.PI : 0}
    >
      {isQuan ? (
        <circleGeometry args={[QUAN_RADIUS - 0.03, 32, -Math.PI / 2, Math.PI]} />
      ) : (
        <planeGeometry args={[CELL - 0.05, CELL - 0.05]} />
      )}
      <meshBasicMaterial ref={material} transparent opacity={0} depthWrite={false} />
    </mesh>
  )
}

/** Where each DOM label is anchored in the scene. */
function cellLabelAnchor(pos: string): Vec3 {
  const [cx] = cellCenter(pos)
  if (pos === "QA" || pos === "QB") return [cx + (pos === "QA" ? -0.35 : 0.35), 0.05, -1.4]
  return [cx, 0.05, pos.startsWith("A") ? -1.42 : 1.42]
}

function potLabelAnchor(team: Team): Vec3 {
  const [x, , z] = potCenter(team)
  return [x + 0.95, 0.2, z]
}

/**
 * Positions DOM overlay elements (labels, speech bubbles) over their 3D anchors
 * every frame. Plain DOM instead of drei <Html>, which mounts one React root per label.
 */
function Projector({
  elements,
  anchor,
}: {
  elements: Map<string, HTMLElement>
  anchor: (key: string, out: THREE.Vector3) => boolean
}) {
  const [point] = useState(() => new THREE.Vector3())
  useFrame(({ camera, size }) => {
    for (const [key, element] of elements) {
      if (!anchor(key, point)) {
        element.style.setProperty("visibility", "hidden")
        continue
      }
      point.project(camera)
      // Keep labels and bubbles fully inside the canvas
      const halfWidth = element.offsetWidth / 2 + 4
      const halfHeight = element.offsetHeight / 2 + 4
      const x = THREE.MathUtils.clamp(((point.x + 1) / 2) * size.width, halfWidth, size.width - halfWidth)
      const y = THREE.MathUtils.clamp(((1 - point.y) / 2) * size.height, halfHeight, size.height - halfHeight)
      element.style.setProperty("transform", `translate(-50%, -50%) translate(${x.toFixed(1)}px, ${y.toFixed(1)}px)`)
      element.style.setProperty("visibility", point.z < 1 ? "visible" : "hidden")
    }
  })
  return null
}

const WOOD_LIGHT = "#c99a62"
const WOOD_DARK = "#5c3b22"
const LINE = "#3d2716"

function BoardModel() {
  const lines = useMemo(() => [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5].map((i) => i * CELL), [])
  return (
    <group>
      {/* Table */}
      <mesh position={[0, -0.12, 0]} receiveShadow>
        <boxGeometry args={[9.6, 0.2, 6]} />
        <meshStandardMaterial color={WOOD_DARK} roughness={0.8} />
      </mesh>
      {/* Playing board: row of ten cells and two half-round Quan cells */}
      <mesh position={[0, BOARD_TOP / 2 - 0.02, 0]} receiveShadow castShadow>
        <boxGeometry args={[QUAN_EDGE * 2, BOARD_TOP + 0.04, CELL * 2]} />
        <meshStandardMaterial color={WOOD_LIGHT} roughness={0.75} />
      </mesh>
      {(["QA", "QB"] as const).map((pos) => (
        <mesh
          key={pos}
          position={[pos === "QA" ? -QUAN_EDGE : QUAN_EDGE, BOARD_TOP / 2 - 0.02, 0]}
          receiveShadow
          castShadow
        >
          <cylinderGeometry args={[QUAN_RADIUS, QUAN_RADIUS, BOARD_TOP + 0.04, 40, 1, false, pos === "QA" ? Math.PI : 0, Math.PI]} />
          <meshStandardMaterial color={WOOD_LIGHT} roughness={0.75} />
        </mesh>
      ))}
      {/* Carved lines */}
      <mesh position={[0, BOARD_TOP + 0.002, 0]}>
        <boxGeometry args={[QUAN_EDGE * 2, 0.004, 0.035]} />
        <meshStandardMaterial color={LINE} />
      </mesh>
      {lines.map((x) => (
        <mesh key={x} position={[x, BOARD_TOP + 0.002, 0]}>
          <boxGeometry args={[0.035, 0.004, CELL * 2]} />
          <meshStandardMaterial color={LINE} />
        </mesh>
      ))}
      {(["A", "B"] as const).map((team) => (
        <Pot key={team} team={team} />
      ))}
    </group>
  )
}

function Pot({ team }: { team: Team }) {
  const [x, , z] = potCenter(team)
  return (
    <group position={[x, 0, z]}>
      <mesh position-y={0.14} castShadow receiveShadow>
        <cylinderGeometry args={[0.42, 0.3, 0.28, 32, 1, true]} />
        <meshStandardMaterial color="#9b5a35" roughness={0.9} side={THREE.DoubleSide} />
      </mesh>
      <mesh position-y={0.02} rotation-x={-Math.PI / 2}>
        <circleGeometry args={[0.3, 32]} />
        <meshStandardMaterial color="#3a2116" />
      </mesh>
      <mesh position-y={0.28} rotation-x={Math.PI / 2}>
        <torusGeometry args={[0.42, 0.025, 8, 40]} />
        <meshStandardMaterial color={TEAM_COLORS[team]} roughness={0.5} />
      </mesh>
    </group>
  )
}

const SPEEDS = ["1", "2", "4"]
const BUBBLE_MS = 1800

export default function Board3D(props: Board3DProps) {
  const [speed, setSpeed] = useState("1")
  const [hovered, setHovered] = useState<string | null>(null)
  const [view, setView] = useState<View>(() => nextView(null, props, Number(speed)))
  const [groups] = useState(() => new Map<string, THREE.Group>())
  const [clock] = useState(() => new AnimationClock())
  const [elements] = useState(() => new Map<string, HTMLElement>())
  const [bubbles, setBubbles] = useState<Record<string, { line: string; height: number }>>({})
  const [bubbleTimers] = useState(() => new Map<string, ReturnType<typeof setTimeout>>())
  const controls = useRef<OrbitControlsImpl>(null)
  const [settledStep, setSettledStep] = useState(-1)

  // Derive the next view while rendering when the step or board changes
  if (view.key !== `${props.step}:${boardKey(props.board)}`) {
    setView(nextView(view, props, Number(speed)))
  }

  // Labels and pot scores show the position before the move until its animation ends
  const settled = !view.timeline || settledStep === view.step
  const shownBoard = settled ? props.board : (props.prevBoard ?? props.board)
  const shownScore = settled ? props.score : (props.prevScore ?? props.score)
  const onAnimationEnd = (step: number) => {
    setSettledStep(step)
    props.onAnimationEnd?.(step)
  }

  const cellOf = useMemo(() => {
    const map = new Map<string, string>()
    for (const pos of BOARD_ORDER) for (const piece of view.pieces[pos]) map.set(piece.id, pos)
    return map
  }, [view.pieces])

  const pulses = useMemo(() => {
    const byCell = new Map<string, { kind: string; t: number }[]>()
    for (const mark of view.timeline?.marks ?? []) byCell.set(mark.pos, [...(byCell.get(mark.pos) ?? []), mark])
    return byCell
  }, [view.timeline])

  const register = (id: string) => (group: THREE.Group | null) => {
    if (group) groups.set(id, group)
    else groups.delete(id)
  }

  const overlay = (key: string) => (element: HTMLElement | null) => {
    if (element) elements.set(key, element)
    else elements.delete(key)
  }

  const speak = (id: string, height: number) => (line: string) => {
    setBubbles((current) => ({ ...current, [id]: { line, height } }))
    clearTimeout(bubbleTimers.get(id))
    bubbleTimers.set(
      id,
      setTimeout(() => {
        setBubbles((current) => Object.fromEntries(Object.entries(current).filter(([key]) => key !== id)))
      }, BUBBLE_MS),
    )
  }

  const anchor = (key: string, out: THREE.Vector3) => {
    const [kind, id] = key.split(":")
    if (kind === "bubble") {
      const group = groups.get(id)
      if (!group?.visible) return false
      out.copy(group.position)
      out.y += (bubbles[id]?.height ?? 0.6) * group.scale.y
      return true
    }
    out.set(...(kind === "pot" ? potLabelAnchor(id as Team) : cellLabelAnchor(id)))
    return true
  }

  return (
    <div className="relative h-[22rem] w-full overflow-hidden rounded-lg border bg-gradient-to-b from-muted/40 to-muted sm:h-[28rem]">
      <Canvas
        shadows
        dpr={[1, 2]}
        camera={{ position: [0, 4.6, 5.6], fov: 42 }}
        onPointerMissed={() => setHovered(null)}
        fallback={<p className="p-6 text-sm text-muted-foreground">WebGL is not available in this browser.</p>}
      >
        <hemisphereLight args={["#fff6e5", "#3b2a1a", 0.9]} />
        <directionalLight
          position={[4, 8, 5]}
          intensity={1.6}
          castShadow
          shadow-mapSize={[2048, 2048]}
          shadow-camera-left={-6}
          shadow-camera-right={6}
          shadow-camera-top={4}
          shadow-camera-bottom={-4}
        />
        <BoardModel />

        {BOARD_ORDER.map((pos) => {
          const isQuan = pos === "QA" || pos === "QB"
          const [cx, , cz] = cellCenter(pos)
          return (
            <group key={pos}>
              {/* Invisible hit area for hover */}
              <mesh
                position={[isQuan ? (pos === "QA" ? -QUAN_EDGE : QUAN_EDGE) : cx, BOARD_TOP + 0.006, isQuan ? 0 : cz]}
                rotation-x={-Math.PI / 2}
                rotation-z={pos === "QA" ? Math.PI : 0}
                onPointerOver={(e) => {
                  e.stopPropagation()
                  setHovered(pos)
                }}
                onPointerOut={() => setHovered((h) => (h === pos ? null : h))}
              >
                {isQuan ? (
                  <circleGeometry args={[QUAN_RADIUS, 32, -Math.PI / 2, Math.PI]} />
                ) : (
                  <planeGeometry args={[CELL, CELL]} />
                )}
                <meshBasicMaterial transparent opacity={0} depthWrite={false} />
              </mesh>
              <CellOverlay
                pos={pos}
                hovered={hovered === pos}
                start={props.highlight?.start === pos ? props.highlight.team : undefined}
                captured={Boolean(props.highlight?.captured?.includes(pos))}
                pulses={pulses.get(pos) ?? []}
                clock={clock}
              />
            </group>
          )
        })}

        {view.render.map((piece, i) =>
          piece.kind === "mandarin" ? (
            <Mandarin
              key={piece.id}
              team={piece.owner ?? "A"}
              groupRef={register(piece.id)}
              excited={hovered !== null && cellOf.get(piece.id) === hovered}
              seed={i * 1.7}
              onSpeak={speak(piece.id, 0.95)}
            />
          ) : (
            <Peasant
              key={piece.id}
              groupRef={register(piece.id)}
              excited={hovered !== null && cellOf.get(piece.id) === hovered}
              seed={i * 1.7}
              onSpeak={speak(piece.id, 0.6)}
            />
          ),
        )}

        <Projector elements={elements} anchor={anchor} />
        <Animator view={view} groups={groups} clock={clock} onAnimationEnd={onAnimationEnd} />
        <OrbitControls
          ref={controls}
          makeDefault
          enablePan={false}
          minDistance={4.5}
          maxDistance={13}
          maxPolarAngle={1.35}
          target={[0, 0, 0]}
        />
      </Canvas>

      {/* Labels and speech bubbles, positioned every frame by <Projector /> */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden>
        {BOARD_ORDER.map((pos) => {
          const cell = shownBoard[pos]
          const isQuan = pos === "QA" || pos === "QB"
          const isHovered = hovered === pos
          return (
            <div
              key={pos}
              ref={overlay(`cell:${pos}`)}
              className={`invisible absolute top-0 left-0 rounded px-1.5 py-0.5 text-[11px] font-medium whitespace-nowrap tabular-nums shadow-sm ${
                isHovered ? "z-10 bg-foreground text-background" : "bg-background/85 text-foreground"
              }`}
            >
              {pos} · {isQuan ? cell.peasants + (cell.mandarin ? 10 : 0) : cell.peasants}
              {isHovered && (
                <span className="font-normal">
                  {" "}
                  ({cell.mandarin ? "mandarin + " : ""}
                  {cell.peasants} peasant{cell.peasants === 1 ? "" : "s"})
                </span>
              )}
            </div>
          )
        })}
        {(["A", "B"] as const).map((team) => (
          <div
            key={team}
            ref={overlay(`pot:${team}`)}
            className="invisible absolute top-0 left-0 flex items-center gap-1.5 rounded-full border bg-background/90 px-2.5 py-1 text-xs font-medium whitespace-nowrap shadow-sm"
          >
            <span className="size-2 rounded-full" style={{ background: TEAM_COLORS[team] }} />
            Team {team}
            <span className="font-semibold tabular-nums">{shownScore[team]}</span>
          </div>
        ))}
        {Object.entries(bubbles).map(([id, bubble]) => (
          <div
            key={id}
            ref={overlay(`bubble:${id}`)}
            className="invisible absolute top-0 left-0 z-20 rounded-full border bg-background px-2.5 py-1 text-xs font-medium whitespace-nowrap text-foreground shadow-md"
          >
            {bubble.line}
          </div>
        ))}
      </div>

      <div className="pointer-events-none absolute inset-x-2 top-2 flex items-start justify-between gap-2">
        <p className="rounded bg-background/80 px-2 py-1 text-xs text-muted-foreground">
          Drag to orbit · scroll to zoom · hover a cell · click a character
        </p>
        <div className="pointer-events-auto flex items-center gap-1 rounded-md bg-background/80 p-0.5">
          <ToggleGroup
            type="single"
            size="sm"
            value={speed}
            onValueChange={(value) => value && setSpeed(value)}
            aria-label="Animation speed"
          >
            {SPEEDS.map((s) => (
              <ToggleGroupItem key={s} value={s} className="px-2 text-xs">
                {s}×
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
          <Button variant="ghost" size="icon-sm" aria-label="Reset camera" onClick={() => controls.current?.reset()}>
            <Rotate3dIcon />
          </Button>
        </div>
      </div>
    </div>
  )
}
