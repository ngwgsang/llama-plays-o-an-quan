"use client"

/**
 * Procedural characters built from three.js primitives (no external assets).
 *
 *   Peasant (dân): brown tunic (áo nâu), conical straw hat (nón lá)
 *   Mandarin (quan): long robe in the team colour, gold belt and badge,
 *                    black winged hat (mũ cánh chuồn)
 *
 * The scene moves the outer group (board position); each character animates
 * its inner group: idle breathing and glances, excited hops when its cell is
 * hovered, arms up while flying between cells, and a jump + spin with a speech
 * bubble when clicked. Characters face +z (towards the default camera).
 */
import { useMemo, useRef } from "react"
import { useFrame, type ThreeEvent } from "@react-three/fiber"
import * as THREE from "three"

import type { Team } from "@/lib/types"

export const TEAM_COLORS: Record<Team, string> = { A: "#2a78d6", B: "#eb6834" }

const SKIN = "#f1c79a"
const DARK = "#1f1b16"

// Shared geometries: ~70 characters on the board at most
const geo = {
  leg: new THREE.CylinderGeometry(0.022, 0.026, 0.1, 8),
  peasantBody: new THREE.CylinderGeometry(0.058, 0.078, 0.17, 12),
  peasantArm: new THREE.CapsuleGeometry(0.018, 0.08, 4, 8),
  peasantHead: new THREE.SphereGeometry(0.066, 16, 12),
  eye: new THREE.SphereGeometry(0.011, 8, 6),
  nonLa: new THREE.ConeGeometry(0.15, 0.095, 24, 1, true),
  nonLaRim: new THREE.TorusGeometry(0.148, 0.006, 6, 24),
  robe: new THREE.CylinderGeometry(0.1, 0.19, 0.44, 20),
  belt: new THREE.TorusGeometry(0.112, 0.016, 8, 24),
  badge: new THREE.BoxGeometry(0.08, 0.08, 0.01),
  sleeve: new THREE.CapsuleGeometry(0.035, 0.16, 4, 8),
  hand: new THREE.SphereGeometry(0.03, 8, 6),
  mandarinHead: new THREE.SphereGeometry(0.095, 20, 14),
  beard: new THREE.ConeGeometry(0.035, 0.09, 10),
  hatCrown: new THREE.CylinderGeometry(0.085, 0.1, 0.1, 20),
  hatTop: new THREE.SphereGeometry(0.085, 16, 8, 0, Math.PI * 2, 0, Math.PI / 2),
  hatWing: new THREE.CapsuleGeometry(0.022, 0.14, 4, 8),
  shadow: new THREE.CircleGeometry(1, 20),
}

const mat = {
  skin: new THREE.MeshStandardMaterial({ color: SKIN, roughness: 0.7 }),
  dark: new THREE.MeshStandardMaterial({ color: DARK, roughness: 0.6 }),
  tunic: new THREE.MeshStandardMaterial({ color: "#8a5a32", roughness: 0.85 }),
  trousers: new THREE.MeshStandardMaterial({ color: "#3b3027", roughness: 0.9 }),
  straw: new THREE.MeshStandardMaterial({ color: "#e6cf8f", roughness: 0.95, side: THREE.DoubleSide }),
  strawRim: new THREE.MeshStandardMaterial({ color: "#c9ab62", roughness: 0.95 }),
  gold: new THREE.MeshStandardMaterial({ color: "#d9a520", roughness: 0.35, metalness: 0.6 }),
  hat: new THREE.MeshStandardMaterial({ color: "#121212", roughness: 0.4 }),
  shadow: new THREE.MeshBasicMaterial({ color: "#000000", transparent: true, opacity: 0.22, depthWrite: false }),
}
const robeMaterials: Record<Team, THREE.MeshStandardMaterial> = {
  A: new THREE.MeshStandardMaterial({ color: TEAM_COLORS.A, roughness: 0.55 }),
  B: new THREE.MeshStandardMaterial({ color: TEAM_COLORS.B, roughness: 0.55 }),
}

const PEASANT_LINES = ["Chào bạn!", "Dân đây!", "Rải tiếp đi!", "Đừng ăn tôi nha!", "Ô ăn quan!"]
const MANDARIN_LINES = ["Ta là Quan!", "Chưa đủ 5 dân thì đừng hòng ăn ta!", "Hừm!", "Dân của ta đâu?"]

const CLICK_ANIMATION = 0.8 // seconds

type CharacterProps = {
  /** Lets the scene position the outer group */
  groupRef: (group: THREE.Group | null) => void
  /** The cell under the pointer contains this character */
  excited: boolean
  /** Random phase so characters do not move in sync */
  seed: number
  /** Shows a speech bubble above the character (rendered by the scene) */
  onSpeak: (line: string) => void
}

/** Shared behaviour: click to jump, spin and talk. */
function useClickReaction(lines: string[], seed: number, onSpeak: (line: string) => void) {
  const clickedAt = useRef(-Infinity)
  const counter = useRef(Math.floor(seed))

  const onClick = (event: ThreeEvent<MouseEvent>) => {
    event.stopPropagation()
    clickedAt.current = performance.now() / 1000
    onSpeak(lines[counter.current++ % lines.length])
  }
  return { clickedAt, onClick }
}

function pointerHandlers() {
  return {
    // No stopPropagation: the cell underneath stays hovered
    onPointerOver: () => {
      document.body.style.cursor = "pointer"
    },
    onPointerOut: () => {
      document.body.style.cursor = ""
    },
  }
}

/** Common per-frame motion of the inner group. Returns flags for limb poses. */
function animateBody(
  inner: THREE.Group,
  outer: THREE.Group | null,
  t: number,
  seed: number,
  excited: boolean,
  clickedAt: number,
) {
  const flying = Boolean(outer?.userData.flying)
  const sinceClick = t - clickedAt
  let y = 0
  let spin = 0
  if (sinceClick < CLICK_ANIMATION) {
    const p = sinceClick / CLICK_ANIMATION
    y = Math.sin(p * Math.PI) * 0.28
    spin = p * Math.PI * 2
  } else if (excited && !flying) {
    y = Math.abs(Math.sin(t * 9 + seed)) * 0.07
  } else if (!flying) {
    y = Math.abs(Math.sin(t * 1.6 + seed)) * 0.008
  }
  inner.position.y = y
  inner.rotation.y = spin
  // Squash slightly when landing from a hop, stretch while flying
  const breathe = 1 + Math.sin(t * 2 + seed) * 0.015
  inner.scale.set(1, flying ? 1.08 : breathe, 1)
  return { flying, waving: excited || sinceClick < CLICK_ANIMATION }
}

export function Peasant({ groupRef, excited, seed, onSpeak }: CharacterProps) {
  const outer = useRef<THREE.Group>(null)
  const inner = useRef<THREE.Group>(null)
  const head = useRef<THREE.Group>(null)
  const leftArm = useRef<THREE.Group>(null)
  const rightArm = useRef<THREE.Group>(null)
  const eyes = useRef<THREE.Group>(null)
  const { clickedAt, onClick } = useClickReaction(PEASANT_LINES, seed, onSpeak)
  const handlers = useMemo(() => pointerHandlers(), [])

  useFrame(({ clock }) => {
    const t = clock.elapsedTime
    if (!inner.current) return
    const { flying, waving } = animateBody(inner.current, outer.current, t, seed, excited, clickedAt.current)
    // Arms: up while flying, waving when excited, hanging otherwise
    const armLift = flying ? 2.6 : waving ? 2.2 + Math.sin(t * 14 + seed) * 0.5 : 0.15 + Math.sin(t * 2 + seed) * 0.05
    if (leftArm.current) leftArm.current.rotation.z = -armLift
    if (rightArm.current) rightArm.current.rotation.z = armLift
    if (head.current) head.current.rotation.y = Math.sin(t * 0.6 + seed * 3) * 0.45
    // Blink every few seconds
    if (eyes.current) eyes.current.scale.y = (t + seed) % 4 < 0.12 ? 0.15 : 1
  })

  return (
    <group
      ref={(g) => {
        outer.current = g
        groupRef(g)
      }}
    >
      <mesh geometry={geo.shadow} material={mat.shadow} rotation-x={-Math.PI / 2} position-y={0.002} scale={0.09} />
      <group ref={inner} onClick={onClick} {...handlers}>
        <mesh geometry={geo.leg} material={mat.trousers} position={[-0.03, 0.05, 0]} castShadow />
        <mesh geometry={geo.leg} material={mat.trousers} position={[0.03, 0.05, 0]} castShadow />
        <mesh geometry={geo.peasantBody} material={mat.tunic} position-y={0.185} castShadow />
        <group ref={leftArm} position={[-0.07, 0.24, 0]}>
          <mesh geometry={geo.peasantArm} material={mat.tunic} position-y={-0.045} castShadow />
        </group>
        <group ref={rightArm} position={[0.07, 0.24, 0]}>
          <mesh geometry={geo.peasantArm} material={mat.tunic} position-y={-0.045} castShadow />
        </group>
        <group ref={head} position-y={0.33}>
          <mesh geometry={geo.peasantHead} material={mat.skin} castShadow />
          <group ref={eyes}>
            <mesh geometry={geo.eye} material={mat.dark} position={[-0.024, 0.005, 0.058]} />
            <mesh geometry={geo.eye} material={mat.dark} position={[0.024, 0.005, 0.058]} />
          </group>
          {/* Nón lá */}
          <mesh geometry={geo.nonLa} material={mat.straw} position-y={0.075} castShadow />
          <mesh geometry={geo.nonLaRim} material={mat.strawRim} position-y={0.028} rotation-x={Math.PI / 2} />
        </group>
      </group>
    </group>
  )
}

export function Mandarin({ groupRef, excited, seed, team, onSpeak }: CharacterProps & { team: Team }) {
  const outer = useRef<THREE.Group>(null)
  const inner = useRef<THREE.Group>(null)
  const head = useRef<THREE.Group>(null)
  const leftArm = useRef<THREE.Group>(null)
  const rightArm = useRef<THREE.Group>(null)
  const wings = useRef<THREE.Group>(null)
  const { clickedAt, onClick } = useClickReaction(MANDARIN_LINES, seed, onSpeak)
  const robe = robeMaterials[team]
  const handlers = useMemo(() => pointerHandlers(), [])

  useFrame(({ clock }) => {
    const t = clock.elapsedTime
    if (!inner.current) return
    const { flying, waving } = animateBody(inner.current, outer.current, t, seed, excited, clickedAt.current)
    // Hands folded in the sleeves at rest; raised when excited, flailing when captured
    const lift = flying ? 2.4 + Math.sin(t * 20) * 0.4 : waving ? 1.6 + Math.sin(t * 10 + seed) * 0.3 : 0.35
    if (leftArm.current) leftArm.current.rotation.z = -lift
    if (rightArm.current) rightArm.current.rotation.z = lift
    if (head.current) head.current.rotation.y = Math.sin(t * 0.4 + seed) * 0.3
    // Hat wings bob as he walks or hops
    if (wings.current) wings.current.rotation.z = Math.sin(t * (flying || waving ? 12 : 1.5) + seed) * 0.08
  })

  return (
    <group
      ref={(g) => {
        outer.current = g
        groupRef(g)
      }}
    >
      <mesh geometry={geo.shadow} material={mat.shadow} rotation-x={-Math.PI / 2} position-y={0.002} scale={0.2} />
      <group ref={inner} onClick={onClick} {...handlers}>
        <mesh geometry={geo.robe} material={robe} position-y={0.22} castShadow />
        <mesh geometry={geo.belt} material={mat.gold} position-y={0.3} rotation-x={Math.PI / 2} />
        <mesh geometry={geo.badge} material={mat.gold} position={[0, 0.36, 0.115]} rotation-x={-0.2} />
        <group ref={leftArm} position={[-0.12, 0.4, 0]}>
          <mesh geometry={geo.sleeve} material={robe} position-y={-0.09} castShadow />
          <mesh geometry={geo.hand} material={mat.skin} position-y={-0.19} />
        </group>
        <group ref={rightArm} position={[0.12, 0.4, 0]}>
          <mesh geometry={geo.sleeve} material={robe} position-y={-0.09} castShadow />
          <mesh geometry={geo.hand} material={mat.skin} position-y={-0.19} />
        </group>
        <group ref={head} position-y={0.53}>
          <mesh geometry={geo.mandarinHead} material={mat.skin} castShadow />
          <mesh geometry={geo.eye} material={mat.dark} position={[-0.034, 0.01, 0.087]} scale={1.2} />
          <mesh geometry={geo.eye} material={mat.dark} position={[0.034, 0.01, 0.087]} scale={1.2} />
          <mesh geometry={geo.beard} material={mat.dark} position={[0, -0.09, 0.06]} rotation-x={Math.PI} />
          {/* Mũ cánh chuồn */}
          <group position-y={0.07}>
            <mesh geometry={geo.hatCrown} material={mat.hat} position-y={0.03} castShadow />
            <mesh geometry={geo.hatTop} material={mat.hat} position-y={0.08} scale={[1, 0.6, 1]} />
            <group ref={wings} position-y={0.06}>
              <mesh geometry={geo.hatWing} material={mat.hat} position-x={-0.17} rotation-z={Math.PI / 2 - 0.2} />
              <mesh geometry={geo.hatWing} material={mat.hat} position-x={0.17} rotation-z={-Math.PI / 2 + 0.2} />
            </group>
          </group>
        </group>
      </group>
    </group>
  )
}
