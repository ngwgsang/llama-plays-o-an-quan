"use client"

import { useEffect, useState } from "react"
import dynamic from "next/dynamic"
import {
  ChevronFirstIcon,
  ChevronLastIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  PauseIcon,
  PlayIcon,
} from "lucide-react"

import { GameBoard } from "@/components/game/game-board"
import { MoveList } from "@/components/game/move-list"
import { ScoreChart } from "@/components/game/score-chart"
import { TeamDot } from "@/components/game/team-badge"
import { TurnDetails } from "@/components/game/turn-details"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Slider } from "@/components/ui/slider"
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group"
import { capturedCells, INITIAL_BOARD } from "@/lib/format"
import type { Board, Turn } from "@/lib/types"

// three.js is only loaded when the 3D view is opened
const Board3D = dynamic(() => import("@/components/game/board-3d/scene"), {
  ssr: false,
  loading: () => <Skeleton className="h-[22rem] w-full rounded-lg sm:h-[28rem]" />,
})

const AUTOPLAY_MS = 900
// In 3D, autoplay waits for the move animation and then pauses briefly
const AUTOPLAY_3D_PAUSE_MS = 450

type ViewMode = "2d" | "3d"

/**
 * Step 0 is the initial board, step i is the board after turn i.
 * In live mode the view follows new turns until the user picks a move.
 */
export function Replay({
  turns,
  initialBoard = INITIAL_BOARD,
  live = false,
  header,
}: {
  turns: Turn[]
  initialBoard?: Board
  live?: boolean
  header?: React.ReactNode
}) {
  const [selected, setSelected] = useState(0)
  const [follow, setFollow] = useState(live)
  const [playing, setPlaying] = useState(false)
  const [mode, setMode] = useState<ViewMode>("2d")
  const [animatedStep, setAnimatedStep] = useState(-1)

  const last = turns.length
  const step = follow ? last : Math.min(selected, last)
  const turn = step > 0 ? turns[step - 1] : undefined
  const board = turn?.board ?? (step === 0 ? initialBoard : undefined)

  const goTo = (next: number) => {
    const clamped = Math.max(0, Math.min(next, last))
    setSelected(clamped)
    setFollow(live && clamped === last)
  }

  // Autoplay stops by itself once the last move is reached
  const isPlaying = playing && step < last

  useEffect(() => {
    if (!isPlaying) return
    if (mode === "3d" && animatedStep !== step) return
    const timer = setTimeout(
      () => setSelected((current) => Math.min(current + 1, last)),
      mode === "3d" ? AUTOPLAY_3D_PAUSE_MS : AUTOPLAY_MS,
    )
    return () => clearTimeout(timer)
  }, [isPlaying, last, mode, animatedStep, step])

  const togglePlay = () => {
    setSelected(!isPlaying && step >= last ? 0 : step)
    setFollow(false)
    setPlaying(!isPlaying)
  }

  const score = turn?.score ?? { A: 0, B: 0 }
  const highlight = turn && { start: turn.pos, team: turn.team, captured: capturedCells(turn.steps) }
  const prevBoard = step === 0 ? undefined : step === 1 ? initialBoard : turns[step - 2].board

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_22rem]">
      <div className="grid min-w-0 content-start gap-4">
        <Card>
          <CardHeader className="flex flex-row flex-wrap items-start justify-between gap-4">
            <div className="grid gap-1.5">
              {header ?? <CardTitle>Board</CardTitle>}
              <CardDescription>
                {step === 0 ? "Before the first move" : `After move ${step} of ${last}`}
                {live && follow && " · following live"}
              </CardDescription>
            </div>
            <div className="flex items-center gap-4 text-sm">
              <ToggleGroup
                type="single"
                variant="outline"
                size="sm"
                value={mode}
                onValueChange={(value) => value && setMode(value as ViewMode)}
                aria-label="Board view"
              >
                <ToggleGroupItem value="2d" className="px-3">
                  2D
                </ToggleGroupItem>
                <ToggleGroupItem value="3d" className="px-3">
                  3D
                </ToggleGroupItem>
              </ToggleGroup>
              <span className="flex items-center gap-1.5">
                <TeamDot team="A" /> A <span className="text-xl font-semibold tabular-nums">{score.A}</span>
              </span>
              <span className="flex items-center gap-1.5">
                <TeamDot team="B" /> B <span className="text-xl font-semibold tabular-nums">{score.B}</span>
              </span>
            </div>
          </CardHeader>
          <CardContent className="grid gap-4">
            {board && mode === "3d" ? (
              <Board3D
                board={board}
                prevBoard={prevBoard}
                turn={turn}
                step={step}
                score={score}
                prevScore={step >= 2 ? turns[step - 2].score : { A: 0, B: 0 }}
                highlight={highlight}
                onAnimationEnd={setAnimatedStep}
              />
            ) : board ? (
              <GameBoard board={board} highlight={highlight} />
            ) : (
              <p className="rounded-md border border-dashed p-6 text-center text-sm text-muted-foreground">
                This game was recorded without board snapshots, so only the move list is available.
              </p>
            )}
            <div className="flex flex-wrap items-center gap-3">
              <div className="flex items-center gap-1">
                <Button variant="outline" size="icon" aria-label="First move" onClick={() => goTo(0)}>
                  <ChevronFirstIcon />
                </Button>
                <Button variant="outline" size="icon" aria-label="Previous move" onClick={() => goTo(step - 1)}>
                  <ChevronLeftIcon />
                </Button>
                <Button size="icon" aria-label={isPlaying ? "Pause" : "Play"} onClick={togglePlay} disabled={!last}>
                  {isPlaying ? <PauseIcon /> : <PlayIcon />}
                </Button>
                <Button variant="outline" size="icon" aria-label="Next move" onClick={() => goTo(step + 1)}>
                  <ChevronRightIcon />
                </Button>
                <Button variant="outline" size="icon" aria-label="Last move" onClick={() => goTo(last)}>
                  <ChevronLastIcon />
                </Button>
              </div>
              <Slider
                className="min-w-40 flex-1"
                min={0}
                max={Math.max(last, 1)}
                step={1}
                value={[step]}
                onValueChange={([value]) => {
                  setPlaying(false)
                  goTo(value)
                }}
                disabled={!last}
                aria-label="Move"
              />
              <span className="w-16 text-right text-sm text-muted-foreground tabular-nums">
                {step} / {last}
              </span>
            </div>
          </CardContent>
        </Card>
        <TurnDetails turn={turn} index={step} />
      </div>
      <div className="grid content-start gap-4">
        <Card>
          <CardHeader>
            <CardTitle>Moves</CardTitle>
            <CardDescription>↻ clockwise · ↺ counter-clockwise · +points captured</CardDescription>
          </CardHeader>
          <CardContent>
            <MoveList
              turns={turns}
              step={step}
              onSelect={(value) => {
                setPlaying(false)
                goTo(value)
              }}
            />
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Score</CardTitle>
            <CardDescription>Points after each move</CardDescription>
          </CardHeader>
          <CardContent>
            <ScoreChart turns={turns} step={step} />
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
