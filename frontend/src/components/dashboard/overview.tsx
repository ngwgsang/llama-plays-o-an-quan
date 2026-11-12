"use client"

import Link from "next/link"
import { ArrowRightIcon, PlayIcon, TrophyIcon } from "lucide-react"

import { ApiErrorAlert, LoadingGrid } from "@/components/api-state"
import { DepthChart } from "@/components/dashboard/depth-chart"
import { ModelChart } from "@/components/dashboard/model-chart"
import { PersonaChart } from "@/components/dashboard/persona-chart"
import { ReasoningChart } from "@/components/dashboard/reasoning-chart"
import { StatCards } from "@/components/dashboard/stat-cards"
import { GamesTable } from "@/components/game/games-table"
import { Button } from "@/components/ui/button"
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty"
import { useApi } from "@/lib/api"
import type { GameSummary, Stats } from "@/lib/types"

const RECENT_GAMES = 5

export function Overview() {
  const stats = useApi<Stats>("/stats", { refreshInterval: 10000 })
  const games = useApi<GameSummary[]>("/games", { refreshInterval: 10000 })

  if (stats.error) return <ApiErrorAlert error={stats.error} />
  if (!stats.data) return <LoadingGrid />

  if (stats.data.totals.games === 0) {
    return (
      <Empty className="border">
        <EmptyHeader>
          <EmptyMedia variant="icon">
            <TrophyIcon />
          </EmptyMedia>
          <EmptyTitle>No games yet</EmptyTitle>
          <EmptyDescription>
            Play a game from the dashboard, or run the experiments from the CLI
            (<code className="font-mono">python -m o_an_quan.experiments personas</code>). Results in{" "}
            <code className="font-mono">results/*.jsonl</code> show up here automatically.
          </EmptyDescription>
        </EmptyHeader>
        <EmptyContent>
          <Button asChild>
            <Link href="/play">
              <PlayIcon /> Start a game
            </Link>
          </Button>
        </EmptyContent>
      </Empty>
    )
  }

  return (
    <div className="grid gap-4">
      <StatCards stats={stats.data} />
      <div className="grid gap-4 xl:grid-cols-2">
        <PersonaChart rows={stats.data.personas} />
        <ModelChart rows={stats.data.models} />
        <DepthChart rows={stats.data.depth} />
        <ReasoningChart rows={stats.data.reasoning} />
      </div>
      <Card>
        <CardHeader>
          <CardTitle>Recent games</CardTitle>
          <CardDescription>Click a game to replay it move by move</CardDescription>
          <CardAction>
            <Button variant="outline" size="sm" asChild>
              <Link href="/games">
                All games <ArrowRightIcon />
              </Link>
            </Button>
          </CardAction>
        </CardHeader>
        <CardContent>
          {games.data ? <GamesTable games={games.data.slice(0, RECENT_GAMES)} /> : null}
        </CardContent>
      </Card>
    </div>
  )
}
