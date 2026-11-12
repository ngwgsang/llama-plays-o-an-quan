"use client"

import { useMemo, useState } from "react"
import Link from "next/link"
import { PlayIcon, SearchIcon } from "lucide-react"

import { ApiErrorAlert } from "@/components/api-state"
import { GamesTable } from "@/components/game/games-table"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import { useApi } from "@/lib/api"
import type { GameSummary } from "@/lib/types"

const ALL = "all"

export function GamesBrowser() {
  const { data, error } = useApi<GameSummary[]>("/games", { refreshInterval: 10000 })
  const [query, setQuery] = useState("")
  const [experiment, setExperiment] = useState(ALL)
  const [winner, setWinner] = useState(ALL)

  const experiments = useMemo(() => [...new Set(data?.map((g) => g.experiment) ?? [])].sort(), [data])
  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    return (data ?? []).filter(
      (g) =>
        (experiment === ALL || g.experiment === experiment) &&
        (winner === ALL || g.winner === winner) &&
        (!q || [g.id, g.agent_a, g.agent_b, g.model_a, g.model_b].some((v) => v.toLowerCase().includes(q))),
    )
  }, [data, query, experiment, winner])

  if (error) return <ApiErrorAlert error={error} />

  return (
    <Card>
      <CardHeader>
        <CardTitle>All games</CardTitle>
        <CardDescription>
          {data ? `${filtered.length} of ${data.length} games` : "Loading…"} from <code>results/*.jsonl</code>
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4">
        <div className="flex flex-wrap items-center gap-2">
          <div className="relative min-w-56 flex-1">
            <SearchIcon className="absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search by id, agent or model"
              className="pl-8"
              aria-label="Search games"
            />
          </div>
          <Select value={experiment} onValueChange={setExperiment}>
            <SelectTrigger className="w-44" aria-label="Experiment">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={ALL}>All experiments</SelectItem>
              {experiments.map((e) => (
                <SelectItem key={e} value={e}>
                  {e}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Select value={winner} onValueChange={setWinner}>
            <SelectTrigger className="w-36" aria-label="Winner">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={ALL}>Any winner</SelectItem>
              <SelectItem value="A">Team A won</SelectItem>
              <SelectItem value="B">Team B won</SelectItem>
              <SelectItem value="DRAW">Draw</SelectItem>
            </SelectContent>
          </Select>
          <Button asChild className="ml-auto">
            <Link href="/play">
              <PlayIcon /> New game
            </Link>
          </Button>
        </div>
        {!data ? (
          <Skeleton className="h-64" />
        ) : filtered.length ? (
          <GamesTable games={filtered} />
        ) : (
          <p className="rounded-md border border-dashed p-8 text-center text-sm text-muted-foreground">
            {data.length ? "No games match these filters." : "No games yet."}
          </p>
        )}
      </CardContent>
    </Card>
  )
}
