"use client"

import Link from "next/link"
import { ArrowLeftIcon } from "lucide-react"

import { ApiErrorAlert } from "@/components/api-state"
import { Replay } from "@/components/game/replay"
import { TeamBadge, TeamDot } from "@/components/game/team-badge"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { useApi } from "@/lib/api"
import { formatDate, formatEndReason, shortModel } from "@/lib/format"
import type { GameRecord } from "@/lib/types"

export function GameView({ id }: { id: string }) {
  const { data: game, error } = useApi<GameRecord>(`/games/${encodeURIComponent(id)}`)

  return (
    <div className="grid gap-4">
      <div>
        <Button variant="ghost" size="sm" asChild className="-ml-2">
          <Link href="/games">
            <ArrowLeftIcon /> All games
          </Link>
        </Button>
      </div>
      {error ? (
        <ApiErrorAlert error={error} />
      ) : !game ? (
        <Skeleton className="h-[32rem] rounded-xl" />
      ) : (
        <>
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <span className="font-mono text-muted-foreground">{game.id}</span>
            <Badge variant="secondary">{game.experiment ?? "dashboard"}</Badge>
            <span className="text-muted-foreground">·</span>
            <span>{formatEndReason(game.end_reason)}</span>
            <span className="text-muted-foreground">·</span>
            <span>
              Final {game.final_score.A} : {game.final_score.B}
            </span>
            <TeamBadge team={game.winner} label={game.winner === "DRAW" ? "Draw" : `Team ${game.winner} wins`} />
            {game.created_at && <span className="text-muted-foreground">· {formatDate(game.created_at)}</span>}
          </div>
          <Replay
            turns={game.turns}
            initialBoard={game.initial_board}
            header={
              <CardTitle className="flex flex-wrap items-center gap-x-3 gap-y-1">
                <span className="flex items-center gap-1.5">
                  <TeamDot team="A" /> {game.agent_a}
                  <span className="text-sm font-normal text-muted-foreground">({shortModel(game.model_a)})</span>
                </span>
                <span className="text-muted-foreground">vs</span>
                <span className="flex items-center gap-1.5">
                  <TeamDot team="B" /> {game.agent_b}
                  <span className="text-sm font-normal text-muted-foreground">({shortModel(game.model_b)})</span>
                </span>
              </CardTitle>
            }
          />
        </>
      )}
    </div>
  )
}
