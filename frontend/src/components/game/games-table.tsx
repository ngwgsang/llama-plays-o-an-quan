"use client"

import Link from "next/link"
import { useRouter } from "next/navigation"

import { TeamBadge, TeamDot } from "@/components/game/team-badge"
import { Badge } from "@/components/ui/badge"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { formatEndReason, shortModel } from "@/lib/format"
import type { GameSummary } from "@/lib/types"

function Player({ name, model, team }: { name: string; model: string; team: "A" | "B" }) {
  return (
    <div className="flex min-w-0 items-center gap-2">
      <TeamDot team={team} />
      <div className="min-w-0">
        <div className="truncate font-medium">{name === model ? shortModel(model) : name}</div>
        {name !== model && <div className="truncate text-xs text-muted-foreground">{shortModel(model)}</div>}
      </div>
    </div>
  )
}

export function GamesTable({ games }: { games: GameSummary[] }) {
  const router = useRouter()

  return (
    <Table>
      <TableHeader>
        <TableRow>
          <TableHead>Game</TableHead>
          <TableHead>Team A</TableHead>
          <TableHead>Team B</TableHead>
          <TableHead className="text-right">Score</TableHead>
          <TableHead>Winner</TableHead>
          <TableHead className="text-right">Rounds</TableHead>
          <TableHead>End</TableHead>
          <TableHead className="text-right">Illegal</TableHead>
        </TableRow>
      </TableHeader>
      <TableBody>
        {games.map((game) => (
          <TableRow
            key={game.id}
            className="cursor-pointer"
            onClick={() => router.push(`/games/${game.id}`)}
          >
            <TableCell>
              <Link href={`/games/${game.id}`} className="font-mono text-xs hover:underline">
                {game.id}
              </Link>
              <div className="mt-1">
                <Badge variant="secondary">{game.experiment}</Badge>
              </div>
            </TableCell>
            <TableCell className="max-w-48">
              <Player name={game.agent_a} model={game.model_a} team="A" />
            </TableCell>
            <TableCell className="max-w-48">
              <Player name={game.agent_b} model={game.model_b} team="B" />
            </TableCell>
            <TableCell className="text-right font-medium tabular-nums">
              {game.final_score.A} : {game.final_score.B}
            </TableCell>
            <TableCell>
              <TeamBadge team={game.winner} />
            </TableCell>
            <TableCell className="text-right tabular-nums">{game.rounds_played}</TableCell>
            <TableCell className="text-muted-foreground">{formatEndReason(game.end_reason)}</TableCell>
            <TableCell className="text-right tabular-nums">
              {game.invalid_attempts}
              {game.fallbacks > 0 && <span className="text-muted-foreground"> ({game.fallbacks} fb)</span>}
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}
