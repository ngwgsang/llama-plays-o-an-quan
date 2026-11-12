import { AlertTriangleIcon, ShuffleIcon } from "lucide-react"

import { TeamBadge } from "@/components/game/team-badge"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { formatDirection, shortModel } from "@/lib/format"
import type { Turn } from "@/lib/types"

export function TurnDetails({ turn, index }: { turn?: Turn; index: number }) {
  if (!turn) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Initial position</CardTitle>
          <CardDescription>5 peasants in every cell, one mandarin in each Quan. Team A moves first.</CardDescription>
        </CardHeader>
      </Card>
    )
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-center gap-2">
          <CardTitle>
            Move {index} · Round {turn.round}
          </CardTitle>
          <TeamBadge team={turn.team} label={`Team ${turn.team} · ${turn.agent}`} />
          {turn.invalid_attempts > 0 && (
            <Badge variant="outline" className="gap-1">
              <AlertTriangleIcon className="text-chart-4" /> {turn.invalid_attempts} illegal output
              {turn.invalid_attempts > 1 ? "s" : ""}
            </Badge>
          )}
          {turn.fallback && (
            <Badge variant="outline" className="gap-1">
              <ShuffleIcon /> random fallback
            </Badge>
          )}
        </div>
        <CardDescription>
          {shortModel(turn.model)} played <span className="font-medium text-foreground">{turn.pos}</span>{" "}
          {formatDirection(turn.way)} · captured{" "}
          <span className="font-medium text-foreground">{turn.captured}</span> point{turn.captured === 1 ? "" : "s"}
          {turn.model !== "random" && <> · {turn.output_tokens} tokens generated</>}
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4 text-sm">
        {turn.observation && (
          <section>
            <h3 className="mb-1 text-xs font-medium tracking-wide text-muted-foreground uppercase">Observation</h3>
            <p className="leading-relaxed">{turn.observation}</p>
          </section>
        )}
        <section>
          <h3 className="mb-1 text-xs font-medium tracking-wide text-muted-foreground uppercase">Reasoning</h3>
          <p className="leading-relaxed whitespace-pre-wrap">
            {turn.reason || <span className="text-muted-foreground">No reasoning (fallback move).</span>}
          </p>
        </section>
        {turn.error && (
          <section>
            <h3 className="mb-1 text-xs font-medium tracking-wide text-muted-foreground uppercase">Last error</h3>
            <p className="font-mono text-xs break-all text-muted-foreground">{turn.error}</p>
          </section>
        )}
        <section>
          <h3 className="mb-1 text-xs font-medium tracking-wide text-muted-foreground uppercase">Steps</h3>
          <ol className="grid gap-1 font-mono text-xs">
            {turn.steps.map((step, i) => (
              <li key={i} className="rounded bg-muted px-2 py-1">
                {step}
              </li>
            ))}
          </ol>
        </section>
      </CardContent>
    </Card>
  )
}
