import { Card, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { formatNumber } from "@/lib/format"
import type { Stats } from "@/lib/types"

function StatCard({ label, value, hint }: { label: string; value: React.ReactNode; hint: React.ReactNode }) {
  return (
    <Card className="gap-3">
      <CardHeader>
        <CardDescription>{label}</CardDescription>
        <CardTitle className="text-3xl font-semibold tabular-nums">{value}</CardTitle>
      </CardHeader>
      <CardFooter className="text-sm text-muted-foreground">{hint}</CardFooter>
    </Card>
  )
}

export function StatCards({ stats }: { stats: Stats }) {
  const { totals, winners } = stats
  const decided = (winners.A ?? 0) + (winners.B ?? 0) + (winners.DRAW ?? 0)

  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <StatCard
        label="Games played"
        value={totals.games}
        hint={
          decided
            ? `A won ${winners.A ?? 0} · B won ${winners.B ?? 0} · ${winners.DRAW ?? 0} draws`
            : "No games yet"
        }
      />
      <StatCard
        label="LLM moves"
        value={totals.llm_moves}
        hint={`${totals.classified_moves} with a classified reasoning type`}
      />
      <StatCard
        label="Illegal outputs per move"
        value={formatNumber(totals.invalid_per_move, 2)}
        hint={
          totals.fallback_rate === null
            ? "Rule compliance of LLM agents"
            : `${totals.fallback_rate}% of moves fell back to a random legal move`
        }
      />
      <StatCard
        label="Average game length"
        value={totals.avg_rounds === null ? "-" : `${totals.avg_rounds}`}
        hint="Rounds per game (limit 25)"
      />
    </div>
  )
}
