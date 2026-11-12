import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"
import type { Winner } from "@/lib/types"

export function TeamDot({ team, className }: { team: "A" | "B"; className?: string }) {
  return (
    <span
      aria-hidden
      className={cn("inline-block size-2 shrink-0 rounded-full", team === "A" ? "bg-team-a" : "bg-team-b", className)}
    />
  )
}

export function TeamBadge({ team, label }: { team: Winner; label?: string }) {
  if (team === "DRAW") return <Badge variant="secondary">{label ?? "Draw"}</Badge>
  return (
    <Badge variant="outline" className="gap-1.5">
      <TeamDot team={team} />
      {label ?? `Team ${team}`}
    </Badge>
  )
}
