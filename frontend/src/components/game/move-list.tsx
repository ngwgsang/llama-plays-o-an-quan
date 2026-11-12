import { TeamDot } from "@/components/game/team-badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { cn } from "@/lib/utils"
import type { Turn } from "@/lib/types"

export function MoveList({
  turns,
  step,
  onSelect,
}: {
  turns: Turn[]
  step: number
  onSelect: (step: number) => void
}) {
  return (
    <ScrollArea className="h-80 rounded-md border">
      <ol className="divide-y text-sm">
        <li>
          <button
            type="button"
            onClick={() => onSelect(0)}
            className={cn("w-full px-3 py-2 text-left hover:bg-muted/60", step === 0 && "bg-muted")}
          >
            <span className="text-muted-foreground">Start</span>
          </button>
        </li>
        {turns.map((turn, i) => (
          <li key={i}>
            <button
              type="button"
              onClick={() => onSelect(i + 1)}
              className={cn(
                "grid w-full grid-cols-[2.25rem_1fr_auto] items-center gap-2 px-3 py-2 text-left hover:bg-muted/60",
                step === i + 1 && "bg-muted",
              )}
            >
              <span className="text-xs text-muted-foreground tabular-nums">{i + 1}.</span>
              <span className="flex items-center gap-2 truncate">
                <TeamDot team={turn.team} />
                <span className="font-medium">{turn.pos}</span>
                <span className="truncate text-muted-foreground">
                  {turn.way === "clockwise" ? "↻" : "↺"} R{turn.round}
                </span>
              </span>
              <span className={cn("text-xs tabular-nums", turn.captured ? "font-medium" : "text-muted-foreground")}>
                +{turn.captured}
              </span>
            </button>
          </li>
        ))}
      </ol>
    </ScrollArea>
  )
}
