import { cn } from "@/lib/utils"
import type { Board, Cell, Team } from "@/lib/types"

/**
 * Board layout. Team A on top (A1 -> A5), QB on the right, team B at the
 * bottom (B5 -> B1 going left), QA on the left, so the backend's "clockwise"
 * order reads clockwise on screen.
 */
const TOP_ROW = ["A1", "A2", "A3", "A4", "A5"]
const BOTTOM_ROW = ["B1", "B2", "B3", "B4", "B5"]
const MAX_DOTS = 15

type Highlight = {
  start?: string
  team?: Team
  captured?: string[]
}

function Peasants({ count }: { count: number }) {
  return (
    <div className="flex max-w-[5.5rem] flex-wrap justify-center gap-0.5" aria-hidden>
      {Array.from({ length: Math.min(count, MAX_DOTS) }, (_, i) => (
        <span key={i} className="size-1.5 rounded-full bg-foreground/55 sm:size-2" />
      ))}
    </div>
  )
}

function BoardCell({
  pos,
  cell,
  highlight,
  className,
}: {
  pos: string
  cell: Cell
  highlight?: Highlight
  className?: string
}) {
  const isMandarinCell = pos.startsWith("Q")
  const owner: Team = pos === "QA" || pos.startsWith("A") ? "A" : "B"
  const isStart = highlight?.start === pos
  const isCaptured = highlight?.captured?.includes(pos)
  const total = cell.peasants + (cell.mandarin ? 10 : 0)

  return (
    <div
      role="gridcell"
      aria-label={`${pos}: ${cell.peasants} peasants${cell.mandarin ? " and a mandarin" : ""}`}
      className={cn(
        "relative flex flex-col items-center justify-center gap-1.5 border bg-card p-1.5 transition-colors sm:p-2",
        isMandarinCell ? "row-span-2" : "aspect-square",
        pos === "QA" && "rounded-l-full pl-3",
        pos === "QB" && "rounded-r-full pr-3",
        isStart && (highlight?.team === "B" ? "ring-2 ring-team-b ring-inset" : "ring-2 ring-team-a ring-inset"),
        isCaptured && "bg-muted/70",
        className,
      )}
    >
      <span
        className={cn(
          "absolute top-1 text-[10px] font-medium text-muted-foreground sm:text-xs",
          isMandarinCell ? "left-1/2 -translate-x-1/2" : "left-1.5",
        )}
      >
        {pos}
      </span>
      {cell.mandarin && (
        <span
          className={cn(
            "flex size-7 items-center justify-center rounded-full text-xs font-semibold text-white sm:size-9",
            owner === "A" ? "bg-team-a" : "bg-team-b",
          )}
          title="Mandarin (10 points)"
        >
          Q
        </span>
      )}
      <Peasants count={cell.peasants} />
      <span className="text-sm font-semibold tabular-nums sm:text-base">
        {isMandarinCell ? total : cell.peasants}
      </span>
      {isCaptured && (
        <span className="absolute right-1 bottom-1 rounded bg-foreground px-1 text-[10px] font-medium text-background">
          captured
        </span>
      )}
    </div>
  )
}

export function GameBoard({
  board,
  highlight,
  className,
}: {
  board: Board
  highlight?: Highlight
  className?: string
}) {
  return (
    <div className={cn("w-full", className)}>
      <div className="mb-1.5 flex items-center gap-1.5 text-xs text-muted-foreground">
        <span className="size-2 rounded-full bg-team-a" /> Team A side (top)
      </div>
      <div
        role="grid"
        aria-label="Ô Ăn Quan board"
        className="grid grid-cols-[1.3fr_repeat(5,minmax(0,1fr))_1.3fr] grid-rows-2 overflow-hidden"
      >
        <BoardCell pos="QA" cell={board.QA} highlight={highlight} />
        {TOP_ROW.map((pos) => (
          <BoardCell key={pos} pos={pos} cell={board[pos]} highlight={highlight} />
        ))}
        <BoardCell pos="QB" cell={board.QB} highlight={highlight} className="col-start-7 row-start-1" />
        {BOTTOM_ROW.map((pos) => (
          <BoardCell key={pos} pos={pos} cell={board[pos]} highlight={highlight} />
        ))}
      </div>
      <div className="mt-1.5 flex items-center justify-between gap-1.5 text-xs text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span className="size-2 rounded-full bg-team-b" /> Team B side (bottom)
        </span>
        <span>Clockwise: A1 → A5 → QB → B5 → B1 → QA</span>
      </div>
    </div>
  )
}
