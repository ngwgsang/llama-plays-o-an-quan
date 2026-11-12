"use client"

import { CartesianGrid, Line, LineChart, ReferenceLine, XAxis, YAxis } from "recharts"

import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart"
import type { Turn } from "@/lib/types"

const config = {
  A: { label: "Team A", color: "var(--chart-1)" },
  B: { label: "Team B", color: "var(--chart-2)" },
} satisfies ChartConfig

export function ScoreChart({ turns, step }: { turns: Turn[]; step: number }) {
  const data = [{ move: 0, A: 0, B: 0 }, ...turns.map((t, i) => ({ move: i + 1, ...t.score }))]

  return (
    <ChartContainer config={config} className="aspect-auto h-56 w-full">
      <LineChart data={data} margin={{ left: 4, right: 12, top: 8 }}>
        <CartesianGrid vertical={false} strokeOpacity={0.4} />
        <XAxis dataKey="move" tickLine={false} axisLine={false} tickMargin={8} minTickGap={24} />
        <YAxis tickLine={false} axisLine={false} width={28} allowDecimals={false} />
        <ChartTooltip
          content={<ChartTooltipContent labelFormatter={(_, payload) => `Move ${payload?.[0]?.payload.move}`} />}
        />
        <ReferenceLine x={step} stroke="var(--muted-foreground)" strokeDasharray="4 4" />
        <Line dataKey="A" type="stepAfter" stroke="var(--color-A)" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
        <Line dataKey="B" type="stepAfter" stroke="var(--color-B)" strokeWidth={2} dot={false} activeDot={{ r: 4 }} />
        <ChartLegend itemSorter={null} content={<ChartLegendContent />} />
      </LineChart>
    </ChartContainer>
  )
}
