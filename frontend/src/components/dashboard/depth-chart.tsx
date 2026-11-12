"use client"

import { CartesianGrid, Line, LineChart, XAxis, YAxis } from "recharts"

import { ChartCard, NoData } from "@/components/dashboard/chart-card"
import {
  ChartContainer,
  ChartLegend,
  ChartLegendContent,
  ChartTooltip,
  ChartTooltipContent,
  type ChartConfig,
} from "@/components/ui/chart"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { shortModel } from "@/lib/format"
import type { DepthRow } from "@/lib/types"

// Fixed categorical order; more models than slots are folded out of the chart
const SLOTS = 5

export function DepthChart({ rows }: { rows: DepthRow[] }) {
  // Colour follows the model, not its rank: sort by name so slots are stable
  const models = [...rows].sort((a, b) => a.model.localeCompare(b.model)).slice(0, SLOTS)
  const keys = models.map((_, i) => `m${i}`)
  const config = Object.fromEntries(
    models.map((row, i) => [keys[i], { label: shortModel(row.model), color: `var(--chart-${i + 1})` }]),
  ) satisfies ChartConfig

  const rounds = [...new Set(models.flatMap((row) => row.per_round.map((r) => r.round)))].sort((a, b) => a - b)
  const data = rounds.map((round) => ({
    round,
    ...Object.fromEntries(
      models.map((row, i) => [keys[i], row.per_round.find((r) => r.round === round)?.mean ?? null]),
    ),
  }))
  const phases = Object.keys(rows[0]?.phases ?? {})

  return (
    <ChartCard
      title="Planning depth"
      description="Mean tokens generated per move, by round (Section V-C, Figure 3)"
      empty={!rows.length && <NoData>No LLM moves recorded yet.</NoData>}
      chart={
        <ChartContainer config={config} className="aspect-auto h-72 w-full">
          <LineChart data={data} margin={{ left: 4, right: 12, top: 8 }}>
            <CartesianGrid vertical={false} strokeOpacity={0.4} />
            <XAxis dataKey="round" tickLine={false} axisLine={false} tickMargin={8} />
            <YAxis tickLine={false} axisLine={false} width={40} />
            <ChartTooltip content={<ChartTooltipContent labelFormatter={(value) => `Round ${value}`} />} />
            <ChartLegend itemSorter={null} content={<ChartLegendContent />} />
            {keys.map((key) => (
              <Line
                key={key}
                dataKey={key}
                stroke={`var(--color-${key})`}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4 }}
                connectNulls
              />
            ))}
          </LineChart>
        </ChartContainer>
      }
      table={
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Model</TableHead>
              <TableHead className="text-right">Moves</TableHead>
              {phases.map((phase) => (
                <TableHead key={phase} className="text-right">
                  {phase}
                </TableHead>
              ))}
              <TableHead className="text-right">Overall</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row) => (
              <TableRow key={row.model}>
                <TableCell className="font-medium">{row.model}</TableCell>
                <TableCell className="text-right tabular-nums">{row.moves}</TableCell>
                {phases.map((phase) => {
                  const stat = row.phases[phase]
                  return (
                    <TableCell key={phase} className="text-right tabular-nums">
                      {stat ? `${stat.mean.toFixed(0)} / ${stat.max}` : "-"}
                    </TableCell>
                  )
                })}
                <TableCell className="text-right tabular-nums">
                  {row.overall.mean.toFixed(0)} / {row.overall.max}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      }
    />
  )
}
