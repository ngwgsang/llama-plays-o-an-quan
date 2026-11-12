"use client"

import { Bar, BarChart, CartesianGrid, XAxis, YAxis } from "recharts"

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
import { formatNumber, formatPercent, shortModel } from "@/lib/format"
import type { ModelRow } from "@/lib/types"

const config = {
  win_rate: { label: "Win %", color: "var(--chart-1)" },
  draw_rate: { label: "Draw %", color: "var(--chart-2)" },
} satisfies ChartConfig

export function ModelChart({ rows }: { rows: ModelRow[] }) {
  const data = rows.map((row) => ({
    ...row,
    label: `${shortModel(row.model)}${row.plays_first ? " *" : ""}`,
  }))
  const opponent = rows[0]?.opponent

  return (
    <ChartCard
      title="Win and draw rate by model"
      description={
        <>
          Table II · against {opponent ? shortModel(opponent) : "the opponent model"} · * plays first
        </>
      }
      empty={
        !rows.length && (
          <NoData>
            Run <code className="font-mono">python -m o_an_quan.experiments models</code> to fill this chart.
          </NoData>
        )
      }
      chart={
        <ChartContainer config={config} className="aspect-auto h-72 w-full">
          <BarChart data={data} barGap={2} margin={{ left: 4, right: 4, top: 8 }}>
            <CartesianGrid vertical={false} strokeOpacity={0.4} />
            <XAxis dataKey="label" tickLine={false} axisLine={false} tickMargin={8} />
            <YAxis tickLine={false} axisLine={false} width={32} domain={[0, 100]} unit="%" />
            <ChartTooltip cursor={{ fillOpacity: 0.3 }} content={<ChartTooltipContent />} />
            <ChartLegend itemSorter={null} content={<ChartLegendContent />} />
            <Bar dataKey="win_rate" fill="var(--color-win_rate)" radius={[4, 4, 0, 0]} maxBarSize={36} />
            <Bar dataKey="draw_rate" fill="var(--color-draw_rate)" radius={[4, 4, 0, 0]} maxBarSize={36} />
          </BarChart>
        </ChartContainer>
      }
      table={
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Model</TableHead>
              <TableHead>Team</TableHead>
              <TableHead className="text-right">Games</TableHead>
              <TableHead className="text-right">Win</TableHead>
              <TableHead className="text-right">Draw</TableHead>
              <TableHead className="text-right">Illegal / move</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row) => (
              <TableRow key={`${row.model}-${row.team}`}>
                <TableCell className="font-medium">{row.model}</TableCell>
                <TableCell>{row.team}</TableCell>
                <TableCell className="text-right tabular-nums">{row.games}</TableCell>
                <TableCell className="text-right tabular-nums">{formatPercent(row.win_rate)}</TableCell>
                <TableCell className="text-right tabular-nums">{formatPercent(row.draw_rate)}</TableCell>
                <TableCell className="text-right tabular-nums">{formatNumber(row.invalid_per_move, 2)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      }
    />
  )
}
