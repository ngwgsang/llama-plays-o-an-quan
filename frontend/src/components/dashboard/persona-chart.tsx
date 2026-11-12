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
import { formatNumber, formatPercent } from "@/lib/format"
import type { PersonaRow } from "@/lib/types"

const config = {
  EGE: { label: "Early (R1-10)", color: "var(--chart-1)" },
  MGE: { label: "Mid (R11-20)", color: "var(--chart-2)" },
  LGE: { label: "Late (R21-25)", color: "var(--chart-3)" },
} satisfies ChartConfig

export function PersonaChart({ rows }: { rows: PersonaRow[] }) {
  const data = rows.map((row) => ({ ...row, agent: row.agent.replace(/Agent$/, "") }))

  return (
    <ChartCard
      title="Score by persona and game phase"
      description="Table I · persona agent (team A) vs RandomAgent, score at the end of each phase"
      empty={
        !rows.length && (
          <NoData>
            Run <code className="font-mono">python -m o_an_quan.experiments personas</code> to fill this chart.
          </NoData>
        )
      }
      chart={
        <ChartContainer config={config} className="aspect-auto h-72 w-full">
          <BarChart data={data} barGap={2} margin={{ left: 4, right: 4, top: 8 }}>
            <CartesianGrid vertical={false} strokeOpacity={0.4} />
            <XAxis dataKey="agent" tickLine={false} axisLine={false} tickMargin={8} />
            <YAxis tickLine={false} axisLine={false} width={32} />
            <ChartTooltip cursor={{ fillOpacity: 0.3 }} content={<ChartTooltipContent />} />
            <ChartLegend itemSorter={null} content={<ChartLegendContent />} />
            {(["EGE", "MGE", "LGE"] as const).map((key) => (
              <Bar key={key} dataKey={key} fill={`var(--color-${key})`} radius={[4, 4, 0, 0]} maxBarSize={28} />
            ))}
          </BarChart>
        </ChartContainer>
      }
      table={
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Agent</TableHead>
              <TableHead className="text-right">Games</TableHead>
              <TableHead className="text-right">EGE</TableHead>
              <TableHead className="text-right">MGE</TableHead>
              <TableHead className="text-right">LGE</TableHead>
              <TableHead className="text-right">Avg point</TableHead>
              <TableHead className="text-right">Win</TableHead>
              <TableHead className="text-right">Draw</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row) => (
              <TableRow key={row.agent}>
                <TableCell className="font-medium">{row.agent}</TableCell>
                <TableCell className="text-right tabular-nums">{row.games}</TableCell>
                <TableCell className="text-right tabular-nums">{formatNumber(row.EGE)}</TableCell>
                <TableCell className="text-right tabular-nums">{formatNumber(row.MGE)}</TableCell>
                <TableCell className="text-right tabular-nums">{formatNumber(row.LGE)}</TableCell>
                <TableCell className="text-right font-medium tabular-nums">{formatNumber(row.avg_point)}</TableCell>
                <TableCell className="text-right tabular-nums">{formatPercent(row.win_rate)}</TableCell>
                <TableCell className="text-right tabular-nums">{formatPercent(row.draw_rate)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      }
    />
  )
}
