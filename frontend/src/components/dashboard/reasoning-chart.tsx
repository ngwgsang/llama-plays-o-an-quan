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
import { formatPercent, shortModel } from "@/lib/format"
import type { ReasonRow } from "@/lib/types"

const config = {
  STG: { label: "Short-term gain", color: "var(--chart-1)" },
  LTS: { label: "Long-term strategy", color: "var(--chart-2)" },
  AMB: { label: "Ambiguous", color: "var(--chart-3)" },
} satisfies ChartConfig

const KEYS = ["STG", "LTS", "AMB"] as const

export function ReasoningChart({ rows }: { rows: ReasonRow[] }) {
  const data = rows.map((row) => ({ ...row, label: shortModel(row.model) }))

  return (
    <ChartCard
      title="Reasoning types"
      description="Table III · share of moves per reasoning type, classified zero-shot"
      empty={
        !rows.length && (
          <NoData>
            Run <code className="font-mono">python -m o_an_quan.experiments classify --input results/models.jsonl</code>{" "}
            to fill this chart.
          </NoData>
        )
      }
      chart={
        <ChartContainer config={config} className="aspect-auto h-72 w-full">
          <BarChart data={data} layout="vertical" margin={{ left: 4, right: 12, top: 8 }}>
            <CartesianGrid horizontal={false} strokeOpacity={0.4} />
            <XAxis type="number" domain={[0, 100]} unit="%" tickLine={false} axisLine={false} />
            <YAxis type="category" dataKey="label" tickLine={false} axisLine={false} width={150} />
            <ChartTooltip cursor={{ fillOpacity: 0.3 }} content={<ChartTooltipContent />} />
            <ChartLegend itemSorter={null} content={<ChartLegendContent />} />
            {KEYS.map((key, i) => (
              <Bar
                key={key}
                dataKey={key}
                stackId="share"
                fill={`var(--color-${key})`}
                stroke="var(--card)"
                strokeWidth={2}
                radius={i === KEYS.length - 1 ? [0, 4, 4, 0] : 0}
                maxBarSize={28}
              />
            ))}
          </BarChart>
        </ChartContainer>
      }
      table={
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Model</TableHead>
              <TableHead className="text-right">Moves</TableHead>
              <TableHead className="text-right">STG</TableHead>
              <TableHead className="text-right">LTS</TableHead>
              <TableHead className="text-right">AMB</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row) => (
              <TableRow key={row.model}>
                <TableCell className="font-medium">{row.model}</TableCell>
                <TableCell className="text-right tabular-nums">{row.moves}</TableCell>
                {KEYS.map((key) => (
                  <TableCell key={key} className="text-right tabular-nums">
                    {formatPercent(row[key])}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
      }
    />
  )
}
