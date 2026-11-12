import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

/** Card with a chart view and an equivalent table view. */
export function ChartCard({
  title,
  description,
  chart,
  table,
  empty,
  className,
}: {
  title: string
  description?: React.ReactNode
  chart: React.ReactNode
  table: React.ReactNode
  empty?: React.ReactNode
  className?: string
}) {
  return (
    <Card className={className}>
      <Tabs defaultValue="chart" className="gap-0">
        <CardHeader>
          <CardTitle>{title}</CardTitle>
          {description && <CardDescription>{description}</CardDescription>}
          {!empty && (
            <CardAction>
              <TabsList>
                <TabsTrigger value="chart">Chart</TabsTrigger>
                <TabsTrigger value="table">Table</TabsTrigger>
              </TabsList>
            </CardAction>
          )}
        </CardHeader>
        <CardContent className="pt-4">
          {empty ? (
            empty
          ) : (
            <>
              <TabsContent value="chart">{chart}</TabsContent>
              <TabsContent value="table" className="overflow-x-auto">
                {table}
              </TabsContent>
            </>
          )}
        </CardContent>
      </Tabs>
    </Card>
  )
}

export function NoData({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex h-48 items-center justify-center rounded-md border border-dashed p-6 text-center text-sm text-muted-foreground">
      <div>{children}</div>
    </div>
  )
}
