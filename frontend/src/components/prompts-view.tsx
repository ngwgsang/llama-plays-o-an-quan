"use client"

import { ApiErrorAlert } from "@/components/api-state"
import { Markdown } from "@/components/markdown"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useApi } from "@/lib/api"
import type { Persona } from "@/lib/types"

export function PromptsView() {
  const personas = useApi<Persona[]>("/personas")
  const rules = useApi<{ content: string }>("/rules")

  const error = personas.error ?? rules.error
  if (error) return <ApiErrorAlert error={error} />

  return (
    <Tabs defaultValue="personas" className="gap-4">
      <TabsList>
        <TabsTrigger value="personas">Personas (P)</TabsTrigger>
        <TabsTrigger value="rules">Rules (R)</TabsTrigger>
      </TabsList>
      <TabsContent value="personas">
        <p className="mb-4 text-sm text-muted-foreground">
          Loaded from <code className="font-mono">backend/src/o_an_quan/prompts/personas/*.md</code>. Add a markdown file
          there to create a new persona.
        </p>
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {personas.data
            ? personas.data.map((persona) => (
                <Card key={persona.name}>
                  <CardContent>
                    <Markdown source={persona.content} />
                  </CardContent>
                </Card>
              ))
            : Array.from({ length: 3 }, (_, i) => <Skeleton key={i} className="h-64 rounded-xl" />)}
        </div>
      </TabsContent>
      <TabsContent value="rules">
        <Card>
          <CardHeader>
            <CardTitle>Rule specification given to the agents</CardTitle>
            <CardDescription>
              Basic rules plus extra rules E1-E5 (paper, Section III-A) ·{" "}
              <code className="font-mono">prompts/templates/rules.md</code>
            </CardDescription>
          </CardHeader>
          <CardContent>
            {rules.data ? (
              <pre className="overflow-x-auto rounded-md bg-muted p-4 font-mono text-xs leading-relaxed whitespace-pre-wrap">
                {rules.data.content}
              </pre>
            ) : (
              <Skeleton className="h-96" />
            )}
          </CardContent>
        </Card>
      </TabsContent>
    </Tabs>
  )
}
