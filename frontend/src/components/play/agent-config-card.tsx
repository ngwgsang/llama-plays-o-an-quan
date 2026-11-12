"use client"

import { BotIcon, DicesIcon } from "lucide-react"

import { TeamDot } from "@/components/game/team-badge"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import type { AgentConfig, AppConfig, Team } from "@/lib/types"

const NO_PERSONA = "none"

export function AgentConfigCard({
  team,
  value,
  onChange,
  config,
  llmDisabled,
}: {
  team: Team
  value: AgentConfig
  onChange: (value: AgentConfig) => void
  config: AppConfig
  llmDisabled: boolean
}) {
  const id = `agent-${team}`

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <TeamDot team={team} className="size-2.5" /> Team {team}
        </CardTitle>
        <CardDescription>{team === "A" ? "Moves first, owns A1-A5" : "Moves second, owns B1-B5"}</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-4">
        <Tabs value={value.type} onValueChange={(type) => onChange({ ...value, type: type as AgentConfig["type"] })}>
          <TabsList className="w-full">
            <TabsTrigger value="random">
              <DicesIcon /> Random
            </TabsTrigger>
            <TabsTrigger value="llm" disabled={llmDisabled}>
              <BotIcon /> LLM
            </TabsTrigger>
          </TabsList>
        </Tabs>
        {value.type === "llm" ? (
          <>
            <div className="grid gap-2">
              <Label htmlFor={`${id}-model`}>Model</Label>
              <Select value={value.model ?? config.default_model} onValueChange={(model) => onChange({ ...value, model })}>
                <SelectTrigger id={`${id}-model`} className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {config.models.map((model) => (
                    <SelectItem key={model} value={model}>
                      {model}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="grid gap-2">
              <Label htmlFor={`${id}-persona`}>Persona</Label>
              <Select
                value={value.persona ?? NO_PERSONA}
                onValueChange={(persona) => onChange({ ...value, persona: persona === NO_PERSONA ? null : persona })}
              >
                <SelectTrigger id={`${id}-persona`} className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value={NO_PERSONA}>No persona</SelectItem>
                  {config.personas.map((persona) => (
                    <SelectItem key={persona} value={persona}>
                      {persona}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </>
        ) : (
          <p className="text-sm text-muted-foreground">
            Picks a legal position and direction uniformly at random, like the paper&apos;s RandomAgent baseline.
          </p>
        )}
      </CardContent>
    </Card>
  )
}
