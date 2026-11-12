"use client"

import { useState } from "react"
import Link from "next/link"
import { ArrowRightIcon, LoaderCircleIcon, PlayIcon, RotateCcwIcon } from "lucide-react"
import { toast } from "sonner"

import { ApiErrorAlert, LoadingGrid } from "@/components/api-state"
import { Replay } from "@/components/game/replay"
import { AgentConfigCard } from "@/components/play/agent-config-card"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Progress } from "@/components/ui/progress"
import { ApiError, apiFetch, useApi } from "@/lib/api"
import type { AgentConfig, AppConfig, Health, Job, NewGameRequest } from "@/lib/types"

const POLL_MS = 1000

function LiveGame({ jobId, onReset }: { jobId: string; onReset: () => void }) {
  const { data: job, error } = useApi<Job>(`/jobs/${jobId}`, {
    refreshInterval: POLL_MS,
    shouldPoll: (job) => {
      if (job && job.status !== "running") {
        if (job.status === "finished") toast.success("Game finished and saved")
        else toast.error("Game failed", { description: job.error ?? undefined })
        return false
      }
      return true
    },
  })

  if (error) return <ApiErrorAlert error={error} />
  if (!job) return <LoadingGrid rows={1} />

  const round = job.turns.at(-1)?.round ?? 0
  return (
    <div className="grid gap-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex flex-wrap items-center gap-2">
            {job.agent_a} vs {job.agent_b}
            {job.status === "running" && (
              <Badge variant="outline" className="gap-1">
                <LoaderCircleIcon className="animate-spin" /> running
              </Badge>
            )}
            {job.status === "finished" && <Badge variant="secondary">finished</Badge>}
            {job.status === "failed" && <Badge variant="destructive">failed</Badge>}
          </CardTitle>
          <CardDescription>
            Round {round} of at most {job.max_rounds} · {job.turns.length} moves
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3">
          <Progress
            value={job.status === "running" ? (round / job.max_rounds) * 100 : 100}
            aria-label="Round progress"
          />
          {job.error && (
            <Alert variant="destructive">
              <AlertTitle>The game stopped with an error</AlertTitle>
              <AlertDescription className="font-mono text-xs break-all">{job.error}</AlertDescription>
            </Alert>
          )}
        </CardContent>
        <CardFooter className="gap-2">
          <Button variant="outline" onClick={onReset}>
            <RotateCcwIcon /> New game
          </Button>
          {job.game_id && (
            <Button asChild>
              <Link href={`/games/${job.game_id}`}>
                Open replay <ArrowRightIcon />
              </Link>
            </Button>
          )}
        </CardFooter>
      </Card>
      <Replay turns={job.turns} initialBoard={job.initial_board} live={job.status === "running"} />
    </div>
  )
}

export function PlayView() {
  const config = useApi<AppConfig>("/config")
  const health = useApi<Health>("/health")
  const [agentA, setAgentA] = useState<AgentConfig>({ type: "random" })
  const [agentB, setAgentB] = useState<AgentConfig>({ type: "random" })
  const [maxRounds, setMaxRounds] = useState("")
  const [seed, setSeed] = useState("")
  const [submitting, setSubmitting] = useState(false)
  const [jobId, setJobId] = useState<string>()

  if (config.error) return <ApiErrorAlert error={config.error} />
  if (!config.data || !health.data) return <LoadingGrid rows={1} />
  if (jobId) return <LiveGame jobId={jobId} onReset={() => setJobId(undefined)} />

  const llmDisabled = !health.data.has_api_key
  const withDefaults = (agent: AgentConfig): AgentConfig =>
    agent.type === "llm" ? { ...agent, model: agent.model ?? config.data!.default_model } : { type: "random" }

  const start = async () => {
    const request: NewGameRequest = {
      agent_a: withDefaults(agentA),
      agent_b: withDefaults(agentB),
      max_rounds: Number(maxRounds) || config.data!.max_rounds,
      seed: seed === "" ? null : Number(seed),
    }
    setSubmitting(true)
    try {
      const job = await apiFetch<Job>("/games", { method: "POST", body: JSON.stringify(request) })
      setJobId(job.id)
    } catch (e) {
      toast.error("Could not start the game", { description: e instanceof ApiError ? e.message : String(e) })
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="grid gap-4">
      {llmDisabled && (
        <Alert>
          <AlertTitle>LLM agents are disabled</AlertTitle>
          <AlertDescription>
            Set <code className="font-mono">OPENROUTER_API_KEY</code> in the <code className="font-mono">.env</code> file
            at the repository root and restart the API to let LLM agents play.
          </AlertDescription>
        </Alert>
      )}
      <div className="grid gap-4 lg:grid-cols-2">
        <AgentConfigCard team="A" value={agentA} onChange={setAgentA} config={config.data} llmDisabled={llmDisabled} />
        <AgentConfigCard team="B" value={agentB} onChange={setAgentB} config={config.data} llmDisabled={llmDisabled} />
      </div>
      <Card>
        <CardHeader>
          <CardTitle>Game settings</CardTitle>
          <CardDescription>
            Rules E1-E5 from the paper always apply. LLM games call OpenRouter on every move and can take a few minutes.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 sm:grid-cols-2">
          <div className="grid gap-2">
            <Label htmlFor="max-rounds">Max rounds</Label>
            <Input
              id="max-rounds"
              type="number"
              min={1}
              max={100}
              placeholder={String(config.data.max_rounds)}
              value={maxRounds}
              onChange={(e) => setMaxRounds(e.target.value)}
            />
          </div>
          <div className="grid gap-2">
            <Label htmlFor="seed">Seed (optional)</Label>
            <Input
              id="seed"
              type="number"
              placeholder="random"
              value={seed}
              onChange={(e) => setSeed(e.target.value)}
            />
          </div>
        </CardContent>
        <CardFooter>
          <Button onClick={start} disabled={submitting}>
            {submitting ? <LoaderCircleIcon className="animate-spin" /> : <PlayIcon />} Start game
          </Button>
        </CardFooter>
      </Card>
    </div>
  )
}
