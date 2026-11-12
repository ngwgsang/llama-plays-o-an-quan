"use client"

import { CircleAlertIcon, CircleCheckIcon, KeyRoundIcon } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { useApi } from "@/lib/api"
import type { Health } from "@/lib/types"

export function BackendStatus() {
  const { data, error } = useApi<Health>("/health", { refreshInterval: 15000 })

  if (error) {
    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <Badge variant="destructive">
            <CircleAlertIcon /> API offline
          </Badge>
        </TooltipTrigger>
        <TooltipContent>Start it with: python -m o_an_quan.api</TooltipContent>
      </Tooltip>
    )
  }
  if (!data) return null

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Badge variant="outline">
          {data.has_api_key ? <CircleCheckIcon className="text-chart-3" /> : <KeyRoundIcon />}
          {data.has_api_key ? "API connected" : "No OpenRouter key"}
        </Badge>
      </TooltipTrigger>
      <TooltipContent>
        {data.has_api_key
          ? `Backend v${data.version} · results in ${data.results_dir}`
          : "Only random agents can play until OPENROUTER_API_KEY is set in .env"}
      </TooltipContent>
    </Tooltip>
  )
}
