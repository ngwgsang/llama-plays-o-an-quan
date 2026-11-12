import { CircleAlertIcon } from "lucide-react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Skeleton } from "@/components/ui/skeleton"
import type { ApiError } from "@/lib/api"

export function ApiErrorAlert({ error }: { error: ApiError }) {
  const offline = error.status === 0 || error.status >= 500

  return (
    <Alert variant="destructive">
      <CircleAlertIcon />
      <AlertTitle>{offline ? "Backend API is not reachable" : "Request failed"}</AlertTitle>
      <AlertDescription>
        {offline ? (
          <p>
            Start it from the repository root with{" "}
            <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs text-foreground">
              python -m o_an_quan.api
            </code>{" "}
            (listens on http://127.0.0.1:8000).
          </p>
        ) : (
          <p>{error.message}</p>
        )}
      </AlertDescription>
    </Alert>
  )
}

export function LoadingGrid({ rows = 2 }: { rows?: number }) {
  return (
    <div className="grid gap-4">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        {Array.from({ length: 4 }, (_, i) => (
          <Skeleton key={i} className="h-32 rounded-xl" />
        ))}
      </div>
      {Array.from({ length: rows }, (_, i) => (
        <Skeleton key={i} className="h-80 rounded-xl" />
      ))}
    </div>
  )
}
