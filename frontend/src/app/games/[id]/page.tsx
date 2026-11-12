import { Suspense } from "react"

import { GameView } from "@/components/game/game-view"
import { Skeleton } from "@/components/ui/skeleton"

async function Game({ params }: { params: PageProps<"/games/[id]">["params"] }) {
  const { id } = await params
  return <GameView id={id} />
}

export default function GamePage({ params }: PageProps<"/games/[id]">) {
  // params is request-time data under cacheComponents, so it needs a Suspense boundary
  return (
    <Suspense fallback={<Skeleton className="h-[32rem] rounded-xl" />}>
      <Game params={params} />
    </Suspense>
  )
}
