"use client"

import { Suspense } from "react"
import { usePathname } from "next/navigation"

import { isActive, NAV_ITEMS } from "@/components/app-sidebar"
import { BackendStatus } from "@/components/backend-status"
import { ModeToggle } from "@/components/mode-toggle"
import { Separator } from "@/components/ui/separator"
import { SidebarTrigger } from "@/components/ui/sidebar"

function PageTitle() {
  const pathname = usePathname()
  return [...NAV_ITEMS].reverse().find((item) => isActive(pathname, item.href))?.title
}

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-10 flex h-14 shrink-0 items-center gap-2 border-b bg-background/95 px-4 backdrop-blur">
      <SidebarTrigger className="-ml-1" />
      <Separator orientation="vertical" className="mr-2 data-[orientation=vertical]:h-4" />
      <h1 className="text-base font-medium">
        <Suspense>
          <PageTitle />
        </Suspense>
      </h1>
      <div className="ml-auto flex items-center gap-2">
        <BackendStatus />
        <ModeToggle />
      </div>
    </header>
  )
}
