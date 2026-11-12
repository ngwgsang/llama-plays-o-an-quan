"use client"

import { Suspense } from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { FileTextIcon, GamepadIcon, LayoutDashboardIcon, ListIcon, PlayIcon } from "lucide-react"

import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
} from "@/components/ui/sidebar"

export const NAV_ITEMS = [
  { title: "Overview", href: "/", icon: LayoutDashboardIcon },
  { title: "Games", href: "/games", icon: ListIcon },
  { title: "New game", href: "/play", icon: PlayIcon },
  { title: "Prompts", href: "/prompts", icon: FileTextIcon },
]

export function isActive(pathname: string, href: string) {
  return href === "/" ? pathname === "/" : pathname.startsWith(href)
}

function NavLinks({ pathname }: { pathname?: string }) {
  return (
    <SidebarMenu>
      {NAV_ITEMS.map((item) => (
        <SidebarMenuItem key={item.href}>
          <SidebarMenuButton
            asChild
            isActive={pathname !== undefined && isActive(pathname, item.href)}
            tooltip={item.title}
          >
            <Link href={item.href}>
              <item.icon />
              <span>{item.title}</span>
            </Link>
          </SidebarMenuButton>
        </SidebarMenuItem>
      ))}
    </SidebarMenu>
  )
}

function ActiveNavLinks() {
  return <NavLinks pathname={usePathname()} />
}

export function AppSidebar() {
  return (
    <Sidebar collapsible="icon">
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" asChild>
              <Link href="/">
                <div className="flex aspect-square size-8 items-center justify-center rounded-lg bg-sidebar-primary text-sidebar-primary-foreground">
                  <GamepadIcon className="size-4" />
                </div>
                <div className="grid flex-1 text-left text-sm leading-tight">
                  <span className="truncate font-semibold">Ô Ăn Quan</span>
                  <span className="truncate text-xs text-muted-foreground">LLM planning study</span>
                </div>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Dashboard</SidebarGroupLabel>
          <SidebarGroupContent>
            {/* usePathname suspends on dynamic routes (cacheComponents); render links without highlight meanwhile */}
            <Suspense fallback={<NavLinks />}>
              <ActiveNavLinks />
            </Suspense>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton asChild tooltip="Paper (arXiv)">
              <a href="https://arxiv.org/abs/2507.03711" target="_blank" rel="noreferrer">
                <FileTextIcon />
                <span>Paper · arXiv 2507.03711</span>
              </a>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  )
}
