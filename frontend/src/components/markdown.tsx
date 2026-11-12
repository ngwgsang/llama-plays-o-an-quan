import { Fragment } from "react"

/**
 * Minimal renderer for the prompt files (headings, lists, **bold**, `code`).
 * The prompts are plain markdown written in this repo, so a full parser is not needed.
 */
function inline(text: string) {
  return text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).map((part, i) => {
    if (part.startsWith("**") && part.endsWith("**")) return <strong key={i}>{part.slice(2, -2)}</strong>
    if (part.startsWith("`") && part.endsWith("`"))
      return (
        <code key={i} className="rounded bg-muted px-1 py-0.5 font-mono text-xs">
          {part.slice(1, -1)}
        </code>
      )
    return <Fragment key={i}>{part}</Fragment>
  })
}

type Block = { kind: "h1" | "h2" | "p"; text: string } | { kind: "ul"; items: string[] }

function parse(source: string): Block[] {
  const blocks: Block[] = []
  for (const raw of source.split("\n")) {
    const line = raw.trimEnd()
    const last = blocks.at(-1)
    if (!line.trim()) {
      blocks.push({ kind: "p", text: "" })
    } else if (line.startsWith("## ")) {
      blocks.push({ kind: "h2", text: line.slice(3) })
    } else if (line.startsWith("# ")) {
      blocks.push({ kind: "h1", text: line.slice(2) })
    } else if (/^\s*- /.test(line)) {
      const item = line.replace(/^\s*- /, "")
      if (last?.kind === "ul") last.items.push(item)
      else blocks.push({ kind: "ul", items: [item] })
    } else if (/^\s+/.test(line) && last?.kind === "ul") {
      last.items[last.items.length - 1] += ` ${line.trim()}`
    } else if (last?.kind === "p" && last.text) {
      last.text += `\n${line}`
    } else {
      blocks.push({ kind: "p", text: line })
    }
  }
  return blocks.filter((b) => b.kind !== "p" || b.text)
}

export function Markdown({ source }: { source: string }) {
  return (
    <div className="grid gap-3 text-sm leading-relaxed">
      {parse(source).map((block, i) => {
        switch (block.kind) {
          case "h1":
            return (
              <h2 key={i} className="text-base font-semibold">
                {inline(block.text)}
              </h2>
            )
          case "h2":
            return (
              <h3 key={i} className="text-xs font-medium tracking-wide text-muted-foreground uppercase">
                {inline(block.text)}
              </h3>
            )
          case "ul":
            return (
              <ul key={i} className="ml-4 grid list-disc gap-1 marker:text-muted-foreground">
                {block.items.map((item, j) => (
                  <li key={j}>{inline(item)}</li>
                ))}
              </ul>
            )
          default:
            return (
              <p key={i} className="whitespace-pre-wrap">
                {inline(block.text)}
              </p>
            )
        }
      })}
    </div>
  )
}
