import type { ReactNode } from 'react'

export default function Panel({ title, kicker, right, children, className = '', bodyClass = 'p-3', id }: {
  title?: ReactNode; kicker?: ReactNode; right?: ReactNode; children: ReactNode; className?: string; bodyClass?: string; id?: string
}) {
  return (
    <section className={`panel flex min-w-0 flex-col ${className}`} aria-labelledby={id}>
      {(title || right) && (
        <header className="flex items-center gap-3 border-b border-line px-3 py-2">
          <div className="min-w-0">
            {kicker && <div className="label">{kicker}</div>}
            {title && <h2 id={id} className="truncate text-[13px] font-semibold tracking-wide">{title}</h2>}
          </div>
          <div className="ml-auto flex items-center gap-2">{right}</div>
        </header>
      )}
      <div className={`min-h-0 flex-1 ${bodyClass}`}>{children}</div>
    </section>
  )
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="grid h-full min-h-24 place-items-center p-4 text-center text-[12px] text-ink-3">{children}</div>
}

export function Derived({ children = 'Derived indicator' }: { children?: ReactNode }) {
  return <span className="rounded border border-line-2 px-1 py-px text-[9.5px] font-mono uppercase tracking-wider text-ink-3" title="Computed by the application from model outputs and sensor features; not an industry standard">{children}</span>
}
