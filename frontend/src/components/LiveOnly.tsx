import { ExternalLink, Terminal } from 'lucide-react'
import { Link } from 'react-router-dom'
import { REPO_URL } from '../lib/staticMode'

/** Shown on the static results site for views that need the live Python engine (and the ETDataset sensor rows,
 *  which are not redistributed). Displays a real screenshot of the view running locally. */
export default function LiveOnly({ title, what, shot }: { title: string; what: string; shot: string }) {
  return (
    <div className="mx-auto max-w-6xl space-y-4">
      <div>
        <div className="label">Live view · runs locally</div>
        <h1 className="text-[22px] font-semibold tracking-tight">{title}</h1>
        <p className="mt-1 max-w-3xl text-[13px] text-ink-2">{what}</p>
      </div>
      <div className="panel space-y-3 p-4 text-[13px]">
        <p className="text-ink-2">
          This hosted site is a <b className="text-ink">static results snapshot</b>. The live views stream held-out sensor data through the
          PyTorch/ONNX engine, which can't run on static hosting. The site also doesn't republish the sensor readings: ETDataset is
          licensed CC BY-ND 4.0, and the simulated faults are injected into it. Run it locally to use this view; the first run
          downloads the data and trains the model (~30 min).
        </p>
        <pre className="overflow-x-auto rounded border border-line bg-bg p-3 font-mono text-[12px] text-ink"><Terminal size={13} className="mb-1 inline text-accent" aria-hidden /> git clone {REPO_URL}.git{'\n'}make setup &amp;&amp; make data &amp;&amp; make demo   # → http://127.0.0.1:8000</pre>
        <div className="flex flex-wrap gap-2">
          <a href={REPO_URL} className="flex items-center gap-1.5 rounded border border-accent/70 bg-accent/10 px-3 py-1.5 font-semibold text-accent">GitHub repository <ExternalLink size={13} /></a>
          <Link to="/failures" className="rounded border border-line-2 px-3 py-1.5">Failure Intelligence</Link>
          <Link to="/models" className="rounded border border-line-2 px-3 py-1.5">Model Lab</Link>
          <Link to="/methodology" className="rounded border border-line-2 px-3 py-1.5">Model &amp; Methodology</Link>
        </div>
      </div>
      <figure className="panel overflow-hidden">
        <img src={shot} alt={`Screenshot of the ${title} running locally on held-out data`} className="w-full" loading="lazy" />
        <figcaption className="border-t border-line px-3 py-2 text-[11.5px] text-ink-3">Screenshot of this view running locally (real engine, held-out test period, simulated failure events).</figcaption>
      </figure>
    </div>
  )
}
