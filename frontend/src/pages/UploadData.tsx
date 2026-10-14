import { Download, FileUp } from 'lucide-react'
import { useState } from 'react'
import Panel from '../components/Panel'
import SessionPlayer from '../components/SessionPlayer'
import { api } from '../lib/api'
import { PUBLIC } from '../lib/deploy'
import { MAX_ROWS, MIN_ROWS, parseSensorCsv, type ParsedCsv } from '../lib/csv'
import { useGrid } from '../store/grid'

export default function UploadData() {
  const system = useGrid((s) => s.system)
  const [asset, setAsset] = useState('ETTm1')
  const [parsed, setParsed] = useState<ParsedCsv>()
  const [name, setName] = useState<string>()
  const assets = system?.dataset.assets ?? [{ id: 'ETTm1', label: 'T1' }, { id: 'ETTm2', label: 'T2' }]
  const maxRows = system?.engine?.max_run_rows ?? MAX_ROWS

  const load = async (file: File) => {
    setName(file.name)
    setParsed(parseSensorCsv(await file.text(), maxRows))
  }
  const ok = !!parsed && !parsed.error

  return (
    <div className="space-y-3">
      <div>
        <div className="label">Upload &amp; score · your data</div>
        <h1 className="text-[20px] font-semibold tracking-tight">Run your own transformer readings through the engine</h1>
        <p className="mt-0.5 max-w-3xl text-[12.5px] text-ink-2">Upload a CSV of 15-minute readings: <code className="font-mono">timestamp</code> (or <code className="font-mono">date</code>) plus the seven channels <code className="font-mono">HUFL, HULL, MUFL, MULL, LUFL, LULL, OT</code> (high/middle/low useful &amp; useless load, oil temperature), in the same units as ETDataset. The engine scores every observation after a ~40 h warm-up. Your file is processed in memory for this session only and is not stored.</p>
      </div>
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-12">
        <div className="space-y-3 xl:col-span-3">
          <Panel kicker="1 · input" title="Sensor CSV">
            <div className="space-y-3 text-[12.5px]">
              <label className="flex cursor-pointer flex-col items-center gap-1 rounded border border-dashed border-line-2 px-3 py-5 text-center hover:bg-surface-2">
                <FileUp size={18} className="text-accent" aria-hidden />
                <span>{name ?? 'Choose a CSV file'}</span>
                <span className="text-[11px] text-ink-3">{MIN_ROWS} – {maxRows.toLocaleString()} consecutive 15-min rows</span>
                <input type="file" accept=".csv,text/csv" className="sr-only" onChange={(e) => e.target.files?.[0] && load(e.target.files[0])} aria-label="Upload sensor CSV" />
              </label>
              {parsed && (parsed.error
                ? <div role="alert" className="rounded border border-crit/50 bg-crit/10 px-2 py-1.5 text-crit">{parsed.error}</div>
                : <div className="rounded border border-ok/40 bg-ok/5 px-2 py-1.5 text-ok">{parsed.rows.length} rows · {parsed.rows[0].timestamp} → {parsed.rows.at(-1)!.timestamp}{parsed.note ? ` · ${parsed.note}` : ''}</div>)}
              <fieldset>
                <legend className="label mb-1">Scale with the statistics of</legend>
                <div className="flex gap-1" role="radiogroup" aria-label="Transformer profile">
                  {assets.map((a) => <button key={a.id} role="radio" aria-checked={asset === a.id} onClick={() => setAsset(a.id)}
                    className={`flex-1 rounded border px-2 py-1 ${asset === a.id ? 'border-ink-3 bg-surface-2' : 'border-line text-ink-3'}`}>Transformer {a.label}</button>)}
                </div>
                <p className="mt-1 text-[11px] text-ink-3">The model normalises inputs per transformer; pick the one whose load level is closest to yours.</p>
              </fieldset>
              <div>
                <div className="label mb-1">No data at hand? Download a synthetic template</div>
                <div className="flex flex-wrap gap-1.5">
                  {[['thermal_runaway', 'thermal fault'], ['intermittent_arcing', 'arcing fault'], ['none', 'normal only']].map(([f, l]) => (
                    <a key={f} href={api.sampleUrl(asset, f, 4)} className="flex items-center gap-1 rounded border border-line-2 px-2 py-0.5 text-[11.5px]"><Download size={12} />{l}</a>))}
                </div>
                <p className="mt-1 text-[11px] text-ink-3">Synthetic streams from the Scenario Lab generator — edit them and upload to see how the model responds.</p>
              </div>
            </div>
          </Panel>
        </div>
        <div className="xl:col-span-9">
          <SessionPlayer kind="upload" ready={ok} title={name ? `Your data · ${name}` : 'Your data'}
            {...(PUBLIC ? { run: () => api.runUpload(asset, parsed!.rows) } : { create: () => api.upload(asset, parsed!.rows) })} />
        </div>
      </div>
    </div>
  )
}
