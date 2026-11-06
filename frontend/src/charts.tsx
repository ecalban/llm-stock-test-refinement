import { useId, useState, type PointerEvent } from 'react'
import { dateLabel, number, percent, type BacktestData } from './api'

type Series = { name: string; values: (number | null | undefined)[]; color: string; bars?: boolean }

export function SeriesChart({ label, dates, series, bounds, compact = false, guides = [], format = number, tickLabel }:
  { label: string; dates: string[]; series: Series[]; bounds?: [number, number]; compact?: boolean;
    guides?: number[]; format?: (value: number) => string; tickLabel?: (index: number) => string }) {
  const [hover, setHover] = useState<number | null>(null)
  const uid = useId()
  const values = series.flatMap(s => s.values.filter((v): v is number => v != null && Number.isFinite(v)))
  if (!values.length || !dates.length) return <p className="empty-state">Grafik verisi bulunamadı.</p>
  const low = Math.min(...values, ...guides)
  const high = Math.max(...values, ...guides)
  const pad = (high - low || Math.max(Math.abs(high), 1)) * .08
  const [min, max] = bounds ?? [low - pad, high + pad]
  const width = 660, height = compact ? 115 : 210
  const x = (i: number) => dates.length === 1 ? width / 2 : i * width / (dates.length - 1)
  const y = (v: number) => height - (v - min) * height / (max - min)
  const selected = hover == null ? null : Math.min(hover, dates.length - 1)
  const tickIds = [...new Set([0, .25, .5, .75, 1].map(f => Math.round(f * (dates.length - 1))))]
  const onMove = (event: PointerEvent<SVGSVGElement>) => {
    const { left, width: w } = event.currentTarget.getBoundingClientRect()
    setHover(Math.min(dates.length - 1, Math.max(0, Math.round((event.clientX - left) / w * (dates.length - 1)))))
  }
  return <div className={`chart-frame real-chart ${compact ? 'compact' : ''}`}>
    <div className="y-labels">{[0, .33, .66, 1].map(f => <span key={f}>{format(max - f * (max - min))}</span>)}</div>
    <svg className="area-chart" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" role="img"
      aria-label={label} onPointerMove={onMove} onPointerLeave={() => setHover(null)}>
      <defs><clipPath id={uid}><rect width={width} height={height}/></clipPath>
        <linearGradient id={uid + 'fill'} x2="0" y2="1"><stop stopColor={series[0]?.color} stopOpacity=".22"/>
          <stop offset="100%" stopColor={series[0]?.color} stopOpacity="0"/></linearGradient></defs>
      {[0, .33, .66, 1].map(f => <line key={f} x1="0" x2={width} y1={height * f} y2={height * f} className="grid-line"/>)}
      <g clipPath={`url(#${uid})`}>
        {guides.map(value => <line key={value} x1="0" x2={width} y1={y(value)} y2={y(value)} className="dash-line"/>)}
        {series.map((s, si) => {
          if (s.bars) return <g key={s.name}>{s.values.map((value, i) => value == null ? null :
            <rect key={i} x={x(i) - 2} width={Math.max(1, Math.min(6, width / dates.length * .7))}
              y={Math.min(y(value), y(0))} height={Math.abs(y(value) - y(0))}
              fill={value >= 0 ? 'var(--green)' : 'var(--red)'} opacity=".35"/>)}</g>
          let previous = false
          const path = s.values.map((value, i) => {
            if (value == null || !Number.isFinite(value)) { previous = false; return '' }
            const command = previous ? 'L' : 'M'
            previous = true
            return `${command}${x(i)} ${y(value)}`
          }).join(' ')
          const fill = !compact && si === 0 && !s.values.some(v => v == null) && dates.length > 1
          return <g key={s.name}>
            {fill && <path d={`${path} L${width} ${height} L0 ${height} Z`} fill={`url(#${uid}fill)`}/>}
            <path d={path} fill="none" stroke={s.color} strokeWidth={si === 0 ? 2.5 : 1.6}
              strokeDasharray={si > 0 && !compact ? '5 4' : undefined} vectorEffect="non-scaling-stroke"/>
          </g>
        })}
        {selected != null && <line x1={x(selected)} x2={x(selected)} y1="0" y2={height} className="dash-line"/>}
      </g>
    </svg>
    <div className="x-labels">{tickIds.map(i => <span key={i}>{tickLabel ? tickLabel(i) : dateLabel(dates[i].slice(0, 10))}</span>)}</div>
    {selected != null && <div className="chart-tooltip"><b>{tickLabel ? tickLabel(selected) : dateLabel(dates[selected].slice(0, 10))}</b>
      {series.map(s => <span key={s.name}>{s.name}: {s.values[selected] == null ? '—' : format(s.values[selected]!)}</span>)}
    </div>}
  </div>
}

export function EquityChart({ result }: { result: BacktestData }) {
  // Use valuation dates, including the same-day terminal liquidation.
  const points = [...new Map(result.daily.map(row => [row.valuation_date, row])).values()]
  const change = (v: number) => (v / result.config.initial_cash - 1) * 100
  const series: Series[] = [
    { name: 'Strateji', color: 'var(--cyan)', values: points.map(d => change(d.equity)) },
    { name: 'Al ve tut', color: 'var(--green)', values: points.map(d => change(d.benchmark_equity)) },
  ]
  if (points.some(d => d.bist100_equity != null)) series.push({
    name: 'BIST 100', color: 'var(--purple)', values: points.map(d => d.bist100_equity == null ? null : change(d.bist100_equity)),
  })
  return <SeriesChart label="Kümülatif getiri" dates={points.map(d => d.valuation_date)} series={series}
    guides={[0]} format={value => percent(value / 100)}/>
}
