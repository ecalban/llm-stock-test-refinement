import type { ReactNode } from 'react'

type IconName =
  | 'grid'
  | 'chart'
  | 'history'
  | 'flask'
  | 'compare'
  | 'bell'
  | 'search'
  | 'chevron'
  | 'trend'
  | 'target'
  | 'shield'
  | 'activity'
  | 'play'
  | 'check'
  | 'clock'
  | 'code'
  | 'arrow'
  | 'menu'
  | 'close'
  | 'refresh'
  | 'download'
  | 'info'

const iconPaths: Record<IconName, ReactNode> = {
  grid: <><rect x="3" y="3" width="7" height="7" rx="2"/><rect x="14" y="3" width="7" height="7" rx="2"/><rect x="3" y="14" width="7" height="7" rx="2"/><rect x="14" y="14" width="7" height="7" rx="2"/></>,
  chart: <><path d="M4 19V9"/><path d="M10 19V5"/><path d="M16 19v-7"/><path d="M22 19H2"/></>,
  history: <><path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l3 2"/></>,
  flask: <><path d="M9 3h6"/><path d="M10 3v6l-5.4 8.7A2 2 0 0 0 6.3 21h11.4a2 2 0 0 0 1.7-3.3L14 9V3"/><path d="M7.5 15h9"/></>,
  compare: <><path d="M8 3v18"/><path d="M16 3v18"/><path d="m3 8 5-5 5 5"/><path d="m11 16 5 5 5-5"/></>,
  bell: <><path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9"/><path d="M10 21h4"/></>,
  search: <><circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/></>,
  chevron: <path d="m8 10 4 4 4-4"/>,
  trend: <><path d="m3 17 6-6 4 4 8-9"/><path d="M15 6h6v6"/></>,
  target: <><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/></>,
  shield: <><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10"/><path d="m9 12 2 2 4-5"/></>,
  activity: <path d="M3 12h4l2-7 4 14 2-7h6"/>,
  play: <path d="m8 5 11 7-11 7V5Z"/>,
  check: <path d="m5 12 4 4L19 6"/>,
  clock: <><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>,
  code: <><path d="m8 9-3 3 3 3"/><path d="m16 9 3 3-3 3"/><path d="m14 5-4 14"/></>,
  arrow: <><path d="M5 12h14"/><path d="m15 8 4 4-4 4"/></>,
  menu: <><path d="M4 7h16"/><path d="M4 12h16"/><path d="M4 17h16"/></>,
  close: <><path d="m6 6 12 12"/><path d="m18 6-12 12"/></>,
  refresh: <><path d="M20 7h-5V2"/><path d="M4 17h5v5"/><path d="M5.8 9a7 7 0 0 1 11.7-3L20 8"/><path d="M18.2 15a7 7 0 0 1-11.7 3L4 16"/></>,
  download: <><path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/></>,
  info: <><circle cx="12" cy="12" r="9"/><path d="M12 11v6"/><path d="M12 7h.01"/></>,
}

export function Icon({ name, size = 18 }: { name: IconName; size?: number }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{iconPaths[name]}</svg>
}

export function Button({ children, onClick, variant = 'primary', disabled = false, icon, className = '', ariaLabel }: { children: ReactNode; onClick?: () => void; variant?: 'primary' | 'secondary' | 'ghost'; disabled?: boolean; icon?: IconName; className?: string; ariaLabel?: string }) {
  return <button className={`btn btn-${variant} ${className}`} onClick={onClick} disabled={disabled} aria-label={ariaLabel}>{icon && <Icon name={icon} size={16}/>}<span>{children}</span></button>
}

export function Select({ label, value, onChange, children }: { label?: string; value: string; onChange: (value: string) => void; children: ReactNode }) {
  return <label className="field">{label && <span className="field-label">{label}</span>}<span className="select-wrap"><select value={value} onChange={(e) => onChange(e.target.value)}>{children}</select><Icon name="chevron" size={16}/></span></label>
}

export function Card({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <section className={`card ${className}`}>{children}</section>
}

export function CardHead({ title, sub, action }: { title: string; sub?: string; action?: ReactNode }) {
  return <div className="card-head"><div><div className="card-title">{title}</div>{sub && <div className="card-sub">{sub}</div>}</div>{action}</div>
}

export function Pill({ children, tone = 'neutral' }: { children: ReactNode; tone?: 'green' | 'red' | 'yellow' | 'cyan' | 'neutral' | 'purple' }) {
  return <span className={`pill pill-${tone}`}>{children}</span>
}

export function StatCard({ label, value, note, icon, tone = 'cyan' }: { label: string; value: string; note: string; icon: IconName; tone?: 'cyan' | 'green' | 'purple' | 'yellow' }) {
  return <Card className="stat-card">
    <div className={`stat-icon stat-${tone}`}><Icon name={icon}/></div>
    <div className="stat-copy"><div className="stat-label">{label}</div><div className="stat-value">{value}</div><div className="stat-note">{note}</div></div>
  </Card>
}

export function ResearchBanner({ onOpen }: { onOpen?: () => void }) {
  return <section className="research-banner">
    <div className="research-icon"><Icon name="flask" size={22}/></div>
    <div className="research-copy">
      <span className="research-kicker">ARAŞTIRMA KATKISI</span>
      <b>Açık ve kapalı LLM'lerde mutasyon geri bildiriminin etkisi</b>
      <p>Üç geri bildirim yöntemi; mutasyon skoru, test üretim verimliliği ve iterasyon bazlı iyileşme üzerinden karşılaştırılır.</p>
    </div>
    <div className="research-methods">
      <div><span>01</span><b>Yalnızca skor</b></div>
      <div><span>02</span><b>Satır + operatör</b></div>
      <div><span>03</span><b>Tam mutant kodu</b></div>
    </div>
    {onOpen && <Button variant="secondary" onClick={onOpen}>Sonuçları incele <Icon name="arrow" size={14}/></Button>}
  </section>
}


export function Notice({ message, error = false }: { message: string; error?: boolean }) {
  return <div className={`notice ${error ? 'notice-error' : ''}`} role={error ? 'alert' : 'status'}>
    <Icon name="info" size={16}/><span>{message}</span>
  </div>
}

export function Metric({ title, value, note, tone }: { title: string; value: string; note: string; tone: string }) {
  return <Card className={`metric metric-${tone}`}><span>{title}</span><b>{value}</b><small>{note}</small></Card>
}
