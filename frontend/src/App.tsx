import { useEffect, useState } from 'react'
import { dateLabel, useResource, type Capabilities, type Stock } from './api'
import { AnalysisScreen, BacktestScreen, PanelScreen } from './MarketScreens'
import { ComparisonScreen, TestCenter } from './Experiments'
import { Icon, Notice } from './ui'

const menu = [
  { id: 'panel', label: 'Ana Panel', description: 'Piyasa ve deney süreçlerine genel bakış', icon: 'grid' },
  { id: 'analiz', label: 'Hisse Analizi', description: 'Makine öğrenmesi destekli teknik analiz', icon: 'chart' },
  { id: 'backtest', label: 'Backtest Sonuçları', description: 'Strateji performansını geçmiş verilerle ölçün', icon: 'history' },
  { id: 'test', label: 'LLM Test Merkezi', description: 'Finansal Python kodu için test üretimi ve mutasyon deneyleri', icon: 'flask' },
  { id: 'modeller', label: 'Model Karşılaştırmaları', description: 'LLM ve geri bildirim yöntemlerini karşılaştırın', icon: 'compare' },
] as const
type Screen = typeof menu[number]['id']
const fromHash = (): Screen => menu.find(entry => '#' + entry.id === location.hash)?.id ?? 'panel'

export default function App() {
  const [screen, setScreen] = useState<Screen>(fromHash)
  const [mobileOpen, setMobileOpen] = useState(false)
  const [notification, setNotification] = useState('')
  const [refresh, setRefresh] = useState(0)
  const stockCatalog = useResource<{ items: Stock[] }>('/stocks')
  const caps = useResource<Capabilities>('/capabilities')
  const stocks = stockCatalog.data?.items.filter(s => s.kind === 'stock') ?? []
  const index = stockCatalog.data?.items.find(s => s.kind === 'index')
  const latest = stocks.map(s => s.as_of).sort().at(-1)
  const connectionError = stockCatalog.error || caps.error
  useEffect(() => {
    const change = () => { setScreen(fromHash()); setMobileOpen(false); window.scrollTo(0, 0) }
    window.addEventListener('hashchange', change)
    return () => window.removeEventListener('hashchange', change)
  }, [])
  useEffect(() => {
    if (!notification) return
    const timeout = window.setTimeout(() => setNotification(''), 3500)
    return () => window.clearTimeout(timeout)
  }, [notification])
  const current = menu.find(entry => entry.id === screen)!
  const navigate = (id: Screen) => { location.hash = id; setMobileOpen(false) }
  return <div className="app-shell">
    <aside className={`sidebar ${mobileOpen ? 'open' : ''}`}>
      <div className="brand"><div className="brand-mark"><svg viewBox="0 0 32 32"><path d="M6 22 12 10l5 8 4-11 5 15"/><path d="M5 26h22"/></svg></div><div><b>BIST AI</b><span>LAB</span></div></div>
      <div className="nav-label">ÇALIŞMA ALANI</div>
      <nav aria-label="Çalışma alanı">{menu.map(entry => <button key={entry.id} className={screen === entry.id ? 'active' : ''}
        aria-current={screen === entry.id ? 'page' : undefined} onClick={() => navigate(entry.id)}>
        <Icon name={entry.icon}/><span>{entry.label}</span>{screen === entry.id && <i/>}</button>)}</nav>
      <div className="sidebar-bottom">
        <div className="system-state"><span className={connectionError ? 'pulse offline' : 'pulse'}/><div>
          <b>{connectionError ? 'API Bağlantısı Yok' : stockCatalog.loading || caps.loading ? 'Bağlanıyor…' : 'Yerel API Bağlı'}</b>
          <small>{latest ? `Veri: ${dateLabel(latest)}` : 'BIST günlük verileri'}</small>
        </div></div>
        <div className="profile"><div className="avatar">PE</div><div><b>Proje Ekibi</b><span>Bilgisayar Mühendisliği</span></div></div>
      </div>
    </aside>
    {mobileOpen && <button className="sidebar-overlay" onClick={() => setMobileOpen(false)} aria-label="Menüyü kapat"/>}
    <main>
      <header>
        <button className="mobile-menu" onClick={() => setMobileOpen(!mobileOpen)} aria-label={mobileOpen ? 'Gezinmeyi kapat' : 'Menüyü aç'} aria-expanded={mobileOpen}><Icon name={mobileOpen ? 'close' : 'menu'}/></button>
        <div className="page-heading"><h1 className="page-title">{current.label}</h1><div className="page-desc">{current.description}</div></div>
        <div className="header-actions">{latest && <div className="header-date"><span>Son veri</span><b>{dateLabel(latest)}</b></div>}</div>
      </header>
      <div className="demo-banner"><Icon name="info" size={15}/><span>Geçmiş piyasa verileri ve araştırma sonuçları · Yatırım tavsiyesi değildir.</span></div>
      <div className="content">
        {connectionError && <Notice error message={connectionError}/>}
        {screen === 'panel' && <PanelScreen navigate={navigate} stocks={stocks} refresh={refresh}/>}
        {screen === 'analiz' && <AnalysisScreen stocks={stocks} caps={caps.data} notify={setNotification} onSaved={() => setRefresh(v => v + 1)}/>}
        {screen === 'backtest' && <BacktestScreen stocks={stocks} hasIndex={!!index} caps={caps.data} notify={setNotification}/>}
        {screen === 'test' && <TestCenter/>}
        {screen === 'modeller' && <ComparisonScreen/>}
      </div>
    </main>
    {notification && <div className="toast" role="status"><span><Icon name="check" size={16}/></span>
      <div><b>{notification}</b></div><button onClick={() => setNotification('')} aria-label="Bildirimi kapat"><Icon name="close" size={15}/></button></div>}
  </div>
}
