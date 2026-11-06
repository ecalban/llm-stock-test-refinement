import { useState } from 'react'
import {
  dateLabel, download, errorText, money, number, percent, post, signalLabels, signalTone,
  strategyLabels, ticker, useResource, type Analysis, type Backtest, type Capabilities,
  type Dashboard, type Experiment, type Stock, type Strategy,
} from './api'
import { EquityChart, SeriesChart } from './charts'
import { Button, Card, CardHead, Icon, Metric, Notice, Pill, ResearchBanner, Select, StatCard } from './ui'

const firstStock = (stocks: Stock[]) => (stocks.find(s => s.symbol === 'THYAO.IS') ?? stocks[0])?.symbol ?? ''
const datesBetween = (start: string, end: string) => `${dateLabel(start)} – ${dateLabel(end)}`
const gains = (value: number) => value >= 0 ? 'green' : 'red'
const ranges = [{ months: 1, label: '1 Ay' }, { months: 3, label: '3 Ay' }, { months: 6, label: '6 Ay' },
  { months: 12, label: '1 Yıl' }, { months: 36, label: '3 Yıl' }]

function monthsBefore(end: string, months: number) {
  const value = new Date(end + 'T00:00:00Z')
  const day = value.getUTCDate()
  value.setUTCDate(1)
  value.setUTCMonth(value.getUTCMonth() - months)
  const lastDay = new Date(Date.UTC(value.getUTCFullYear(), value.getUTCMonth() + 1, 0)).getUTCDate()
  value.setUTCDate(Math.min(day, lastDay))
  return value.toISOString().slice(0, 10)
}

function periods(end: string, period: string): { start: string; end: string } {
  switch (period) {
    case '2024': return { start: '2024-01-01', end: '2025-12-31' }
    case '2023': return { start: '2023-01-01', end: '2024-12-31' }
    case '3y': return { start: monthsBefore(end, 36), end }
    default: return { start: end.slice(0, 4) + '-01-01', end }
  }
}

export const runStatus: Record<Experiment['status'], string> = {
  queued: 'Sırada', running: 'Çalışıyor', failed: 'Başarısız', completed: 'Tamamlandı',
}

export function PanelScreen({ navigate, stocks, refresh }: {
  navigate: (screen: 'analiz' | 'test' | 'modeller') => void; stocks: Stock[]; refresh: number
}) {
  const dashboard = useResource<Dashboard>('/dashboard', refresh)
  const experiments = useResource<{ items: Experiment[] }>('/experiments', refresh)
  const data = dashboard.data
  const list = data?.latest_signals ?? []
  const feature = list.find(row => row.symbol === 'THYAO.IS') ?? list[0]
  const saved = useResource<Analysis>(feature ? '/analyses/' + feature.analysis_id : null)
  const history = experiments.data?.items ?? []
  const last = history[0]
  const lastScore = history.find(run => run.status === 'completed')?.rounds.at(-1)?.score
  return <div className="screen-stack">
    {dashboard.error && <Notice error message={dashboard.error}/>}
    {experiments.error && <Notice error message={experiments.error}/>}
    <div className="stats-grid">
      <StatCard label="Analiz Edilen Hisse" value={number(data?.analysed_stock_count, 0)}
        note={data ? `${data.available_stock_count} hisse için veri var` : 'Kayıtlı analizler'} icon="chart"/>
      <StatCard label="ML Doğruluğu" value={percent(data?.pooled_accuracy)}
        note={data ? `${number(data.accuracy_samples, 0)} değerlendirilen tahmin` : 'Kayıtlı analizler'} icon="target" tone="green"/>
      <StatCard label="Mutasyon Skoru" value={percent(lastScore)} note="Son tamamlanan deney" icon="shield" tone="purple"/>
      <StatCard label="Son Test Durumu" value={last ? runStatus[last.status] : '—'}
        note={last ? dateLabel(last.created_at) : 'Henüz deney yok'} icon="check" tone="green"/>
    </div>
    <ResearchBanner onOpen={() => navigate('modeller')}/>
    <div className="grid-main">
      <Card className="performance-card">
        <CardHead title="Strateji Performansı" sub={saved.data ?
          `${ticker(saved.data.symbol)} · ${datesBetween(saved.data.backtest.config.evaluation_start, saved.data.as_of)}` : 'Son kayıtlı analiz'}/>
        {saved.error && <Notice error message={saved.error}/>}
        {saved.data ? <>
          <div className="chart-summary"><div><b>{percent(saved.data.backtest.metrics.total_return)}</b><span>Toplam getiri</span></div>
            <Pill tone="neutral">Al ve tut: {percent(saved.data.backtest.metrics.buy_hold_return)}</Pill></div>
          <EquityChart result={saved.data.backtest}/>
        </> : <p className="empty-state">{dashboard.loading || saved.loading ? 'Sonuçlar yükleniyor…' : 'Kayıtlı analiz bulunamadı.'}</p>}
      </Card>
      <Card>
        <CardHead title="Son Sinyaller" sub="Kayıtlı günlük analizler"
          action={<Button variant="ghost" onClick={() => navigate('analiz')}>Analize git <Icon name="arrow" size={14}/></Button>}/>
        <div className="signal-list">{list.map(s => <div className="signal-row" key={s.symbol}>
          <div className="stock-avatar">{ticker(s.symbol).slice(0, 2)}</div>
          <div className="stock-name"><b>{ticker(s.symbol)}</b><span>{stocks.find(stock => stock.symbol === s.symbol)?.name ?? s.symbol}</span></div>
          <div className="stock-price"><b>{money(s.quote.close)}</b><span>{dateLabel(s.as_of)}</span></div>
          <Pill tone={signalTone(s.prediction.action)}>{signalLabels[s.prediction.action]} · {percent(s.prediction.probability)}</Pill>
        </div>)}</div>
        {!list.length && <p className="empty-state">{dashboard.loading ? 'Sinyaller yükleniyor…' : 'Henüz sinyal yok.'}</p>}
      </Card>
    </div>
    <Card>
      <CardHead title="Son Deneyler" sub="Kayıtlı pytest ve mutasyon sonuçları"
        action={<Button variant="ghost" onClick={() => navigate('test')}>Test merkezine git <Icon name="arrow" size={14}/></Button>}/>
      {history.length ? <div className="table-wrap"><table><thead><tr><th>Modül</th><th>Test Kaynağı</th>
        <th>Mutant</th><th>Skor</th><th>Durum</th><th>Tarih</th></tr></thead><tbody>
        {history.slice(0, 5).map(run => <tr key={run.id}><td>{run.config.target}.py</td><td>{run.model}</td>
          <td>{number(run.selected_mutants, 0)}</td><td>{percent(run.rounds.at(-1)?.score)}</td>
          <td><Pill tone={run.status === 'completed' ? 'green' : run.status === 'failed' ? 'red' : 'cyan'}>{runStatus[run.status]}</Pill></td>
          <td>{dateLabel(run.created_at)}</td></tr>)}
      </tbody></table></div> : <p className="empty-state">{experiments.loading ? 'Deneyler yükleniyor…' : 'Henüz deney çalıştırılmadı.'}</p>}
    </Card>
  </div>
}

export function AnalysisScreen({ stocks, caps, notify, onSaved }: {
  stocks: Stock[]; caps: Capabilities | null; notify: (message: string) => void; onSaved: () => void
}) {
  const [choice, setStock] = useState('')
  const [range, setRange] = useState('12')
  const [model, setModel] = useState('logistic_signal')
  const [result, setResult] = useState<Analysis | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')
  const stock = choice || firstStock(stocks)
  const catalogRow = stocks.find(s => s.symbol === stock)
  const dashboard = useResource<Dashboard>('/dashboard')
  const prior = dashboard.data?.latest_signals.find(item => item.symbol === stock)
  const saved = useResource<Analysis>(prior && result?.symbol !== stock ? '/analyses/' + prior.analysis_id : null)
  const data = result?.symbol === stock ? result : saved.data
  const prediction = data?.latest_prediction
  const bars = data?.series ?? []
  const history = [...(data?.signals ?? [])].reverse()
  const [page, setPage] = useState(0)
  const visiblePage = Math.min(page, Math.max(0, Math.ceil(history.length / 20) - 1))
  const labels = bars.map(bar => bar.date)
  const refresh = async () => {
    if (!catalogRow) return
    setError(''); setRunning(true)
    try {
      setResult(await post<Analysis>('/analyses', { symbol: stock, model, start: monthsBefore(catalogRow.as_of, +range), end: catalogRow.as_of }))
      setPage(0); onSaved(); notify(`${ticker(stock)} analizi güncellendi.`)
    } catch (e) { setError(errorText(e)) } finally { setRunning(false) }
  }
  const csv = async () => {
    if (!data) return
    try { await download(`/analyses/${data.id}/signals?format=csv`, `${ticker(data.symbol)}-sinyaller.csv`) }
    catch (e) { setError(errorText(e)) }
  }
  return <div className="screen-stack">
    <Card className="filter-bar">
      <Select label="Hisse Senedi" value={stock} onChange={value => { setStock(value); setPage(0); setError('') }}>
        {!stocks.length && <option value="">Hisse bulunamadı</option>}
        {stocks.map(s => <option key={s.symbol} value={s.symbol}>{ticker(s.symbol)}</option>)}
      </Select>
      <Select label="Tarih Aralığı" value={range} onChange={setRange}>
        {ranges.map(r => <option key={r.months} value={r.months}>{r.label}</option>)}
      </Select>
      <Select label="Makine Öğrenmesi Modeli" value={model} onChange={setModel}>
        {(caps?.models ?? []).map(m => <option key={m.id} value={m.id} disabled={!m.available}>{m.name}{m.available ? '' : ' · kullanılamıyor'}</option>)}
      </Select>
      <Button icon="refresh" onClick={refresh} disabled={running || !stock || !caps}>{running ? 'Hesaplanıyor…' : 'Analizi Güncelle'}</Button>
    </Card>
    {(error || saved.error || dashboard.error) && <Notice error message={error || saved.error || dashboard.error || ''}/>}
    {!data && <Card><p className="empty-state">{running || saved.loading || dashboard.loading ? 'Analiz yükleniyor…' : 'Bir hisse seçip Analizi Güncelle’ye tıkla.'}</p></Card>}
    {data && prediction && <>
      <div className="analysis-grid data-refresh" key={data.id}>
        <div className="screen-stack">
          <Card>
            <CardHead title={`${ticker(data.symbol)} Fiyat Grafiği`} sub={`${datesBetween(data.backtest.config.evaluation_start, data.as_of)} · Günlük kapanış`}
              action={<div className="quote"><b>{money(data.quote.close)}</b><Pill tone={gains(data.quote.daily_change)}>{percent(data.quote.daily_change)}</Pill></div>}/>
            <SeriesChart label="Günlük kapanış fiyatı" dates={labels} series={[{ name: 'Kapanış', values: bars.map(b => b.close), color: 'var(--cyan)' }]}/>
          </Card>
          <div className="indicator-grid">
            <Card><CardHead title="RSI (14)" action={<Pill tone="yellow">{number(bars.at(-1)?.rsi, 1)}</Pill>}/>
              <SeriesChart compact label="RSI" dates={labels} bounds={[0, 100]} guides={[30, 70]}
                series={[{ name: 'RSI', values: bars.map(b => b.rsi), color: 'var(--yellow)' }]}/></Card>
            <Card><CardHead title="MACD" action={<Pill tone="cyan">{number(bars.at(-1)?.macd, 2)}</Pill>}/>
              <SeriesChart compact label="MACD ve sinyal" dates={labels} guides={[0]} series={[
                { name: 'Histogram', values: bars.map(b => b.macd_histogram), color: 'var(--green)', bars: true },
                { name: 'MACD', values: bars.map(b => b.macd), color: 'var(--cyan)' },
                { name: 'Sinyal', values: bars.map(b => b.macd_signal), color: 'var(--yellow)' },
              ]}/>
              <div className="indicator-legend"><span><i className="line-sample cyan"/>MACD</span><span><i className="line-sample yellow"/>Sinyal</span><span><i className="bar-sample"/>Histogram</span></div>
            </Card>
          </div>
        </div>
        <Card className="prediction-card">
          <div className="prediction-kicker"><Icon name="activity" size={15}/> YAPAY ZEKÂ TAHMİNİ</div>
          <div className="prediction-signal" data-tone={signalTone(prediction.action)}>{signalLabels[prediction.action]}</div>
          <div className="prediction-prob">{percent(prediction.probability)}</div><div className="prediction-label">Seçilen sınıfın olasılığı</div>
          <div className="class-probabilities">{(['buy', 'hold', 'sell'] as const).map(s =>
            <div key={s} className={s === prediction.action ? 'active' : ''}><span><b>{signalLabels[s]}</b><small>{percent(prediction.probabilities[s])}</small></span>
              <i><span style={{ width: `${prediction.probabilities[s] * 100}%` }}/></i></div>)}</div>
          <div className="prediction-facts"><div><span>Son seans</span><b>{dateLabel(data.as_of)}</b></div><div><span>Model</span><b>Lojistik Regresyon</b></div>
            <div><span>Doğruluk</span><b>{percent(data.classification_metrics.accuracy)}</b></div><div><span>Makro F1</span><b>{number(data.classification_metrics.macro_f1, 3)}</b></div></div>
          <div className="model-note"><Icon name="info" size={16}/><span>Tahmin ufku: ertesi açılıştan bir sonraki açılışa.</span></div>
        </Card>
      </div>
      <Card>
        <CardHead title="Geçmiş Sinyaller" sub={`${history.length} tahmin · ${data.classification_metrics.pending} sonucu bekleniyor`}
          action={<Button variant="secondary" icon="download" onClick={csv}>CSV İndir</Button>}/>
        <div className="table-wrap"><table><thead><tr><th>Tarih</th><th>Tahmin</th><th>Kapanış</th><th>Olasılık</th><th>Gerçekleşen</th><th>Sonuç</th></tr></thead>
          <tbody>{history.slice(visiblePage * 20, (visiblePage + 1) * 20).map(s => <tr key={s.date}>
            <td>{dateLabel(s.date)}</td><td><Pill tone={signalTone(s.action)}>{signalLabels[s.action]}</Pill></td>
            <td>{money(s.close)}</td><td>{percent(s.probability)}</td>
            <td>{s.actual_class ? signalLabels[s.actual_class] : '—'}</td>
            <td><Pill tone={s.correct == null ? 'neutral' : s.correct ? 'green' : 'red'}>{s.correct == null ? 'Bekleniyor' : s.correct ? 'Doğru' : 'Yanlış'}</Pill></td>
          </tr>)}</tbody></table></div>
        {history.length > 20 && <div className="pager"><Button variant="secondary" disabled={!visiblePage} onClick={() => setPage(visiblePage - 1)}>Önceki</Button>
          <span>Sayfa {visiblePage + 1} / {Math.ceil(history.length / 20)}</span>
          <Button variant="secondary" disabled={(visiblePage + 1) * 20 >= history.length} onClick={() => setPage(visiblePage + 1)}>Sonraki</Button></div>}
      </Card>
    </>}
  </div>
}

export function BacktestScreen({ stocks, hasIndex, caps, notify }: {
  stocks: Stock[]; hasIndex: boolean; caps: Capabilities | null; notify: (message: string) => void
}) {
  const [choice, setStock] = useState('')
  const [strategy, setStrategy] = useState<Strategy>('ml')
  const [period, setPeriod] = useState('ytd')
  const [benchmark, setBenchmark] = useState(true)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<Backtest | null>(null)
  const [error, setError] = useState('')
  const [lastRun, setLastRun] = useState(() => sessionStorage.getItem('bist:last-backtest'))
  const saved = useResource<Backtest>(lastRun && !result ? '/backtests/' + lastRun : null)
  const run = result ?? saved.data
  const stock = choice || firstStock(stocks)
  const current = stocks.find(s => s.symbol === stock)
  const stats = run?.metrics
  const execute = async () => {
    if (!current) return
    setRunning(true); setError('')
    try {
      const value = await post<Backtest>('/backtests', { symbol: stock, strategy, ...periods(current.as_of, period), benchmark: hasIndex && benchmark })
      setResult(value); setLastRun(value.id); sessionStorage.setItem('bist:last-backtest', value.id)
      notify(`${ticker(stock)} backtesti tamamlandı.`)
    } catch (e) { setError(errorText(e)) } finally { setRunning(false) }
  }
  const csv = async () => {
    if (!run) return
    try { await download(`/backtests/${run.id}/trades?format=csv`, `${ticker(run.symbol)}-islemler.csv`) }
    catch (e) { setError(errorText(e)) }
  }
  return <div className="screen-stack">
    <Card className="filter-bar backtest-filter">
      <Select label="Hisse Senedi" value={stock} onChange={setStock}>
        {!stocks.length && <option value="">Hisse bulunamadı</option>}
        {stocks.map(s => <option key={s.symbol} value={s.symbol}>{ticker(s.symbol)}</option>)}
      </Select>
      <Select label="Strateji" value={strategy} onChange={s => setStrategy(s as Strategy)}>
        {(caps?.strategies ?? []).map(s => <option key={s} value={s}>{strategyLabels[s]}</option>)}
      </Select>
      <Select label="Dönem" value={period} onChange={setPeriod}><option value="ytd">Son verinin yılı</option>
        <option value="2024">2024–2025</option><option value="2023">2023–2024</option><option value="3y">Son 3 yıl</option></Select>
      <label className="check-field"><input type="checkbox" checked={hasIndex && benchmark} disabled={!hasIndex}
        onChange={e => setBenchmark(e.target.checked)}/> BIST 100</label>
      <Button icon="play" onClick={execute} disabled={running || !current || !caps}>{running ? 'Hesaplanıyor…' : 'Backtest Çalıştır'}</Button>
    </Card>
    {(error || saved.error) && <Notice error message={error || saved.error || ''}/>}
    {!run && <Card><p className="empty-state">{running || saved.loading ? 'Backtest yükleniyor…' : 'Bir strateji seçip backtesti çalıştır.'}</p></Card>}
    {run && stats && <>
      <div className="metrics-grid data-refresh" key={run.id}>
        <Metric title="Toplam Getiri" value={percent(stats.total_return)} note={stats.bist100_return == null ?
          `Al ve tut: ${percent(stats.buy_hold_return)}` : `BIST 100: ${percent(stats.bist100_return)}`} tone={gains(stats.total_return)}/>
        <Metric title="Sharpe Oranı" value={number(stats.sharpe_ratio)} note="Yıllıklandırılmış risk/getiri" tone="cyan"/>
        <Metric title="Maks. Düşüş" value={percent(stats.max_drawdown)} note={stats.recovery_sessions == null ?
          'Zirveye dönüş gerçekleşmedi' : `Toparlanma: ${stats.recovery_sessions} seans`} tone="red"/>
        <Metric title="Kapalı Pozisyon" value={number(stats.closed_trade_count, 0)}
          note={`${stats.order_count} emir · kazanma ${percent(stats.win_rate)}`} tone="purple"/>
      </div>
      <div className="grid-main backtest-charts">
        <Card><CardHead title="Strateji ve Karşılaştırmalar"
          sub={`${ticker(run.symbol)} · ${strategyLabels[run.config.strategy]} · ${datesBetween(run.config.evaluation_start, run.config.evaluation_end)}`}/>
          <div className="indicator-legend"><span><i className="dot cyan"/>Strateji</span><span><i className="dot green"/>Al ve tut</span>
            {stats.bist100_return != null && <span><i className="dot benchmark-dot"/>BIST 100</span>}</div><EquityChart result={run}/></Card>
        <Card><CardHead title="Portföy Düşüşü" sub="Zirveden yüzdesel gerileme"/>
          <SeriesChart label="Portföy düşüşü" dates={run.daily.map(d => d.valuation_date)} guides={[0]}
            format={v => percent(v, 1)} series={[{ name: 'Düşüş', values: run.daily.map(d => d.drawdown), color: 'var(--red)' }]}/></Card>
      </div>
      <Card>
        <CardHead title="İşlem Geçmişi" sub={`Komisyon ${run.config.commission_bps} bp · kayma ${run.config.slippage_bps} bp · toplam ${money(stats.total_cost)}`}
          action={<Button variant="secondary" icon="download" onClick={csv}>CSV İndir</Button>}/>
        {run.trades.length ? <div className="table-wrap"><table><thead><tr><th>Giriş Tarihi</th><th>Çıkış Tarihi</th><th>Yön</th>
          <th>Giriş</th><th>Çıkış</th><th>İşlem Ücreti</th><th>Net Getiri</th><th>Süre</th></tr></thead>
          <tbody>{run.trades.map(t => <tr key={t.entry_date}><td>{dateLabel(t.entry_date)}</td><td>{dateLabel(t.exit_date)}</td>
            <td><Pill tone="green">UZUN</Pill></td><td>{money(t.entry_price)}</td><td>{money(t.exit_price)}</td>
            <td>{money(t.total_cost)}</td><td className={`${gains(t.net_return)}-text`}>{percent(t.net_return, 2)}</td>
            <td>{t.duration_sessions} seans</td></tr>)}</tbody></table></div> : <p className="empty-state">Bu dönemde kapanan pozisyon yok.</p>}
      </Card>
    </>}
  </div>
}
