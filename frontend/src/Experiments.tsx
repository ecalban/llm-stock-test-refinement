import { useEffect, useRef, useState, type CSSProperties } from 'react'
import {
  api, dateLabel, download, errorText, feedbackLabels, number, percent, post, useResource,
  type Comparison, type Experiment, type Feedback, type Mutant, type Options, type Provider,
} from './api'
import { SeriesChart } from './charts'
import { Button, Card, CardHead, Icon, Notice, Pill, ResearchBanner, Select } from './ui'

const statusLabels: Record<Experiment['status'], string> = {
  queued: 'Sırada', running: 'Çalışıyor', failed: 'Başarısız', completed: 'Tamamlandı',
}
const mutationLabels: Record<Mutant['status'], string> = {
  killed: 'Yakalandı', survived: 'Hayatta', timeout: 'Zaman aşımı', error: 'Çalıştırma hatası',
}
const steps = [
  ['generate', 'Test Üretimi'], ['validate', 'Pytest'], ['mutate', 'Mutasyon Testi'],
  ['feedback', 'Geri Bildirim'], ['completed', 'Sonuç'],
]
const busy = (run?: Experiment | null) => run?.status === 'queued' || run?.status === 'running'

function MutantDetails({ mutant, close, previous }: { mutant: Mutant; close: () => void; previous?: Mutant }) {
  const [tab, setTab] = useState<'diff' | 'source'>('diff')
  const dialog = useRef<HTMLDialogElement>(null)
  useEffect(() => { dialog.current?.showModal() }, [])
  return <dialog className="mutant-modal actual-modal" ref={dialog} aria-label="Mutant ayrıntısı"
    onCancel={event => { event.preventDefault(); close() }}>
    <div className="modal-head">
      <CardHead title={`Mutant · ${mutant.id.slice(0, 8)}`} sub={`Satır ${mutant.line} · ${mutationLabels[mutant.status]}`}/>
      <Button variant="ghost" icon="close" onClick={close} ariaLabel="Mutant ayrıntısını kapat">Kapat</Button>
    </div>
    <div className="modal-body">
      <div className="code-compare">
        <div><h3>Özgün satır</h3><pre>{mutant.original}</pre></div>
        <div><h3>Değişen satır</h3><pre>{mutant.replacement}</pre></div>
      </div>
      {previous && <p>Önceki tur: {mutationLabels[previous.status]} → Bu tur: {mutationLabels[mutant.status]}</p>}
      <div className="feedback-tabs">
        <Button variant={tab === 'diff' ? 'primary' : 'secondary'} onClick={() => setTab('diff')}>Kod Farkı</Button>
        <Button variant={tab === 'source' ? 'primary' : 'secondary'} onClick={() => setTab('source')}>Tam Mutant Kodu</Button>
      </div>
      <pre className="full-code">{tab === 'diff' ? mutant.diff : mutant.source}</pre>
      {mutant.tests.length > 0 && <><h3>Mutasyonu yakalayan testler</h3><ul>{mutant.tests.map((test, i) => <li key={i}><code>{test}</code></li>)}</ul></>}
      {mutant.detail && <pre className="full-code">{mutant.detail}</pre>}
    </div>
  </dialog>
}

export function TestCenter() {
  const [target, setTarget] = useState('indicators')
  const [provider, setProvider] = useState<Provider>('reference')
  const [feedback, setFeedback] = useState<Feedback>('score')
  const [rounds, setRounds] = useState('3')
  const [mutants, setMutants] = useState('8')
  const [version, setVersion] = useState(0)
  const [selectedId, setSelectedId] = useState('')
  const [detail, setDetail] = useState<Experiment | null>(null)
  const [mutant, setMutant] = useState<Mutant | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState('')
  const options = useResource<Options>('/experiments/options', version)
  const history = useResource<{ items: Experiment[] }>('/experiments', version)
  const list = history.data?.items ?? []
  const selected = list.find(run => run.id === selectedId) ?? list[0]
  const id = selectedId || selected?.id
  const run = detail?.id === id ? detail : selected?.id === id ? selected : null
  const running = busy(run)
  const targetInfo = options.data?.targets.find(t => t.id === target)
  const providers = options.data?.providers ?? []
  const selectable = providers.find(p => p.id === provider)
  useEffect(() => {
    if (!id) return
    const controller = new AbortController()
    let timer: number | undefined
    let wasBusy = false
    const poll = async () => {
      try {
        const next = await api<Experiment>('/experiments/' + id, { signal: controller.signal })
        if (controller.signal.aborted) return
        setDetail(next)
        if (busy(next)) {
          wasBusy = true
          timer = window.setTimeout(poll, 700)
        } else if (wasBusy) setVersion(v => v + 1)
      } catch (e) { if (!controller.signal.aborted) setError(errorText(e)) }
    }
    void poll()
    return () => { controller.abort(); window.clearTimeout(timer) }
  }, [id])

  const start = async () => {
    setError(''); setSubmitting(true); setMutant(null)
    try {
      const queued = await post<Experiment>('/experiments', {
        target, provider, feedback, rounds: provider === 'reference' ? 1 : +rounds, mutant_limit: +mutants,
      })
      setDetail(queued); setSelectedId(queued.id); setVersion(v => v + 1)
    } catch (e) { setError(errorText(e)) } finally { setSubmitting(false) }
  }
  const last = run?.rounds.at(-1)
  const phase = steps.findIndex(([value]) => run?.phase === value)
  const progress = running && run?.phase === 'mutate' && run.selected_mutants
    ? (run.mutants_done ?? 0) / run.selected_mutants : null
  const status = run ? statusLabels[run.status] : 'Hazır'
  const statusTone = run?.status === 'failed' ? 'red' : run?.status === 'completed' ? 'green' : 'cyan'
  const blocked = submitting || running || list.some(item => busy(item) && item.id !== run?.id)
  return <div className="screen-stack">
    {(error || options.error || history.error) && <Notice error message={error || options.error || history.error || ''}/>}
    <div className="test-layout">
      <Card className="test-config">
        <CardHead title="Deney Yapılandırması" sub="Gerçek pytest ve mutmut çalıştırması"/>
        <Select label="Python Modülü" value={target} onChange={value => {
          setTarget(value)
          if (!options.data?.targets.find(item => item.id === value)?.llm) setProvider('reference')
        }}>{options.data?.targets.map(t => <option key={t.id} value={t.id}>{t.name}</option>)}</Select>
        <Select label="Test Kaynağı" value={provider} onChange={value => setProvider(value as Provider)}>
          {providers.map(p => <option key={p.id} value={p.id}
            disabled={!p.available || (p.id !== 'reference' && !targetInfo?.llm)}>
            {p.name}{p.available ? '' : ' · kullanılamıyor'}
          </option>)}
        </Select>
        {providers.filter(p => p.reason).map(p => <p key={p.id} className="form-note">{p.name}: {p.reason}</p>)}
        {!targetInfo?.llm && options.data && <p className="form-note">Bu modülde repo referans testleri destekleniyor.</p>}
        {provider !== 'reference' && <>
          <Select label="Geri Bildirim Yöntemi" value={feedback} onChange={v => setFeedback(v as Feedback)}>
            {Object.entries(feedbackLabels).map(([k, label]) => <option key={k} value={k}>{label}</option>)}
          </Select>
          <Select label="En Fazla Tur" value={rounds} onChange={setRounds}><option>1</option><option>3</option><option>5</option></Select>
        </>}
        <Select label="Mutant Örneklemi" value={mutants} onChange={setMutants}><option>8</option><option>16</option><option>24</option></Select>
        <div className="config-summary"><Icon name="info" size={16}/>
          <span>{provider === 'reference' ? 'Repo testleriyle tek tur ölçüm.' : 'Modelin ürettiği vakalar önce özgün kodda doğrulanır.'}</span></div>
        <Button icon="play" onClick={start} disabled={blocked || !selectable?.available}>{blocked ? 'Deney çalışıyor…' : 'Testi Başlat'}</Button>
      </Card>
      <div className="screen-stack">
        <Card>
          <CardHead title="Deney Durumu" sub={run ? `${run.config.target}.py · ${run.model}` : 'Bir deney başlat veya geçmişten seç'}
            action={<Pill tone={statusTone}>{status}</Pill>}/>
          <div className="pipeline-steps">{steps.map(([value, label], i) => {
            const skipped = run?.config.provider === 'reference' && (value === 'generate' || value === 'feedback')
            return <div key={value} className={!run || skipped ? '' : phase > i ? 'done' : phase === i ? 'active' : ''}>
              <span>{skipped ? '—' : phase > i ? <Icon name="check" size={12}/> : i + 1}</span><small>{label}</small>
            </div>
          })}</div>
          {run?.error && <Notice error message={run.error}/>}
          {running && <div className="progress-banner" role="status"><span className="spinner"/>
            <div><b>Tur {run?.round || 1}/{run?.config.rounds} · {steps[phase]?.[1] ?? 'Hazırlanıyor'}</b>
              {progress != null && <span>{run?.mutants_done ?? 0}/{run?.selected_mutants} mutant çalıştırıldı</span>}</div>
            <div className={`progress-track ${progress == null ? '' : 'determinate'}`}><span style={progress == null ? undefined : { width: (progress * 100) + '%' }}/></div>
          </div>}
          {last && <div className="big-progress">
            <div className="progress-ring" style={{ '--progress': `${last.score * 100}%` } as CSSProperties}>
              <div><b>{percent(last.score)}</b><span>Mutasyon Skoru</span></div>
            </div>
            <div className="test-counts">
              <div><i className="count-dot green"/><span>Yakalanan</span><b>{last.killed}</b></div>
              <div><i className="count-dot red"/><span>Hayatta</span><b>{last.survived}</b></div>
              <div><i className="count-dot cyan"/><span>Test paketi</span><b>{last.suite_size}</b></div>
              <p className="form-note">{last.timeout} zaman aşımı · {last.errors} hata</p>
            </div>
          </div>}
          {run?.catalog && <p className="form-note">Katalog {run.catalog} · {run.selected_mutants}/{run.pool_size} mutant · skor = yakalanan / seçilen</p>}
        </Card>
        {run && run.rounds.length > 1 && <Card><CardHead title="Tur Bazında Mutasyon Skoru"/>
          <SeriesChart label="Tur bazında mutasyon skoru"
            dates={run.rounds.map((_, i) => new Date(Date.UTC(2000, 0, i + 1)).toISOString())}
            bounds={[0, 1]} format={v => percent(v, 0)}
            series={[{ name: 'Skor', color: 'var(--cyan)', values: run.rounds.map(r => r.score) }]} tickLabel={i => `Tur ${i + 1}`}/>
        </Card>}
        {last?.validation && <Card>
          <CardHead title={run?.config.provider === 'reference' ? 'Doğrulanan Referans Testleri' : 'Üretilen Test Vakaları'}
            sub={`${last.valid}/${last.generated} özgün kodda geçti · ${last.suite_size} test pakette`}/>
          <div className="test-results">{last.validation.map((test, i) => <details key={i}>
            <summary><code>{test.name}</code><Pill tone={test.passed ? 'green' : 'red'}>{test.passed ? 'Geçti' : 'Kaldı'}</Pill></summary>
            {test.case ? <pre className="full-code">{JSON.stringify(test.case, null, 2)}</pre>
              : <p className="form-note">Repo testi özgün kod üzerinde geçti.</p>}
          </details>)}</div>
        </Card>}
        {last?.mutants && <Card>
          <CardHead title="Mutasyon Sonuçları" sub={`Tur ${last.round} · gerçek kod farkları`}/>
          <div className="table-wrap"><table><thead><tr><th>Mutant</th><th>Satır</th><th>Durum</th><th>Test</th><th>Ayrıntı</th></tr></thead>
            <tbody>{last.mutants.map(m => <tr key={m.id}><td><code>{m.id.slice(0, 8)}</code></td><td>{m.line}</td>
              <td><Pill tone={m.status === 'killed' ? 'green' : m.status === 'survived' ? 'red' : 'yellow'}>{mutationLabels[m.status]}</Pill></td>
              <td>{m.tests.length}</td><td><Button variant="secondary" onClick={() => setMutant(m)}>Ayrıntı</Button></td></tr>)}</tbody>
          </table></div>
        </Card>}
      </div>
    </div>
    <Card>
      <CardHead title="Çalıştırma Geçmişi" sub="Sunucuda kaydedilen deneyler"
        action={<Button icon="refresh" variant="secondary" onClick={() => setVersion(v => v + 1)}>Yenile</Button>}/>
      {list.length ? <div className="table-wrap"><table><thead><tr><th>Modül</th><th>Kaynak</th><th>Geri Bildirim</th>
        <th>Tur</th><th>Skor</th><th>Durum</th><th>Sonuç</th></tr></thead><tbody>
        {list.map(item => <tr key={item.id}><td>{item.config.target}.py</td><td>{item.model}</td>
          <td>{item.config.feedback ? feedbackLabels[item.config.feedback] : '—'}</td><td>{item.rounds.length}/{item.config.rounds}</td>
          <td>{percent(item.rounds.at(-1)?.score)}</td><td>{statusLabels[item.status]}</td>
          <td><Button variant="ghost" onClick={() => { setSelectedId(item.id); setError(''); setMutant(null) }}>Sonucu Gör</Button></td></tr>)}
      </tbody></table></div> : <p className="empty-state">{history.loading ? 'Geçmiş yükleniyor…' : 'Henüz kayıt yok.'}</p>}
    </Card>
    {mutant && <MutantDetails mutant={mutant} close={() => setMutant(null)}
      previous={run?.rounds.at(-2)?.mutants?.find(item => item.id === mutant.id)}/>}
  </div>
}

const cohortKey = (r: Comparison) => `${r.target}:${r.catalog}:${r.planned_rounds}`
export function ComparisonScreen() {
  const result = useResource<{ items: Comparison[] }>('/experiments/comparisons')
  const [choice, setChoice] = useState('')
  const [error, setError] = useState('')
  const all = result.data?.items ?? []
  const cohorts = [...new Map(all.map(row => [cohortKey(row), row])).values()]
  const current = cohorts.find(row => cohortKey(row) === choice) ?? cohorts[0]
  const rows = current ? all.filter(row => cohortKey(row) === cohortKey(current)) : []
  const mean = (get: (row: Comparison) => number) =>
    rows.length ? rows.reduce((sum, row) => sum + get(row), 0) / rows.length : null
  const csv = async () => {
    if (!current) return
    try { await download(`/experiments/comparisons?catalog=${current.catalog}&rounds=${current.planned_rounds}&format=csv`, 'llm-deneyleri.csv') }
    catch (e) { setError(errorText(e)) }
  }
  return <div className="screen-stack">
    <ResearchBanner/>
    {(error || result.error) && <Notice error message={error || result.error || ''}/>}
    <Card className="filter-bar compare-filter">
      <div className="filter-title"><Icon name="compare"/><div><b>Deney Karşılaştırması</b><span>{rows.length} kayıt</span></div></div>
      <Select label="Ortak Mutant Kataloğu ve Tur Bütçesi" value={current ? cohortKey(current) : ''} onChange={setChoice}>
        {!cohorts.length && <option value="">Karşılaştırılabilir deney yok</option>}
        {cohorts.map(row => <option key={cohortKey(row)} value={cohortKey(row)}>{row.target}.py · {row.catalog} · {row.planned_rounds} tur</option>)}
      </Select>
      <Button variant="secondary" icon="download" onClick={csv} disabled={!rows.length}>CSV İndir</Button>
    </Card>
    {rows.length ? <>
      <div className="comparison-highlights"><div><span>Ortalama Başlangıç</span><b>{percent(mean(row => row.initial_score))}</b></div>
        <div><span>Ortalama Nihai Skor</span><b>{percent(mean(row => row.final_score))}</b></div>
        <div><span>Ortalama Değişim</span><b>{number((mean(row => row.delta) ?? 0) * 100)} yüzde puan</b></div></div>
      <div className="grid-main comparison-charts">
        <Card><CardHead title="Yönteme Göre Nihai Skor" sub="Aynı katalogdaki ölçülmüş koşular"/>
          <div className="actual-bars">{(Object.keys(feedbackLabels) as Feedback[]).map(method => {
            const matching = rows.filter(row => row.feedback === method)
            const average = matching.length ? matching.reduce((sum, row) => sum + row.final_score, 0) / matching.length : null
            return <div key={method}><span>{feedbackLabels[method]} · {matching.length} koşu</span>
              <div><i className={`method-${method}`} style={{ width: `${(average ?? 0) * 100}%` }}/></div><b>{percent(average)}</b></div>
          })}</div>
        </Card>
        <Card><CardHead title="Karşılaştırma Protokolü"/><p className="empty-state">
          {current?.mutants} ortak mutant ve en fazla {current?.planned_rounds} tur. Tamamlanmış LLM deneyleri gösteriliyor.
        </p></Card>
      </div>
      <Card><CardHead title="Tamamlanan LLM Deneyleri"/>
        <div className="table-wrap"><table><thead><tr><th>Model</th><th>Yöntem</th><th>Başlangıç</th><th>Nihai</th>
          <th>Tur</th><th>Geçerli Test</th><th>Token</th><th>Süre</th></tr></thead><tbody>
          {rows.map(row => <tr key={row.id}><td>{row.model} · {row.provider}</td><td>{feedbackLabels[row.feedback]}</td>
            <td>{percent(row.initial_score)}</td><td>{percent(row.final_score)}</td><td>{row.rounds}/{row.planned_rounds}</td>
            <td>{row.valid}/{row.generated}</td><td>{number(row.tokens, 0)}</td><td>{number(row.duration_seconds, 1)} sn</td></tr>)}
        </tbody></table></div>
      </Card>
    </> : <Card><CardHead title="Model Karşılaştırmaları"/>
      <p className="empty-state">{result.loading ? 'Sonuçlar yükleniyor…' : 'Tamamlanan LLM deneyi yok. Repo referans koşuları Test Merkezi’nde görüntülenir.'}</p>
    </Card>}
  </div>
}
