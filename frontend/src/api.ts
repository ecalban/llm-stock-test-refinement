import { useEffect, useState } from 'react'

export type Signal = 'buy' | 'hold' | 'sell'
export const signalLabels: Record<Signal, string> = { buy: 'AL', hold: 'TUT', sell: 'SAT' }
export const signalTone = (signal: Signal) => signal === 'buy' ? 'green' : signal === 'sell' ? 'red' : 'yellow'
export const number = (value: number | null | undefined, digits = 2) =>
  value == null ? '—' : new Intl.NumberFormat('tr-TR', { maximumFractionDigits: digits }).format(value)
export const percent = (value: number | null | undefined, digits = 1) =>
  value == null ? '—' : new Intl.NumberFormat('tr-TR', { style: 'percent', maximumFractionDigits: digits }).format(value)
export const money = (value: number) => new Intl.NumberFormat('tr-TR', {
  style: 'currency', currency: 'TRY', maximumFractionDigits: 2,
}).format(value)
export const dateLabel = (value?: string | null) =>
  value ? new Intl.DateTimeFormat('tr-TR', { dateStyle: 'medium', timeZone: 'Europe/Istanbul' })
    .format(new Date(value.length === 10 ? value + 'T12:00:00Z' : value)) : '—'
export const ticker = (symbol: string) => symbol.replace(/\.IS$/, '')

export interface Stock { symbol: string; name: string; kind: 'stock' | 'index'; first_date: string; as_of: string; rows: number }
export interface Model { id: string; name: string; available: boolean }
export type Strategy = 'ml' | 'buy_hold' | 'sma'
export const strategyLabels: Record<Strategy, string> = { ml: 'ML Sinyal Stratejisi', buy_hold: 'Al ve Tut', sma: 'SMA Kesişimi' }
export interface Capabilities { models: Model[]; strategies: Strategy[] }
export interface Quote { close: number; daily_change: number }
export interface SignalRow {
  date: string; action: Signal; probability: number; actual_class: Signal | null
  correct: boolean | null; close: number
}
export interface LatestSignal extends SignalRow { probabilities: Record<Signal, number> }
export interface Bar { date: string; close: number; rsi: number | null; macd: number | null; macd_signal: number | null; macd_histogram: number | null }
export interface Daily {
  date: string; valuation_date: string; equity: number; benchmark_equity: number
  drawdown: number; bist100_equity?: number
}
export interface Trade {
  entry_date: string; exit_date: string; entry_price: number; exit_price: number
  net_return: number; duration_sessions: number; total_cost: number
}
export interface Metrics {
  total_return: number; buy_hold_return: number; bist100_return: number | null
  sharpe_ratio: number; max_drawdown: number; closed_trade_count: number; order_count: number
  win_rate: number | null; recovery_sessions: number | null; total_cost: number
}
export interface RunConfig {
  evaluation_start: string; evaluation_end: string; initial_cash: number
  strategy: Strategy; commission_bps: number; slippage_bps: number
}
export interface BacktestData { config: RunConfig; metrics: Metrics; daily: Daily[]; trades: Trade[] }
export interface Backtest extends BacktestData { id: string; symbol: string }
export interface Analysis {
  id: string; symbol: string; as_of: string; quote: Quote; latest_prediction: LatestSignal
  series: Bar[]; signals: SignalRow[]; backtest: BacktestData
  classification_metrics: { accuracy: number | null; macro_f1: number | null; samples: number; pending: number }
}
export interface Dashboard {
  analysed_stock_count: number; available_stock_count: number; pooled_accuracy: number | null; accuracy_samples: number
  latest_signals: { symbol: string; as_of: string; analysis_id: string; quote: Quote; prediction: LatestSignal }[]
}

export type Feedback = 'score' | 'line' | 'code'
export const feedbackLabels: Record<Feedback, string> = { score: 'Yalnızca skor', line: 'Satır ve operatör', code: 'Tam mutant kodu' }
export type Provider = 'reference' | 'openai' | 'ollama'
export interface Options {
  targets: { id: string; name: string; llm: boolean }[]
  providers: { id: Provider; name: string; available: boolean; reason: string | null }[]
}
export interface Mutant {
  id: string; line: number; original: string; replacement: string
  diff: string; source: string; status: 'killed' | 'survived' | 'timeout' | 'error'; tests: string[]; detail: string
}
export interface Round {
  round: number; score: number; killed: number; survived: number; timeout: number; errors: number
  generated: number; valid: number; suite_size: number; mutants?: Mutant[]
  validation?: { name: string; passed: boolean; case?: Record<string, unknown> }[]
}
export interface Experiment {
  id: string; status: 'queued' | 'running' | 'failed' | 'completed'; phase: string; model: string
  config: { target: string; provider: Provider; feedback: Feedback | null; rounds: number; mutant_limit: number }
  round: number; selected_mutants?: number; mutants_done?: number; pool_size?: number
  rounds: Round[]; error: string | null; created_at: string; duration_seconds?: number; catalog?: string
}
export interface Comparison {
  id: string; model: string; provider: Provider; feedback: Feedback; target: string; catalog: string; source_hash: string
  initial_score: number; final_score: number; delta: number; rounds: number; planned_rounds: number; mutants: number
  generated: number; valid: number; tokens: number; duration_seconds: number
}

export async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch('/api' + path, options)
  if (!response.ok) {
    const error = await response.json().catch(() => null)
    const detail = error?.detail
    throw new Error(typeof detail === 'string' ? detail :
      Array.isArray(detail) ? detail.map(item => item.msg).join(' · ') : `Sunucu yanıtı: ${response.status}`)
  }
  return response.json() as Promise<T>
}
export const post = <T,>(path: string, body: unknown) => api<T>(path, {
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
})
export const errorText = (error: unknown) => error instanceof Error && error.message !== 'Failed to fetch'
  ? error.message : 'API’ye bağlanılamadı. Yerel API sunucusunun açık olduğunu kontrol et.'

export function useResource<T>(path: string | null, version = 0) {
  const [result, setResult] = useState<{ key: string; data: T | null; error: string | null }>()
  const key = path + ':' + version
  useEffect(() => {
    if (!path) return
    const controller = new AbortController()
    api<T>(path, { signal: controller.signal }).then(
      data => { if (!controller.signal.aborted) setResult({ key, data, error: null }) },
      error => { if (!controller.signal.aborted) setResult({ key, data: null, error: errorText(error) }) },
    )
    return () => controller.abort()
  }, [path, key])
  return result?.key === key ? { ...result, loading: false } : { data: null, error: null, loading: !!path }
}

export async function download(path: string, name: string) {
  const response = await fetch('/api' + path)
  if (!response.ok) throw new Error('Dosya indirilemedi.')
  const url = URL.createObjectURL(await response.blob())
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = name
  anchor.click()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
