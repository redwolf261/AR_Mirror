import { useEffect, useRef, useState, useCallback, useMemo } from 'react'

// ─── Types ───────────────────────────────────────────────────────────────────

interface Quality {
  collar_score: number; width_score: number; height_score: number
  face_clear: number; face_pixel_score?: number; collar_pixel_score?: number
  coverage_score?: number; geo_total?: number; total: number
  diagnostics?: {
    collar_err_px: number; width_ratio: number; height_ratio: number
    overlap_px: number; sh_y: number; placed_top: number
    sh_span: number; placed_w: number; torso_h: number; placed_h: number
  }
  px_diag?: { skin_at_face: number; skin_at_collar: number; skin_at_mid: number }
}
interface Measurements {
  shoulder_cm: number | null; chest_cm: number | null
  waist_cm: number | null; torso_cm: number | null; size: string | null
}
interface State {
  fps: number; garment: string; ts: number
  measurements?: Measurements; torso_box?: number[]
  quality?: Quality; quality_smooth?: number; auto_locked?: boolean
}
interface ParamMeta {
  type: 'bool' | 'float' | 'int'; label: string; min?: number; max?: number; step?: number
}
interface ParamsResponse { values: Record<string, unknown>; meta: Record<string, ParamMeta> }

// ─── Helpers ─────────────────────────────────────────────────────────────────

async function fetchJson<T>(url: string): Promise<T> {
  const r = await fetch(url)
  if (!r.ok) throw new Error(String(r.status))
  return r.json()
}
async function postJson<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  return r.json()
}
function neonColor(v: number) {
  return v >= 0.8 ? '#00ff88' : v >= 0.55 ? '#ffcc00' : '#ff3355'
}
function neonShadow(v: number, intense = false) {
  const c = neonColor(v)
  const b = intense ? '12px' : '6px'
  const b2 = intense ? '24px' : '12px'
  return `0 0 ${b} ${c}, 0 0 ${b2} ${c}40`
}
const MAX_HIST = 120

// ─── Global CSS ──────────────────────────────────────────────────────────────

const GLOBAL_CSS = `
  @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Share+Tech+Mono&family=Rajdhani:wght@300;400;600&display=swap');
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html, body, #root { height: 100%; background: #030308; }
  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-track { background: #06060f; }
  ::-webkit-scrollbar-thumb { background: #00f5ff30; border-radius: 2px; }
  ::-webkit-scrollbar-thumb:hover { background: #00f5ff80; }

  @keyframes flicker {
    0%, 94%, 100% { opacity: 1; }
    95% { opacity: 0.93; }
    97% { opacity: 0.97; }
  }
  @keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.35; transform: scale(0.65); }
  }
  @keyframes corner-pulse {
    0%, 100% { opacity: 0.55; }
    50% { opacity: 1; }
  }
  @keyframes slide-in-right {
    from { opacity: 0; transform: translateX(18px); }
    to   { opacity: 1; transform: translateX(0); }
  }
  @keyframes grid-reveal {
    from { opacity: 0; transform: scale(0.95) translateY(6px); }
    to   { opacity: 1; transform: scale(1) translateY(0); }
  }
  @keyframes bar-glow {
    0%, 100% { box-shadow: 0 0 4px currentColor; }
    50%       { box-shadow: 0 0 12px currentColor, 0 0 24px currentColor; }
  }

  .garment-card { transition: transform 0.15s ease, border-color 0.15s ease, box-shadow 0.15s ease; cursor: pointer; }
  .garment-card:hover {
    transform: translateY(-4px) scale(1.03);
    border-color: #00f5ff90 !important;
    box-shadow: 0 0 24px #00f5ff40, 0 8px 28px #00000095 !important;
  }
  .garment-card.active {
    border-color: #00f5ff !important;
    box-shadow: 0 0 28px #00f5ff65, 0 0 56px #00f5ff20, inset 0 0 18px #00f5ff14 !important;
  }
  .tab-btn {
    background: transparent; border: none; cursor: pointer; padding: 10px 16px;
    font-family: 'Rajdhani', sans-serif; font-size: 10px; letter-spacing: 2px;
    font-weight: 600; text-transform: uppercase; transition: color 0.15s;
    border-bottom: 2px solid transparent;
  }
  .tab-btn.active  { color: #00f5ff; border-bottom-color: #00f5ff; text-shadow: 0 0 10px #00f5ff; }
  .tab-btn:not(.active) { color: #1e2d3d; }
  .tab-btn:not(.active):hover { color: #00f5ff60; }
`

// ─── CornerBrackets ───────────────────────────────────────────────────────────

function CornerBrackets({
  color = '#00f5ff', size = 20, thickness = 2, animated = false,
}: { color?: string; size?: number; thickness?: number; animated?: boolean }) {
  const s = `${size}px`, b = `${thickness}px solid ${color}`
  const c: React.CSSProperties = { position: 'absolute', width: s, height: s }
  return (
    <div style={{
      position: 'absolute', inset: 0, pointerEvents: 'none', zIndex: 10,
      animation: animated ? 'corner-pulse 2.2s ease-in-out infinite' : undefined,
    }}>
      <div style={{ ...c, top: 0, left: 0, borderTop: b, borderLeft: b }} />
      <div style={{ ...c, top: 0, right: 0, borderTop: b, borderRight: b }} />
      <div style={{ ...c, bottom: 0, left: 0, borderBottom: b, borderLeft: b }} />
      <div style={{ ...c, bottom: 0, right: 0, borderBottom: b, borderRight: b }} />
    </div>
  )
}

// ─── NeonBar ─────────────────────────────────────────────────────────────────

function NeonBar({
  label, value, detail, color,
}: { label: string; value: number; detail?: string; color?: string }) {
  const pct = Math.round(value * 100)
  const c = color ?? neonColor(value)
  return (
    <div style={{ marginBottom: 11 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{
          fontFamily: "'Rajdhani', sans-serif", fontSize: 10,
          color: '#44596a', letterSpacing: 1, textTransform: 'uppercase',
        }}>{label}</span>
        <span style={{
          fontFamily: "'Share Tech Mono', monospace", fontSize: 10,
          color: c, textShadow: `0 0 7px ${c}`,
        }}>
          {pct}%
          {detail && <span style={{ color: '#2a3a44', fontSize: 9 }}> · {detail}</span>}
        </span>
      </div>
      <div style={{ height: 4, background: '#081218', borderRadius: 2, overflow: 'hidden', position: 'relative' }}>
        <div style={{
          width: `${pct}%`, height: '100%',
          background: `linear-gradient(90deg, ${c}55, ${c})`,
          boxShadow: `0 0 9px ${c}`,
          transition: 'width 0.45s cubic-bezier(0.4,0,0.2,1)',
          borderRadius: 2,
        }} />
      </div>
    </div>
  )
}

// ─── FPS Ring ────────────────────────────────────────────────────────────────

function FpsRing({ fps }: { fps: number }) {
  const max = 30, pct = Math.min(fps / max, 1)
  const r = 40, stroke = 5, S = 100
  const circ = 2 * Math.PI * r
  const dash = circ * pct
  const c = fps >= 15 ? '#00ff88' : fps >= 8 ? '#ffcc00' : '#ff3355'
  return (
    <div style={{ position: 'relative', width: S, height: S }}>
      <svg width={S} height={S} style={{ transform: 'rotate(-90deg)' }}>
        <circle cx={S/2} cy={S/2} r={r} fill="none" stroke="#0a1a16" strokeWidth={stroke} />
        <circle cx={S/2} cy={S/2} r={r} fill="none" stroke={c} strokeWidth={stroke}
          strokeDasharray={`${dash} ${circ - dash}`} strokeLinecap="round"
          style={{
            filter: `drop-shadow(0 0 7px ${c}) drop-shadow(0 0 14px ${c}60)`,
            transition: 'all 0.4s ease',
          }} />
      </svg>
      <div style={{
        position: 'absolute', inset: 0,
        display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
      }}>
        <span style={{
          fontFamily: "'Orbitron', monospace", fontSize: 21, fontWeight: 900,
          color: c, textShadow: `0 0 16px ${c}`, lineHeight: 1,
        }}>{fps.toFixed(0)}</span>
        <span style={{
          fontFamily: "'Share Tech Mono', monospace", fontSize: 7,
          color: '#2a3a44', letterSpacing: 2,
        }}>FPS</span>
      </div>
    </div>
  )
}

// ─── Sparkline ───────────────────────────────────────────────────────────────

function Sparkline({ history }: { history: number[] }) {
  const W = 196, H = 32
  if (history.length < 2) {
    return <svg width={W} height={H}><line x1="0" y1={H / 2} x2={W} y2={H / 2} stroke="#0c1820" strokeWidth="1" /></svg>
  }
  const pts = history.map((v, i) =>
    `${((i / (history.length - 1)) * W).toFixed(1)},${(H - v * (H - 6) - 3).toFixed(1)}`
  ).join(' ')
  const last = history[history.length - 1]
  const c = neonColor(last)
  const lx = W, ly = H - last * (H - 6) - 3
  const fillPath = `M ${pts.replace(/ /g, ' L ')} L ${W} ${H} L 0 ${H} Z`
  return (
    <svg width={W} height={H} style={{ display: 'block', overflow: 'visible' }}>
      <defs>
        <linearGradient id="sfill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={c} stopOpacity="0.28" />
          <stop offset="100%" stopColor={c} stopOpacity="0.01" />
        </linearGradient>
      </defs>
      <path d={fillPath} fill="url(#sfill)" />
      <polyline points={pts} fill="none" stroke={c} strokeWidth="1.6" strokeLinejoin="round"
        style={{ filter: `drop-shadow(0 0 3px ${c})` }} />
      <circle cx={lx} cy={ly} r="3.5" fill={c}
        style={{ filter: `drop-shadow(0 0 5px ${c})` }} />
    </svg>
  )
}

// ─── MeasureBadge ────────────────────────────────────────────────────────────

function MeasureBadge({ label, value }: { label: string; value: number | null }) {
  return (
    <div style={{
      flex: 1, textAlign: 'center', padding: '8px 6px',
      background: 'linear-gradient(180deg, #070d18 0%, #04080f 100%)',
      border: '1px solid #0e2030', borderRadius: 6,
    }}>
      <div style={{
        fontFamily: "'Rajdhani', sans-serif", fontSize: 8, color: '#2a3a44',
        letterSpacing: 2, textTransform: 'uppercase', marginBottom: 6,
      }}>{label}</div>
      <div style={{
        fontFamily: "'Orbitron', monospace", fontSize: 20, fontWeight: 900,
        color: value !== null ? '#00f5ff' : '#1a2530',
        textShadow: value !== null ? '0 0 14px #00f5ff70' : 'none',
        lineHeight: 1,
      }}>
        {value != null ? value.toFixed(1) : '--'}
      </div>
      <div style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: 7, color: '#1a2a30', marginTop: 3, letterSpacing: 1 }}>cm</div>
    </div>
  )
}

// ─── ParamRow ────────────────────────────────────────────────────────────────

function ParamRow({
  name, value, meta, onChange,
}: { name: string; value: unknown; meta: ParamMeta; onChange: (n: string, v: unknown) => void }) {
  if (meta.type === 'bool') {
    const checked = !!value
    return (
      <label style={{ display: 'flex', alignItems: 'center', gap: 10, cursor: 'pointer', padding: '5px 0' }}>
        <div
          onClick={() => onChange(name, !checked)}
          style={{
            width: 28, height: 16, borderRadius: 8, position: 'relative', flexShrink: 0,
            background: checked ? '#00f5ff' : '#0c1820',
            border: `1px solid ${checked ? '#00f5ff' : '#182830'}`,
            boxShadow: checked ? '0 0 9px #00f5ff70' : 'none',
            transition: 'all 0.18s', cursor: 'pointer',
          }}
        >
          <div style={{
            position: 'absolute', top: 2, left: checked ? 13 : 2, width: 10, height: 10,
            borderRadius: '50%', background: checked ? '#001820' : '#2a3a44',
            transition: 'left 0.18s',
          }} />
        </div>
        <span style={{ fontFamily: "'Rajdhani', sans-serif", fontSize: 11, color: '#5a6878' }}>
          {meta.label}
        </span>
      </label>
    )
  }
  const numVal = Number(value)
  const min = meta.min ?? 0, max = meta.max ?? 100, step = meta.step ?? 1
  const pct = ((numVal - min) / (max - min)) * 100
  return (
    <div style={{ padding: '5px 0' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ fontFamily: "'Rajdhani', sans-serif", fontSize: 10, color: '#4a5868' }}>
          {meta.label}
        </span>
        <span style={{
          fontFamily: "'Share Tech Mono', monospace", fontSize: 9,
          color: '#00f5ff', minWidth: 36, textAlign: 'right',
        }}>
          {numVal.toFixed(step < 1 ? 2 : 0)}
        </span>
      </div>
      <div style={{ position: 'relative', height: 4, background: '#081218', borderRadius: 2 }}>
        <div style={{
          position: 'absolute', left: 0, top: 0, width: `${pct}%`, height: '100%',
          background: 'linear-gradient(90deg, #00f5ff44, #00f5ff)',
          borderRadius: 2, boxShadow: '0 0 7px #00f5ff70',
        }} />
        <input
          type="range" min={min} max={max} step={step} value={numVal}
          onChange={e => onChange(name, meta.type === 'int' ? parseInt(e.target.value) : parseFloat(e.target.value))}
          style={{ position: 'absolute', inset: 0, opacity: 0, cursor: 'pointer', width: '100%', height: '100%' }}
        />
      </div>
    </div>
  )
}

// ─── GarmentCard ─────────────────────────────────────────────────────────────

function GarmentCard({
  name, active, onClick, idx,
}: { name: string; active: boolean; onClick: () => void; idx: number }) {
  const [imgError, setImgError] = useState(false)
  const label = name.replace(/\.[^/.]+$/, '').replace(/_/g, ' ')
  const shortId = String(idx + 1).padStart(3, '0')
  return (
    <div
      className={`garment-card${active ? ' active' : ''}`}
      onClick={onClick}
      title={label}
      style={{
        position: 'relative', borderRadius: 8, overflow: 'hidden',
        background: '#060c14',
        border: active ? '1px solid #00f5ff' : '1px solid #0a1820',
        boxShadow: active ? '0 0 24px #00f5ff60' : '0 2px 10px #00000090',
        animation: 'grid-reveal 0.25s ease both',
        animationDelay: `${Math.min(idx * 0.015, 0.4)}s`,
        aspectRatio: '3/4',
      }}
    >
      {!imgError ? (
        <img
          src={`/api/garment_image/${encodeURIComponent(name)}`}
          alt={label}
          onError={() => setImgError(true)}
          style={{
            width: '100%', height: '100%', objectFit: 'cover', display: 'block',
            filter: active ? 'brightness(1.08) saturate(1.1)' : 'brightness(0.72) saturate(0.85)',
            transition: 'filter 0.2s',
          }}
        />
      ) : (
        <div style={{
          width: '100%', height: '100%',
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 6,
          background: 'linear-gradient(135deg, #060c14, #030810)',
        }}>
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#152030" strokeWidth="1.2">
            <path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3l-2 3M15 3l2 3" />
          </svg>
          <span style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: 7, color: '#1a3040', letterSpacing: 1 }}>
            #{shortId}
          </span>
        </div>
      )}
      {/* Gradient overlay bottom */}
      <div style={{
        position: 'absolute', bottom: 0, left: 0, right: 0, height: '45%',
        background: 'linear-gradient(0deg, rgba(3,3,8,0.92) 0%, rgba(3,3,8,0.5) 50%, transparent 100%)',
        pointerEvents: 'none',
      }} />
      {/* Active shimmer */}
      {active && (
        <div style={{
          position: 'absolute', inset: 0,
          background: 'linear-gradient(135deg, #00f5ff0c 0%, transparent 60%)',
          pointerEvents: 'none',
        }} />
      )}
      {/* ID badge top-left */}
      <div style={{
        position: 'absolute', top: 5, left: 5,
        fontFamily: "'Share Tech Mono', monospace", fontSize: 7,
        color: active ? '#00f5ffcc' : '#1e3040',
        textShadow: active ? '0 0 8px #00f5ff' : 'none', letterSpacing: 1,
        background: 'rgba(3,3,8,0.7)', padding: '1px 4px', borderRadius: 2,
      }}>
        #{shortId}
      </div>
      {/* Active tick */}
      {active && (
        <div style={{
          position: 'absolute', top: 5, right: 5, width: 16, height: 16,
          borderRadius: '50%', background: '#00f5ff',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          boxShadow: '0 0 12px #00f5ff',
        }}>
          <svg width="9" height="9" viewBox="0 0 8 8" fill="none">
            <path d="M1 4l2 2 4-4" stroke="#030308" strokeWidth="1.8" strokeLinecap="round" />
          </svg>
        </div>
      )}
      {/* Label bottom */}
      <div style={{
        position: 'absolute', bottom: 0, left: 0, right: 0, padding: '5px 7px',
      }}>
        <div style={{
          fontFamily: "'Rajdhani', sans-serif", fontSize: 8, letterSpacing: 1.2,
          color: active ? '#00f5ff' : '#5a7080',
          textShadow: active ? '0 0 8px #00f5ff60' : 'none',
          overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          textTransform: 'uppercase', fontWeight: 600,
        }}>
          {label}
        </div>
      </div>
    </div>
  )
}

// ─── App ─────────────────────────────────────────────────────────────────────

export default function App() {
  const [state, setState]           = useState<State | null>(null)
  const [params, setParams]         = useState<ParamsResponse | null>(null)
  const [garments, setGarments]     = useState<string[]>([])
  const [pending, setPending]       = useState<Record<string, unknown>>({})
  const [connected, setConnected]   = useState(false)
  const [scoreHist, setScoreHist]   = useState<number[]>([])
  const [activeTab, setActiveTab]   = useState<'quality' | 'params' | 'diag'>('quality')
  const [searchQuery, setSearchQuery] = useState('')
  const timerRef = useRef<number>(0)

  useEffect(() => {
    const el = document.createElement('style')
    el.textContent = GLOBAL_CSS
    document.head.appendChild(el)
    document.title = 'AR MIRROR // NEURAL TRY-ON'
    return () => el.remove()
  }, [])

  useEffect(() => {
    let alive = true
    const poll = async () => {
      try {
        const s = await fetchJson<State>('/api/state')
        if (!alive) return
        setState(s); setConnected(true)
        if (s.quality) setScoreHist(h => {
          const n = [...h, s.quality!.total]
          return n.length > MAX_HIST ? n.slice(-MAX_HIST) : n
        })
      } catch { if (alive) setConnected(false) }
    }
    poll()
    const id = setInterval(poll, 250)
    return () => { alive = false; clearInterval(id) }
  }, [])

  useEffect(() => {
    fetchJson<ParamsResponse>('/api/params').then(setParams).catch(console.error)
    fetchJson<{ garments: string[] }>('/api/garments').then(r => setGarments(r.garments)).catch(console.error)
  }, [])

  useEffect(() => {
    const id = setInterval(() =>
      fetchJson<{ garments: string[] }>('/api/garments')
        .then(r => setGarments(r.garments)).catch(() => {}), 10_000)
    return () => clearInterval(id)
  }, [])

  const handleParam = useCallback((name: string, v: unknown) => {
    setParams(prev => prev ? { ...prev, values: { ...prev.values, [name]: v } } : prev)
    setPending(prev => ({ ...prev, [name]: v }))
  }, [])

  useEffect(() => {
    if (!Object.keys(pending).length) return
    clearTimeout(timerRef.current)
    timerRef.current = setTimeout(async () => {
      await postJson('/api/params', pending); setPending({})
    }, 150) as unknown as number
  }, [pending])

  const switchGarment = useCallback((name: string) => {
    postJson('/api/garment', { name }).catch(console.error)
  }, [])

  const filteredGarments = useMemo(() =>
    searchQuery
      ? garments.filter(g => g.toLowerCase().includes(searchQuery.toLowerCase()))
      : garments,
    [garments, searchQuery])

  const q = state?.quality, d = q?.diagnostics
  const qs = state?.quality_smooth ?? q?.total ?? 0
  const locked = state?.auto_locked ?? false
  const meas = state?.measurements, fps = state?.fps ?? 0

  return (
    <div style={{
      display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden',
      background: '#030308', fontFamily: "'Rajdhani', sans-serif", color: '#c0ccd5',
    }}>

      {/* ── Header ──────────────────────────────────────────────────────────── */}
      <div style={{
        height: 58, flexShrink: 0,
        display: 'flex', alignItems: 'center', padding: '0 26px',
        background: 'linear-gradient(90deg, #030308 0%, #050822 40%, #060418 70%, #030308 100%)',
        borderBottom: '1px solid #00f5ff20',
        boxShadow: '0 1px 40px #00f5ff12, 0 2px 80px #00000080',
        position: 'relative', zIndex: 100,
      }}>

        {/* Logo */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            <div style={{ width: 3, height: 11, background: '#00f5ff', boxShadow: '0 0 9px #00f5ff', borderRadius: 1 }} />
            <div style={{ width: 3, height: 6, background: '#b400ff55', borderRadius: 1 }} />
          </div>
          <span style={{
            fontFamily: "'Orbitron', monospace", fontSize: 18, fontWeight: 900, letterSpacing: 6,
            color: '#00f5ff', textShadow: '0 0 20px #00f5ff, 0 0 60px #00f5ff45',
          }}>AR MIRROR</span>
          <span style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: 8, color: '#00f5ff30', letterSpacing: 2 }}>
            v2.0 // NEURAL TPS
          </span>
        </div>

        {/* Status pills */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 14, marginLeft: 30 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <div style={{
              width: 6, height: 6, borderRadius: '50%',
              background: connected ? '#00ff88' : '#ff3355',
              boxShadow: connected ? '0 0 9px #00ff88' : '0 0 9px #ff3355',
              animation: 'pulse-dot 1.5s ease-in-out infinite',
            }} />
            <span style={{
              fontFamily: "'Share Tech Mono', monospace", fontSize: 8, letterSpacing: 1,
              color: connected ? '#00ff8868' : '#ff335568',
            }}>{connected ? 'LIVE' : 'OFFLINE'}</span>
          </div>
          <span style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: 7, color: '#1a2a34', letterSpacing: 1 }}>
            RTX-2050 | CUDA | TPS | RVM | ESRGAN
          </span>
        </div>

        {/* Right side */}
        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 14 }}>
          {state?.garment && (
            <span style={{
              fontFamily: "'Share Tech Mono', monospace", fontSize: 8, color: '#00f5ff50', letterSpacing: 1,
              maxWidth: 260, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
            }}>
              &rsaquo; {state.garment.replace(/\.[^/.]+$/, '').replace(/_/g, ' ').toUpperCase()}
            </span>
          )}
          {meas?.size && (
            <div style={{
              padding: '2px 12px', borderRadius: 3,
              background: '#00897b18', border: '1px solid #00897b50',
              fontFamily: "'Orbitron', monospace", fontSize: 9, fontWeight: 700,
              color: '#00ff88', letterSpacing: 2, boxShadow: '0 0 9px #00ff8825',
            }}>
              SIZE {meas.size}
            </div>
          )}
          <div style={{
            padding: '3px 14px', borderRadius: 3,
            background: locked ? '#00ff8812' : '#00f5ff0a',
            border: `1px solid ${locked ? '#00ff8845' : '#00f5ff28'}`,
            fontFamily: "'Orbitron', monospace", fontSize: 8, fontWeight: 700, letterSpacing: 1.2,
            color: locked ? '#00ff88' : '#00f5ff',
            boxShadow: locked ? '0 0 12px #00ff8830' : '0 0 9px #00f5ff18',
          }}>
            {locked ? '\u2714 PERFECT FIT' : `\u25cf CALIBRATING ${Math.round(qs * 100)}%`}
          </div>
        </div>
      </div>

      {/* ── Main area ───────────────────────────────────────────────────────── */}
      <div style={{ flex: 1, display: 'flex', overflow: 'hidden', minHeight: 0 }}>

        {/* ── Col 1: Stream ─────────────────────────────────────────────────── */}
        <div style={{
          flex: '1 1 0', minWidth: 0, display: 'flex', flexDirection: 'column',
          background: '#030308', borderRight: '1px solid #00f5ff0d',
        }}>
          <div style={{ flex: 1, position: 'relative', overflow: 'hidden', background: '#000' }}>
            <img
              src="/stream" alt="live"
              style={{
                width: '100%', height: '100%', objectFit: 'cover', display: 'block',
                filter: connected ? 'none' : 'grayscale(1) brightness(0.2)',
                animation: connected ? 'flicker 9s ease-in-out infinite' : 'none',
              }}
            />
            {/* Scanlines */}
            <div style={{
              position: 'absolute', inset: 0, pointerEvents: 'none', zIndex: 5,
              background: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,245,255,0.011) 2px, rgba(0,245,255,0.011) 4px)',
            }} />
            <CornerBrackets color="#00f5ff" size={26} thickness={2} animated />
            {/* FPS ring */}
            <div style={{ position: 'absolute', top: 18, left: 18, zIndex: 20 }}>
              <FpsRing fps={fps} />
            </div>
            {/* Fit score overlay */}
            {q && (
              <div style={{
                position: 'absolute', bottom: 14, left: 14, zIndex: 20,
                background: 'rgba(3,3,8,0.88)', border: '1px solid #00f5ff1a',
                borderRadius: 5, padding: '7px 13px', backdropFilter: 'blur(8px)',
              }}>
                <div style={{
                  fontFamily: "'Share Tech Mono', monospace", fontSize: 7,
                  color: '#2a3a44', letterSpacing: 2, marginBottom: 2,
                }}>FIT SCORE</div>
                <div style={{
                  fontFamily: "'Orbitron', monospace", fontSize: 32, fontWeight: 900,
                  color: neonColor(qs), textShadow: neonShadow(qs, true), lineHeight: 1,
                }}>
                  {Math.round(qs * 100)}
                  <span style={{ fontSize: 13, color: '#2a3a44', fontWeight: 400 }}>%</span>
                </div>
              </div>
            )}
            {/* Offline overlay */}
            {!connected && (
              <div style={{
                position: 'absolute', inset: 0, zIndex: 30,
                display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
                background: 'rgba(3,3,8,0.93)',
              }}>
                <CornerBrackets color="#ff3355" size={40} thickness={2} animated />
                <div style={{
                  fontFamily: "'Orbitron', monospace", fontSize: 13, fontWeight: 700,
                  color: '#ff3355', letterSpacing: 5, textShadow: '0 0 22px #ff3355',
                  animation: 'flicker 3s ease-in-out infinite', marginBottom: 12,
                }}>SIGNAL LOST</div>
                <div style={{
                  fontFamily: "'Share Tech Mono', monospace", fontSize: 8,
                  color: '#2a3a44', letterSpacing: 2,
                }}>RUN: python app.py --phase 2 --duration 300</div>
              </div>
            )}
          </div>
          {/* Measurements strip */}
          {meas && (
            <div style={{
              height: 80, flexShrink: 0, display: 'flex', gap: 5, padding: '6px 10px',
              background: 'linear-gradient(0deg, #030308 0%, #04080f 100%)',
              borderTop: '1px solid #00f5ff14',
            }}>
              {([
                ['SHOULDER', meas.shoulder_cm],
                ['CHEST',    meas.chest_cm],
                ['WAIST',    meas.waist_cm],
                ['TORSO',    meas.torso_cm],
              ] as [string, number | null][]).map(([l, v]) => (
                <MeasureBadge key={l} label={l} value={v} />
              ))}
            </div>
          )}
        </div>

        {/* ── Col 2: Metrics ────────────────────────────────────────────────── */}
        <div style={{
          width: 256, flexShrink: 0, display: 'flex', flexDirection: 'column',
          background: '#030510', borderRight: '1px solid #00f5ff0a', overflow: 'hidden',
        }}>
          {/* Tabs */}
          <div style={{ display: 'flex', borderBottom: '1px solid #0c1824', background: '#03050f', flexShrink: 0 }}>
            {(['quality', 'params', 'diag'] as const).map(t => (
              <button
                key={t}
                className={`tab-btn${activeTab === t ? ' active' : ''}`}
                onClick={() => setActiveTab(t)}
              >
                {t === 'quality' ? 'QUALITY' : t === 'params' ? 'PARAMS' : 'DIAG'}
              </button>
            ))}
          </div>
          <div style={{ flex: 1, overflowY: 'auto', padding: '12px 14px' }}>

            {/* QUALITY */}
            {activeTab === 'quality' && (
              <div style={{ animation: 'slide-in-right 0.2s ease' }}>
                <div style={{
                  display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16,
                  background: '#060c14', border: '1px solid #0c1e2c', borderRadius: 6, padding: '10px 12px',
                }}>
                  <div style={{ flex: 1 }}>
                    <div style={{
                      fontFamily: "'Share Tech Mono', monospace", fontSize: 7,
                      color: '#2a3a44', letterSpacing: 2, marginBottom: 5,
                    }}>FIT HISTORY</div>
                    <Sparkline history={scoreHist} />
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{
                      fontFamily: "'Orbitron', monospace", fontSize: 30, fontWeight: 900,
                      color: neonColor(qs), textShadow: neonShadow(qs, true), lineHeight: 1,
                    }}>{Math.round(qs * 100)}</div>
                    <div style={{
                      fontFamily: "'Share Tech Mono', monospace", fontSize: 7,
                      color: '#2a3a44', letterSpacing: 1, marginTop: 2,
                    }}>/ 100</div>
                  </div>
                </div>

                {q ? (
                  <>
                    {q.face_pixel_score !== undefined && (
                      <NeonBar label="Face Visible" value={q.face_pixel_score}
                        detail={q.px_diag ? `${(q.px_diag.skin_at_face * 100).toFixed(0)}% skin` : undefined} />
                    )}
                    {q.collar_pixel_score !== undefined && (
                      <NeonBar label="Collar on Shoulder" value={q.collar_pixel_score}
                        detail={q.px_diag ? `${(q.px_diag.skin_at_collar * 100).toFixed(0)}% skin` : undefined} />
                    )}
                    {q.coverage_score !== undefined && (
                      <NeonBar label="Torso Coverage" value={q.coverage_score}
                        detail={q.px_diag ? `${(q.px_diag.skin_at_mid * 100).toFixed(0)}% skin` : undefined} />
                    )}
                    <div style={{ height: 1, background: '#091520', margin: '10px 0' }} />
                    <div style={{
                      fontFamily: "'Share Tech Mono', monospace", fontSize: 7,
                      color: '#1a2a34', letterSpacing: 2, marginBottom: 8,
                    }}>GEOMETRIC ANALYSIS</div>
                    <NeonBar label="Collar Alignment" value={q.collar_score}
                      detail={d ? `${d.collar_err_px > 0 ? '+' : ''}${d.collar_err_px.toFixed(0)}px` : undefined} />
                    <NeonBar label="Width Fit"  value={q.width_score}
                      detail={d ? `${(d.width_ratio * 100).toFixed(0)}%` : undefined} />
                    <NeonBar label="Height Fit" value={q.height_score}
                      detail={d ? `${(d.height_ratio * 100).toFixed(0)}%` : undefined} />
                    <NeonBar label="Face Clearance" value={q.face_clear}
                      detail={d && d.overlap_px > 0 ? `+${d.overlap_px.toFixed(0)}px` : undefined} />
                  </>
                ) : (
                  <div style={{
                    textAlign: 'center', padding: '30px 10px',
                    fontFamily: "'Share Tech Mono', monospace", fontSize: 8,
                    color: '#182430', letterSpacing: 2, lineHeight: 2.8,
                  }}>AWAITING<br />BODY DETECTION</div>
                )}
              </div>
            )}

            {/* PARAMS */}
            {activeTab === 'params' && (
              <div style={{ animation: 'slide-in-right 0.2s ease' }}>
                <div style={{
                  fontFamily: "'Share Tech Mono', monospace", fontSize: 7,
                  color: '#1a2a34', letterSpacing: 2, marginBottom: 12,
                }}>RUNTIME PARAMS // DRAG TO OVERRIDE</div>
                {params
                  ? Object.entries(params.meta).map(([k, m]) => (
                    <ParamRow key={k} name={k} value={params.values[k]} meta={m} onChange={handleParam} />
                  ))
                  : <div style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: 8, color: '#1a2a34', letterSpacing: 2 }}>LOADING...</div>}
              </div>
            )}

            {/* DIAG */}
            {activeTab === 'diag' && (
              <div style={{ animation: 'slide-in-right 0.2s ease' }}>
                <div style={{
                  fontFamily: "'Share Tech Mono', monospace", fontSize: 7,
                  color: '#1a2a34', letterSpacing: 2, marginBottom: 12,
                }}>DIAGNOSTIC READOUT</div>
                {d ? (
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '4px 6px' }}>
                    {([
                      ['sh_y',       d.sh_y],
                      ['placed_top', d.placed_top],
                      ['sh_span',    d.sh_span],
                      ['placed_w',   d.placed_w],
                      ['torso_h',    d.torso_h],
                      ['placed_h',   d.placed_h],
                    ] as [string, number][]).map(([k, v]) => (
                      <div key={k} style={{
                        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                        padding: '4px 6px', background: '#060c14', border: '1px solid #0c1824', borderRadius: 3,
                      }}>
                        <span style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: 8, color: '#2a3a44' }}>{k}</span>
                        <span style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: 8, color: '#00f5ff55' }}>
                          {v.toFixed(0)}<span style={{ color: '#1a2a34' }}>px</span>
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: 8, color: '#182430', letterSpacing: 2 }}>
                    NO DATA YET
                  </div>
                )}
                {state && (
                  <div style={{ marginTop: 14 }}>
                    <div style={{
                      fontFamily: "'Share Tech Mono', monospace", fontSize: 7,
                      color: '#1a2a34', letterSpacing: 2, marginBottom: 6,
                    }}>LIVE STATE</div>
                    {([
                      ['fps',     `${fps.toFixed(1)} Hz`],
                      ['ts',      new Date((state.ts || 0) * 1000).toISOString().slice(11, 19)],
                      ['garment', (state.garment || 'none').slice(0, 22)],
                    ] as [string, string][]).map(([k, v]) => (
                      <div key={k} style={{
                        display: 'flex', justifyContent: 'space-between',
                        padding: '3px 0', borderBottom: '1px solid #06090f',
                      }}>
                        <span style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: 8, color: '#2a3a44' }}>{k}</span>
                        <span style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: 8, color: '#3a4c58' }}>{v}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* ── Col 3: Wardrobe ───────────────────────────────────────────────── */}
        <div style={{
          width: 370, flexShrink: 0, display: 'flex', flexDirection: 'column',
          background: '#030308', overflow: 'hidden',
        }}>
          {/* Wardrobe header */}
          <div style={{
            flexShrink: 0, padding: '10px 12px 8px',
            background: 'linear-gradient(180deg, #040614 0%, #030308 100%)',
            borderBottom: '1px solid #b400ff18',
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <div style={{ width: 3, height: 18, background: '#b400ff', boxShadow: '0 0 12px #b400ff', borderRadius: 1 }} />
                <span style={{
                  fontFamily: "'Orbitron', monospace", fontSize: 9, fontWeight: 700,
                  color: '#b400ff', letterSpacing: 4,
                  textShadow: '0 0 12px #b400ff65',
                }}>WARDROBE</span>
              </div>
              <div style={{ fontFamily: "'Share Tech Mono', monospace", fontSize: 8, color: '#2a3a44', letterSpacing: 1 }}>
                {filteredGarments.length}
                <span style={{ color: '#1a2a34' }}>/{garments.length}</span>
              </div>
            </div>
            {/* Search */}
            <div style={{ position: 'relative' }}>
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="#2a3a44" strokeWidth="2"
                style={{ position: 'absolute', left: 8, top: '50%', transform: 'translateY(-50%)' }}>
                <circle cx="11" cy="11" r="8" /><path d="m21 21-4.35-4.35" />
              </svg>
              <input
                type="text"
                placeholder="FILTER..."
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                style={{
                  width: '100%',
                  background: '#060c14', border: '1px solid #0c1824',
                  borderRadius: 4, padding: '5px 8px 5px 24px',
                  fontFamily: "'Share Tech Mono', monospace", fontSize: 8,
                  color: '#5a7080', letterSpacing: 1, outline: 'none',
                  caretColor: '#00f5ff',
                }}
              />
            </div>
          </div>

          {/* Active garment strip */}
          {state?.garment && (
            <div style={{
              flexShrink: 0, padding: '5px 12px',
              background: '#04080f', borderBottom: '1px solid #00f5ff0a',
              display: 'flex', alignItems: 'center', gap: 8,
            }}>
              <div style={{
                width: 4, height: 4, borderRadius: '50%',
                background: '#00f5ff', boxShadow: '0 0 7px #00f5ff', flexShrink: 0,
              }} />
              <span style={{
                fontFamily: "'Share Tech Mono', monospace", fontSize: 7,
                color: '#00f5ff65', letterSpacing: 1,
                flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
              }}>
                {state.garment.replace(/\.[^/.]+$/, '').replace(/_/g, ' ').toUpperCase()}
              </span>
              {meas?.size && (
                <span style={{
                  fontFamily: "'Orbitron', monospace", fontSize: 8,
                  color: '#00ff88', letterSpacing: 2, textShadow: '0 0 7px #00ff8845',
                }}>
                  {meas.size}
                </span>
              )}
            </div>
          )}

          {/* Grid */}
          <div style={{ flex: 1, overflowY: 'auto', padding: '8px' }}>
            {garments.length === 0 ? (
              <div style={{
                padding: '40px 10px', textAlign: 'center',
                fontFamily: "'Share Tech Mono', monospace", fontSize: 8,
                color: '#182430', letterSpacing: 2, lineHeight: 3,
              }}>
                NO GARMENTS<br />
                <span style={{ color: '#0e1c24', fontSize: 7 }}>dataset/train/cloth/</span>
              </div>
            ) : (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 6 }}>
                {filteredGarments.map(g => (
                  <GarmentCard
                    key={g}
                    name={g}
                    idx={garments.indexOf(g)}
                    active={g === state?.garment}
                    onClick={() => switchGarment(g)}
                  />
                ))}
                {filteredGarments.length === 0 && (
                  <div style={{
                    gridColumn: '1/-1', padding: '24px 10px', textAlign: 'center',
                    fontFamily: "'Share Tech Mono', monospace", fontSize: 8,
                    color: '#182430', letterSpacing: 2,
                  }}>
                    NO MATCH FOR &quot;{searchQuery.toUpperCase()}&quot;
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Footer hint */}
          <div style={{
            flexShrink: 0, padding: '5px 12px',
            borderTop: '1px solid #06090f', background: '#030308',
          }}>
            <div style={{
              fontFamily: "'Share Tech Mono', monospace", fontSize: 6,
              color: '#182430', letterSpacing: 1.5, textAlign: 'center',
            }}>
              CLICK GARMENT TO WEAR &bull; SCROLL TO BROWSE
            </div>
          </div>
        </div>

      </div>
    </div>
  )
}
