import { useState, useEffect, useRef } from 'react'
import axios from 'axios'

const STATS = [
  { key: 'PTS', label: 'Points',   color: '#17c3ff' },
  { key: 'AST', label: 'Assists',  color: '#2ecc71' },
  { key: 'REB', label: 'Rebounds', color: '#ff7a1a' },
]

export default function AccuracyPanel({ league }) {
  const [open, setOpen]       = useState(false)
  const [stats, setStats]     = useState(null)
  const [loading, setLoading] = useState(false)
  const [seeding, setSeeding] = useState(false)
  const [seedDone, setSeedDone] = useState(false)
  const ref = useRef(null)

  async function load() {
    setLoading(true)
    try {
      const res = await axios.get('/accuracy')
      setStats(res.data)
    } catch {
      setStats(null)
    } finally {
      setLoading(false)
    }
  }

  async function seed() {
    setSeeding(true)
    setSeedDone(false)
    try {
      await axios.post(`/accuracy/seed?league=${league}`)
      setSeedDone(true)
    } catch {
      // silent
    } finally {
      setSeeding(false)
    }
  }

  function toggle() {
    if (!open && !stats) load()
    setOpen(o => !o)
  }

  useEffect(() => {
    function handler(e) {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  // Reset seedDone when league changes
  useEffect(() => { setSeedDone(false) }, [league])

  const overall = stats?.pct

  return (
    <div className="acc-wrap" ref={ref}>
      <button className="acc-trigger" onClick={toggle}>
        <span className="acc-trigger-icon">🎯</span>
        {overall != null ? `${overall}%` : 'Accuracy'}
      </button>

      {open && (
        <div className="acc-panel">
          <div className="acc-panel-title">O/U Accuracy</div>

          {loading && <div className="acc-spinner" />}

          {!loading && stats && stats.total === 0 && (
            <div className="acc-empty">
              No resolved predictions yet.<br />
              Check back after today's games finish.
            </div>
          )}

          {!loading && stats && stats.total > 0 && (
            <>
              {/* Per-stat cards */}
              <div className="acc-stat-cards">
                {STATS.map(({ key, label, color }) => {
                  const s = stats.by_stat[key]
                  const haData = s && s.total > 0
                  return (
                    <div key={key} className="acc-stat-card">
                      <div className="acc-card-label" style={{ color }}>{label}</div>
                      <div className="acc-card-pct" style={{ color: haData ? pctColor(s.pct) : 'var(--muted)' }}>
                        {haData ? `${s.pct}%` : '—'}
                      </div>
                      <div className="acc-card-count">
                        {haData ? `${s.correct}/${s.total}` : 'no data'}
                      </div>
                      {haData && (
                        <div className="acc-card-bar-wrap">
                          <div className="acc-card-bar" style={{ width: `${s.pct}%`, background: pctColor(s.pct) }} />
                        </div>
                      )}
                    </div>
                  )
                })}
              </div>

              {/* Overall */}
              <div className="acc-overall-row">
                <span className="acc-overall-label">Overall</span>
                <span className="acc-overall-pct" style={{ color: pctColor(overall) }}>{overall}%</span>
                <span className="acc-overall-n">{stats.correct}/{stats.total}</span>
              </div>

              {stats.pending_count > 0 && (
                <div className="acc-pending">{stats.pending_count} pending result{stats.pending_count > 1 ? 's' : ''}</div>
              )}
            </>
          )}

          {/* Seed button */}
          <div className="acc-seed-row">
            <button
              className="acc-seed-btn"
              onClick={seed}
              disabled={seeding}
            >
              {seeding ? 'Running predictions…' : seedDone ? '✓ Done — check back later' : `Seed all ${league} players today`}
            </button>
            {seedDone && (
              <div className="acc-seed-note">
                Running in background (~2–3 min). Refresh accuracy after games finish.
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function pctColor(pct) {
  if (pct == null) return 'var(--muted)'
  if (pct >= 60) return 'var(--green)'
  if (pct >= 50) return 'var(--orange)'
  return '#ff8a8a'
}
