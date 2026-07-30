import { useState, useEffect, useRef } from 'react'
import axios from 'axios'

// Pacific fire times: 12:30 PM, 3:30 PM, 6:30 PM, 10:00 PM
const FIRE_TIMES_PT = [[12, 30], [15, 30], [18, 30], [22, 0]]

function msUntilNextFire() {
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/Los_Angeles',
    hour: '2-digit', minute: '2-digit', hour12: false,
  }).formatToParts(new Date())
  const h = parseInt(parts.find(p => p.type === 'hour').value)
  const m = parseInt(parts.find(p => p.type === 'minute').value)
  const nowMins = h * 60 + m
  for (const [fh, fm] of FIRE_TIMES_PT) {
    const fireMins = fh * 60 + fm
    if (fireMins > nowMins) return (fireMins - nowMins) * 60 * 1000
  }
  // Past 10 PM — next fire is 12:30 PM tomorrow
  return ((24 * 60 - nowMins) + (12 * 60 + 30)) * 60 * 1000
}

const STATS = [
  { key: 'PTS', label: 'Points',   color: '#17c3ff' },
  { key: 'AST', label: 'Assists',  color: '#2ecc71' },
  { key: 'REB', label: 'Rebounds', color: '#ff7a1a' },
]

function pctColor(pct) {
  if (pct == null) return 'var(--muted)'
  if (pct >= 60) return 'var(--green)'
  if (pct >= 50) return 'var(--orange)'
  return '#ff8a8a'
}

export default function AccuracyCards({ league }) {
  const [stats, setStats] = useState(null)
  const timer = useRef(null)

  useEffect(() => {
    function fetchAndSchedule() {
      axios.get(`/accuracy?league=${league}`).then(r => setStats(r.data)).catch(() => setStats(null))
      timer.current = setTimeout(fetchAndSchedule, msUntilNextFire())
    }
    fetchAndSchedule()
    return () => clearTimeout(timer.current)
  }, [league])

  const hasData = stats && stats.total > 0

  return (
    <div className="accuracy-section">
      <div className="accuracy-title">O/U Accuracy</div>

      <div className="accuracy-cards">
        {STATS.map(({ key, label, color }) => {
          const s = hasData ? stats.by_stat[key] : null
          const hasStatData = s && s.total > 0
          return (
            <div key={key} className="accuracy-card">
              <div className="accuracy-card-label" style={{ color }}>{label}</div>
              <div
                className="accuracy-card-pct"
                style={{ color: hasStatData ? pctColor(s.pct) : 'var(--border2)' }}
              >
                {hasStatData ? `${s.pct}%` : '—'}
              </div>
              <div className="accuracy-card-sub">
                {hasStatData ? `${s.correct}/${s.total}` : 'no data'}
              </div>
              <div className="accuracy-card-bar-track">
                <div
                  className="accuracy-card-bar-fill"
                  style={{
                    width: hasStatData ? `${s.pct}%` : '0%',
                    background: hasStatData ? pctColor(s.pct) : 'var(--border2)',
                  }}
                />
              </div>
            </div>
          )
        })}
      </div>

      {stats?.pending_count > 0 && (
        <div className="accuracy-pending-note">{stats.pending_count} pending</div>
      )}
    </div>
  )
}
