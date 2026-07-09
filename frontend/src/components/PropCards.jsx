const STATS = [
  { key: 'PTS', label: 'Points',   cls: 'pts', color: '#38bdf8' },
  { key: 'AST', label: 'Assists',  cls: 'ast', color: '#22c55e' },
  { key: 'REB', label: 'Rebounds', cls: 'reb', color: '#f97316' },
]

export default function PropCards({ predictions, gameLog, lines = {} }) {
  return (
    <div className="prop-grid">
      {STATS.map(s => (
        <PropCard
          key={s.key}
          stat={s}
          p={predictions[s.key]}
          gameLog={gameLog}
          line={lines[s.key] ?? null}
        />
      ))}
    </div>
  )
}

function PropCard({ stat, p, gameLog, line }) {
  const { key, label, cls, color } = stat

  const last5 = gameLog.slice(0, 5)
  const hits  = last5.filter(g => g[key] > p.prediction).length
  const dots  = Array.from({ length: 5 }, (_, i) => {
    if (i >= last5.length) return 'empty'
    return last5[i][key] > p.prediction ? 'over' : 'under'
  })

  const isHot = p.last5_avg > p.season_avg

  const max      = (p.ceiling * 1.2) || 1
  const floorPct = Math.min(98, (p.floor      / max) * 100)
  const predPct  = Math.min(98, (p.prediction / max) * 100)
  const ceilPct  = Math.min(98, (p.ceiling    / max) * 100)

  const hasLine  = line != null
  const isPush   = hasLine && p.prediction === line
  const isOver   = hasLine && p.prediction > line
  const edge     = hasLine ? (p.prediction - line).toFixed(1) : null

  // dots relative to line when available, otherwise relative to prediction
  const lineDots = line != null
    ? Array.from({ length: 5 }, (_, i) => {
        if (i >= last5.length) return 'empty'
        return last5[i][key] > line ? 'over' : 'under'
      })
    : dots

  const lineHits = line != null
    ? last5.filter(g => g[key] > line).length
    : hits

  return (
    <div className={`prop-card ${cls}`}>
      <div className="prop-top">
        <div className="prop-stat-name">{label}</div>
        {isHot && <span className="prop-fire">🔥</span>}
      </div>

      <div className="prop-number">{p.prediction}</div>
      <div className="prop-sublabel">Projection</div>

      {hasLine && (
        <div className="pp-line">
          <span className="pp-label">DraftKings Line</span>
          {!isPush && (
            <span className={`pp-badge ${isOver ? 'over' : 'under'}`}>
              {isOver ? '▲ OVER' : '▼ UNDER'}
            </span>
          )}
          <span className="pp-score">{line}</span>
          {!isPush && (
            <span className={`pp-edge ${isOver ? 'pos' : 'neg'}`}>
              ({isOver ? '+' : ''}{edge})
            </span>
          )}
        </div>
      )}

      {/* Range bar */}
      <div className="range-track">
        <div className="range-fill" style={{
          left: `${floorPct}%`,
          width: `${ceilPct - floorPct}%`,
          background: color,
        }} />
        <div className="range-pip" style={{ left: `${predPct}%`, background: color }} />
      </div>
      <div className="range-ends">
        <span>Floor {p.floor}</span>
        <span>Ceiling {p.ceiling}</span>
      </div>

      {/* Hit rate dots — relative to line when available */}
      <div className="hit-row">
        {lineDots.map((type, i) => (
          <div key={i} className={`hit-dot ${type}`} title={type} />
        ))}
        <span className="hit-label">
          L5: <span className="hit-rate">{lineHits}/5</span>{' '}
          {line != null ? `over ${line}` : 'over proj'}
        </span>
      </div>

      {/* Meta */}
      <div className="prop-meta">
        <div className="prop-meta-item">
          <div className="meta-label">L5 Avg</div>
          <div className="meta-val" style={{ color }}>{p.last5_avg}</div>
        </div>
        <div className="prop-meta-item">
          <div className="meta-label">Season</div>
          <div className="meta-val">{p.season_avg}</div>
        </div>
      </div>
    </div>
  )
}
