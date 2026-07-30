import { useState } from 'react'
import {
  ResponsiveContainer, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine,
} from 'recharts'

const TABS = [
  { key: 'PTS', label: 'Points',   cls: 'pts', color: '#17c3ff', gradId: 'gPTS' },
  { key: 'AST', label: 'Assists',  cls: 'ast', color: '#2ecc71', gradId: 'gAST' },
  { key: 'REB', label: 'Rebounds', cls: 'reb', color: '#ff7a1a', gradId: 'gREB' },
]

const Tip = ({ active, payload, label, color }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: '#131316', border: '1px solid #3a3a40',
      borderRadius: 8, padding: '9px 13px', fontSize: '0.8rem',
      boxShadow: '0 8px 24px rgba(0,0,0,0.8)',
    }}>
      <div style={{ color: '#8f8f99', marginBottom: 6 }}>{label}</div>
      <div style={{ color, fontWeight: 700, fontSize: '1rem' }}>{payload[0].value}</div>
    </div>
  )
}

export default function StatChart({ gameLog, predictions }) {
  const [active, setActive] = useState('PTS')
  const tab = TABS.find(t => t.key === active)

  const data = [...gameLog].reverse().map(g => ({
    date: g.GAME_DATE.slice(5),
    val: g[active],
  }))

  const pred = predictions[active].prediction

  return (
    <div className="chart-wrap">
      <div className="chart-tabs">
        {TABS.map(t => (
          <button
            key={t.key}
            className={`chart-tab ${t.cls} ${active === t.key ? 'active' : ''}`}
            onClick={() => setActive(t.key)}
          >
            {t.label}
          </button>
        ))}
      </div>

      <div className="chart-inner">
        <svg style={{ position: 'absolute', width: 0, height: 0 }}>
          <defs>
            {TABS.map(t => (
              <linearGradient key={t.gradId} id={t.gradId} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor={t.color} stopOpacity={0.3} />
                <stop offset="95%" stopColor={t.color} stopOpacity={0} />
              </linearGradient>
            ))}
          </defs>
        </svg>

        <ResponsiveContainer width="100%" height={260}>
          <AreaChart data={data} margin={{ top: 8, right: 8, left: -20, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1e" vertical={false} />
            <XAxis
              dataKey="date"
              tick={{ fill: '#8f8f99', fontSize: 11 }}
              axisLine={false} tickLine={false}
              interval="preserveStartEnd"
            />
            <YAxis
              tick={{ fill: '#8f8f99', fontSize: 11 }}
              axisLine={false} tickLine={false}
              domain={[0, 'auto']}
            />
            <Tooltip content={<Tip color={tab.color} />} />
            <ReferenceLine
              y={pred}
              stroke={tab.color}
              strokeDasharray="5 4"
              strokeOpacity={0.7}
              strokeWidth={1.5}
              label={{
                value: `Proj ${pred}`,
                fill: tab.color,
                fontSize: 11,
                fontWeight: 700,
                position: 'insideTopRight',
              }}
            />
            <Area
              type="monotone"
              dataKey="val"
              name={tab.label}
              stroke={tab.color}
              strokeWidth={2.5}
              fill={`url(#${tab.gradId})`}
              dot={false}
              activeDot={{ r: 5, strokeWidth: 0, fill: tab.color }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
