import { useState } from 'react'
import axios from 'axios'
import GamesSidebar from './components/GamesSidebar'
import AccuracyCards from './components/AccuracyCards'
import PlayerSearch from './components/PlayerSearch'
import PropCards from './components/PropCards'
import StatChart from './components/StatChart'
import GameLogTable from './components/GameLogTable'

export default function App() {
  const [league, setLeague]   = useState('NBA')   // 'NBA' | 'WNBA'
  const [data, setData]       = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError]     = useState(null)

  function switchLeague(l) {
    if (l === league) return
    setLeague(l)
    setData(null)
    setError(null)
  }

  async function handleSearch(playerName) {
    setLoading(true)
    setError(null)
    setData(null)
    const endpoint = league === 'WNBA' ? '/wnba/predict' : '/predict'
    try {
      const res = await axios.get(endpoint, { params: { player: playerName } })
      setData(res.data)
    } catch (err) {
      setError(err.response?.data?.detail ?? 'Player not found or no data available.')
    } finally {
      setLoading(false)
    }
  }

  const hasResults = data && !loading
  const isWNBA     = league === 'WNBA'

  const NBA_ESPN_MAP = { GSW: 'gs', SAS: 'sa', NYK: 'ny', NOP: 'no', UTA: 'utah' }
  function teamLogoUrl() {
    if (!data) return null
    const abbr = isWNBA ? data.team_abbr : data.team
    if (!abbr) return null
    if (isWNBA) return `https://a.espncdn.com/i/teamlogos/wnba/500/${abbr.toLowerCase()}.png`
    const espn = NBA_ESPN_MAP[abbr] ?? abbr.toLowerCase()
    return `https://a.espncdn.com/i/teamlogos/nba/500/${espn}.png`
  }

  function rankClass(rank) {
    if (!rank) return 'rank-mid'
    const total = isWNBA ? 15 : 30
    const third = total / 3
    if (rank <= third) return 'rank-good'
    if (rank > third * 2) return 'rank-bad'
    return 'rank-mid'
  }

  return (
    <>
      <header className="header">
        <div className="logo">
          CourtCast
        </div>

        <AccuracyCards league={league} />

        {/* League toggle */}
        <div className="league-toggle">
          <button
            className={`league-btn ${league === 'NBA' ? 'active' : ''}`}
            onClick={() => switchLeague('NBA')}
          >
            NBA
          </button>
          <button
            className={`league-btn wnba ${league === 'WNBA' ? 'active' : ''}`}
            onClick={() => switchLeague('WNBA')}
          >
            WNBA
          </button>
        </div>
      </header>

      <div className="shell">
        <GamesSidebar league={league} />

        <main className={`main ${hasResults ? '' : 'empty'}`}>
          <div className="search-section">
            {!hasResults && (
              <div className="search-hero">
                <h1>CourtCast</h1>
              </div>
            )}
            <PlayerSearch onSearch={handleSearch} loading={loading} league={league} />
            {error && <div className="error">⚠ {error}</div>}
          </div>

          {loading && (
            <div className="loading">
              <div className="spinner" />
              <span>
                {isWNBA
                  ? 'Fetching WNBA game log & training model… (~15s)'
                  : 'Fetching game log & training model… (~10s)'}
              </span>
            </div>
          )}

          {hasResults && (
            <div className="results fade-in">
              <div className="player-banner">
                <div className="player-left">
                  <div className="player-avi" style={isWNBA ? { background: 'linear-gradient(135deg,#a855f7,#ec4899)' } : {}}>
                    {teamLogoUrl()
                      ? <img src={teamLogoUrl()} alt={data.team} style={{ width: '85%', height: '85%', objectFit: 'contain' }} />
                      : '🏀'}
                  </div>
                  <div>
                    <div className="player-name">{data.player}</div>
                    <div className="player-sub">
                      {data.team_name ?? data.team}
                      {data.team_off_rank != null && (
                        <span className={`rank-badge ${rankClass(data.team_off_rank)}`}>OFF #{data.team_off_rank}</span>
                      )}
                      {data.team_def_rank != null && (
                        <span className={`rank-badge ${rankClass(data.team_def_rank)}`}>DEF #{data.team_def_rank}</span>
                      )}
                      {' · '}{data.season} · {data.games_used} games analyzed
                    </div>
                  </div>
                </div>
              </div>

              {data.injury && (() => {
                const inj = data.injury
                const isOut = inj.status.toLowerCase() === 'out'
                return (
                  <div className={`injury-banner ${isOut ? 'out' : 'limited'}`}>
                    <span className="injury-icon">{isOut ? '🚫' : '⚠️'}</span>
                    <div>
                      <span className="injury-status">{inj.status}</span>
                      {inj.injury_type && <span className="injury-type">· {inj.injury_type}</span>}
                      {inj.comment && <div style={{ marginTop: 4, opacity: 0.85 }}>{inj.comment}</div>}
                    </div>
                  </div>
                )
              })()}

              <div className="section-label">Next Game Projections</div>
              <PropCards predictions={data.predictions} gameLog={data.game_log} league={league} lines={data.lines ?? {}} />

              <div className="section-label">Performance Trend</div>
              <StatChart gameLog={data.game_log} predictions={data.predictions} />

              <div className="section-label">Game Log</div>
              <GameLogTable gameLog={data.game_log} predictions={data.predictions} league={league} />
            </div>
          )}
        </main>
      </div>
    </>
  )
}
