import { useEffect, useState } from 'react'
import axios from 'axios'
import { teamLogoUrl as teamLogo } from '../teamLogo'

export default function GamesSidebar({ league, selectedGameId, onSelectGame }) {
  const [games, setGames]     = useState([])
  const [loading, setLoading] = useState(true)

  async function fetchGames() {
    setLoading(true)
    try {
      const endpoint = league === 'WNBA' ? '/wnba/games/today' : '/games/today'
      const res = await axios.get(endpoint)
      setGames(res.data)
    } catch {
      setGames([])
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchGames()
    const interval = setInterval(fetchGames, 30000)
    return () => clearInterval(interval)
  }, [league])

  const hasLive = games.some(g => g.statusCode === 2)

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="sidebar-title">
          {hasLive && <div className="live-dot" />}
          {league === 'WNBA' ? 'WNBA' : 'NBA'} · Today's Games
        </div>
      </div>

      {loading && <div className="sidebar-spinner" />}

      {!loading && games.length === 0 && (
        <div className="no-games">
          No games scheduled today.
        </div>
      )}

      {games.map(g => {
        const isLive    = g.statusCode === 2
        const isFinal   = g.statusCode === 3
        const statusCls = isLive ? 'live' : isFinal ? 'final' : 'sched'
        const isWNBA    = league === 'WNBA'
        const isSelected = g.gameId === selectedGameId

        return (
          <div
            key={g.gameId}
            className={`game-card ${statusCls} ${isSelected ? 'selected' : ''}`}
            onClick={() => onSelectGame?.(g)}
            role="button"
            tabIndex={0}
          >
            <div className="game-teams">
              {/* Away */}
              <div className="game-team">
                <img
                  src={teamLogo(g.away.tricode, isWNBA)}
                  alt={g.away.tricode}
                  className="sidebar-team-logo"
                  onError={e => { e.target.style.visibility = 'hidden' }}
                />
                <div className="team-tricode">{g.away.tricode}</div>
                <div className="team-record">{g.away.wins}–{g.away.losses}</div>
              </div>

              {/* Score or vs */}
              <div className="game-score">
                {(isLive || isFinal) ? (
                  <>
                    <span>{g.away.score}</span>
                    <span className="score-sep">–</span>
                    <span>{g.home.score}</span>
                  </>
                ) : (
                  <span style={{ fontSize: '0.75rem', color: '#ffffff', fontWeight: 500 }}>vs</span>
                )}
              </div>

              {/* Home */}
              <div className="game-team">
                <img
                  src={teamLogo(g.home.tricode, isWNBA)}
                  alt={g.home.tricode}
                  className="sidebar-team-logo"
                  onError={e => { e.target.style.visibility = 'hidden' }}
                />
                <div className="team-tricode">{g.home.tricode}</div>
                <div className="team-record">{g.home.wins}–{g.home.losses}</div>
              </div>
            </div>

            <div className={`game-status ${statusCls}`}>
              {isLive ? `🔴 ${g.status}` : g.status}
            </div>
          </div>
        )
      })}
    </aside>
  )
}
