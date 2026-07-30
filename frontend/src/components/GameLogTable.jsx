const NBA_ESPN_MAP = { GSW: 'gs', SAS: 'sa', NYK: 'ny', NOP: 'no', UTA: 'utah' }

function oppLogoUrl(abbr, isWNBA) {
  if (!abbr) return null
  if (isWNBA) return `https://a.espncdn.com/i/teamlogos/wnba/500/${abbr.toLowerCase()}.png`
  const espn = NBA_ESPN_MAP[abbr] ?? abbr.toLowerCase()
  return `https://a.espncdn.com/i/teamlogos/nba/500/${espn}.png`
}

function rankClass(rank, total) {
  if (!rank) return 'rank-mid'
  const third = total / 3
  if (rank <= third) return 'rank-good'
  if (rank > third * 2) return 'rank-bad'
  return 'rank-mid'
}

function RankPill({ label, rank, total }) {
  if (rank == null) return null
  return <span className={`rank-badge ${rankClass(rank, total)}`}>{label} #{rank}</span>
}

export default function GameLogTable({ gameLog, predictions, league }) {
  const isWNBA = league === 'WNBA'
  const totalTeams = isWNBA ? 15 : 30

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th style={{ textAlign: 'left' }}>Date</th>
            <th style={{ textAlign: 'left', minWidth: 200 }}>Opponent</th>
            <th style={{ textAlign: 'center' }}>W/L</th>
            <th style={{ textAlign: 'center' }}>Min</th>
            <th style={{ textAlign: 'center' }}>PTS</th>
            <th style={{ textAlign: 'center' }}>AST</th>
            <th style={{ textAlign: 'center' }}>REB</th>
          </tr>
        </thead>
        <tbody>
          {gameLog.map((row, i) => {
            const abbr     = row.OPP_ABBR ?? row.MATCHUP ?? ''
            const logo     = row.OPP_LOGO || oppLogoUrl(abbr, isWNBA)
            const teamName = row.OPP_NAME ?? abbr

            return (
              <tr key={i}>
                <td className="td-muted">{row.GAME_DATE}</td>
                <td>
                  <div className="matchup-cell">
                    <div className="matchup-logo-wrap">
                      {logo
                        ? <img src={logo} alt={abbr} className="matchup-logo" onError={e => { e.target.style.display = 'none' }} />
                        : <div className="matchup-logo-placeholder" />}
                    </div>
                    <div className="matchup-info">
                      <span className="matchup-name">{teamName}</span>
                      <div className="matchup-ranks">
                        <RankPill label="OFF" rank={row.OPP_OFF_RANK} total={totalTeams} />
                        <RankPill label="DEF" rank={row.OPP_DEF_RANK} total={totalTeams} />
                      </div>
                    </div>
                  </div>
                </td>
                <td className={row.WL === 'W' ? 'td-win' : 'td-loss'} style={{ textAlign: 'center' }}>
                  {row.WL}
                </td>
                <td className="td-muted" style={{ textAlign: 'center' }}>{row.MIN}</td>
                <td className="td-pts" style={{ textAlign: 'center' }}>
                  {row.PTS}
                  {row.PTS > predictions.PTS.prediction && <sup style={{ color: '#2ecc71', fontSize: '0.6rem', marginLeft: 2 }}>↑</sup>}
                </td>
                <td className="td-ast" style={{ textAlign: 'center' }}>
                  {row.AST}
                  {row.AST > predictions.AST.prediction && <sup style={{ color: '#2ecc71', fontSize: '0.6rem', marginLeft: 2 }}>↑</sup>}
                </td>
                <td className="td-reb" style={{ textAlign: 'center' }}>
                  {row.REB}
                  {row.REB > predictions.REB.prediction && <sup style={{ color: '#2ecc71', fontSize: '0.6rem', marginLeft: 2 }}>↑</sup>}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
