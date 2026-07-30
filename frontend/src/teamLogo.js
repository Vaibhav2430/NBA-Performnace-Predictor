const NBA_ESPN_MAP = { GSW: 'gs', SAS: 'sa', NYK: 'ny', NOP: 'no', UTA: 'utah' }

export function teamLogoUrl(tricode, isWNBA) {
  if (!tricode) return null
  if (isWNBA) return `https://a.espncdn.com/i/teamlogos/wnba/500/${tricode.toLowerCase()}.png`
  const espn = NBA_ESPN_MAP[tricode] ?? tricode.toLowerCase()
  return `https://a.espncdn.com/i/teamlogos/nba/500/${espn}.png`
}

export function playerHeadshotUrl(playerId, isWNBA) {
  if (!playerId) return null
  const league = isWNBA ? 'wnba' : 'nba'
  return `https://cdn.${league}.com/headshots/${league}/latest/1040x760/${playerId}.png`
}
