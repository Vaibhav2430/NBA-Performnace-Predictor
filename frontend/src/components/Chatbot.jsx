import { useState, useRef, useEffect } from 'react'
import axios from 'axios'

// Three canned questions — routing is just which chip was tapped.
const PROMPTS = [
  { id: 'reasoning', label: 'Why these projections?' },
  { id: 'best',      label: 'Best hit rates' },
  { id: 'worst',     label: 'Worst hit rates' },
]

function Leaderboard({ rows, kind }) {
  return (
    <div className="cb-board">
      <div className="cb-board-head">
        <span>Player</span><span>Hit rate</span><span>Picks</span>
      </div>
      {rows.map((r, i) => (
        <div key={r.player} className="cb-board-row">
          <span className="cb-board-rank">{i + 1}.</span>
          <span className="cb-board-name">{r.player}</span>
          <span className={`cb-board-pct ${kind === 'best' ? 'good' : 'bad'}`}>{r.pct}%</span>
          <span className="cb-board-picks">{r.correct}/{r.total}</span>
        </div>
      ))}
    </div>
  )
}

export default function Chatbot({ league, currentData }) {
  const [open, setOpen]       = useState(false)
  const [busy, setBusy]       = useState(false)
  const [msgs, setMsgs]       = useState([
    { role: 'bot', text: `Hi — I can answer three things about ${league} projections. Pick one below.` },
  ])
  const bodyRef = useRef(null)

  useEffect(() => {
    bodyRef.current?.scrollTo({ top: bodyRef.current.scrollHeight, behavior: 'smooth' })
  }, [msgs, open])

  function push(m) { setMsgs(prev => [...prev, m]) }

  async function ask(promptId) {
    if (busy) return
    const prompt = PROMPTS.find(p => p.id === promptId)
    push({ role: 'user', text: prompt.label })
    setBusy(true)
    try {
      if (promptId === 'reasoning') {
        if (!currentData?.predictions) {
          push({ role: 'bot', text: 'Search for a player first, then tap this again and I\'ll break down their numbers.' })
        } else {
          const res = await axios.post('/chat/reasoning', currentData, { params: { league } })
          push({ role: 'bot', blocks: res.data.blocks })
        }
      } else {
        const res = await axios.get('/chat/leaderboard', { params: { league, limit: 8, min_picks: 8 } })
        const rows = promptId === 'best' ? res.data.best : res.data.worst
        if (!rows?.length) {
          push({ role: 'bot', text: `Not enough resolved ${league} predictions yet to rank players.` })
        } else {
          push({
            role: 'bot',
            text: promptId === 'best'
              ? `Best ${league} hit rates (min ${res.data.min_picks} graded picks, ${res.data.ranked_players} players ranked):`
              : `Worst ${league} hit rates (min ${res.data.min_picks} graded picks):`,
            board: { rows, kind: promptId },
          })
        }
      }
    } catch (err) {
      push({ role: 'bot', text: err.response?.data?.detail ?? 'Something went wrong — try again in a moment.' })
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      {!open && (
        <button className="cb-fab" onClick={() => setOpen(true)} aria-label="Open chat">
          💬
        </button>
      )}

      {open && (
        <div className="cb-panel">
          <div className="cb-head">
            <span className="cb-title">CourtCast Assistant</span>
            <div className="cb-head-right">
              <span className="cb-league">{league}</span>
              <button className="cb-close" onClick={() => setOpen(false)} aria-label="Close chat">
                ✕
              </button>
            </div>
          </div>

          <div className="cb-body" ref={bodyRef}>
            {msgs.map((m, i) => (
              <div key={i} className={`cb-msg ${m.role}`}>
                {m.text && <div className="cb-bubble">{m.text}</div>}
                {m.blocks && (
                  <div className="cb-bubble">
                    {m.blocks.map((b, j) => <p key={j} className="cb-block">{b}</p>)}
                  </div>
                )}
                {m.board && (
                  <div className="cb-bubble wide">
                    <Leaderboard rows={m.board.rows} kind={m.board.kind} />
                  </div>
                )}
              </div>
            ))}
            {busy && <div className="cb-msg bot"><div className="cb-bubble cb-typing">…</div></div>}
          </div>

          <div className="cb-prompts">
            {PROMPTS.map(p => (
              <button key={p.id} className="cb-chip" disabled={busy} onClick={() => ask(p.id)}>
                {p.label}
              </button>
            ))}
          </div>
        </div>
      )}
    </>
  )
}
