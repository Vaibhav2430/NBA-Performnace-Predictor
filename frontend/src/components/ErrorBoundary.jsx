import { Component } from 'react'

export default class ErrorBoundary extends Component {
  state = { error: null }

  static getDerivedStateFromError(error) {
    return { error }
  }

  render() {
    if (this.state.error) {
      return (
        <div style={{
          padding: '40px', color: '#ff9494',
          background: '#000000', minHeight: '100vh',
          fontFamily: 'monospace', fontSize: '0.85rem',
        }}>
          <h2 style={{ color: '#ff7a1a', marginBottom: 12 }}>React Error</h2>
          <pre style={{ whiteSpace: 'pre-wrap' }}>{this.state.error.message}</pre>
          <pre style={{ whiteSpace: 'pre-wrap', color: '#8f8f99', marginTop: 12 }}>
            {this.state.error.stack}
          </pre>
        </div>
      )
    }
    return this.props.children
  }
}
