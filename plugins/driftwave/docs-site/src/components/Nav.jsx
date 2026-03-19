import { useState } from 'react'

const sections = ['Pipeline', 'Axioms', 'Skills', 'Install']

export default function Nav() {
  const [theme, setTheme] = useState('dark')
  const [menuOpen, setMenuOpen] = useState(false)

  const toggle = () => {
    const next = theme === 'dark' ? 'light' : 'dark'
    setTheme(next)
    document.documentElement.setAttribute('data-theme', next)
  }

  return (
    <header style={{
      position: 'sticky', top: 0, zIndex: 100,
      background: 'color-mix(in oklch, var(--color-bg) 88%, transparent)',
      backdropFilter: 'blur(18px) saturate(1.3)',
      borderBottom: '1px solid var(--color-divider)',
    }}>
      <div style={{
        maxWidth: 'var(--content-wide)', margin: '0 auto',
        padding: 'var(--space-4) var(--space-6)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <a href="#" style={{
          textDecoration: 'none', color: 'var(--color-text)',
          fontFamily: 'var(--font-display)', fontSize: 'var(--text-lg)',
          fontWeight: 500, letterSpacing: '-0.02em',
          display: 'flex', alignItems: 'center', gap: 'var(--space-3)',
        }}>
          <span style={{ color: 'var(--color-primary)' }}>~</span> driftwave
        </a>

        <nav style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-8)' }}>
          {sections.map(s => (
            <a key={s} href={`#${s.toLowerCase()}`} style={{
              fontSize: 'var(--text-sm)', fontWeight: 500,
              textDecoration: 'none', color: 'var(--color-text-muted)',
              letterSpacing: '0.025em', textTransform: 'uppercase',
            }}>{s}</a>
          ))}
          <button onClick={toggle} style={{
            width: 40, height: 40, display: 'flex', alignItems: 'center',
            justifyContent: 'center', borderRadius: 'var(--radius-full)',
            color: 'var(--color-text-muted)', background: 'none', border: 'none', cursor: 'pointer',
          }}>
            {theme === 'dark' ? '☀' : '☾'}
          </button>
        </nav>
      </div>
    </header>
  )
}
