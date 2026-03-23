export default function Footer() {
  return (
    <footer style={{
      borderTop: '1px solid var(--color-divider)',
      padding: 'var(--space-8) var(--space-6)',
      textAlign: 'center',
    }}>
      <p style={{
        fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)',
        color: 'var(--color-text-faint)', letterSpacing: '0.06em',
      }}>
        driftwave — Aaron Jones, 2026
        <br />
        <span style={{ color: 'var(--color-text-muted)' }}>
          Shape over count. Trajectory over snapshot.
        </span>
      </p>
    </footer>
  )
}
