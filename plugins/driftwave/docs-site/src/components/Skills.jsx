const skills = [
  { name: '/dw-map', layer: 'L0', desc: 'Raw artifact ingestion with entropy gate. Preserves full variance.', color: 'var(--color-text-faint)' },
  { name: '/dw-filter', layer: 'L1', desc: 'H₀ persistent clustering and module routing. Adaptive epsilon.', color: 'var(--color-teal)' },
  { name: '/dw-ascend', layer: 'L2/L3', desc: 'Gini routing, sheaf-valued synthesis, ker(L_F) convergence.', color: 'var(--color-primary)' },
  { name: '/wavefront', layer: 'ALL', desc: 'Full pipeline orchestrator. Enforces all 5 axioms end-to-end.', color: 'var(--color-success)' },
  { name: '/topological-brainstorm', layer: 'ALL', desc: 'Brainstorming-as-filtration. Idea-spaces as point clouds.', color: 'var(--color-primary)' },
  { name: '/boundary-mode', layer: 'L3', desc: 'Cross-system collaboration at sheaf-valued abstraction level.', color: 'var(--color-teal)' },
]

export default function Skills() {
  return (
    <section id="skills" style={{ padding: 'clamp(3rem, 8vw, 6rem) 0' }}>
      <div style={{ maxWidth: 'var(--content-wide)', margin: '0 auto', padding: '0 var(--space-6)' }}>
        <div style={{ marginBottom: 'var(--space-12)' }}>
          <div style={{
            fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)',
            letterSpacing: '0.12em', textTransform: 'uppercase',
            color: 'var(--color-primary)', marginBottom: 'var(--space-4)',
            display: 'flex', alignItems: 'center', gap: 'var(--space-3)',
          }}>
            Skills
            <span style={{ flex: 1, maxWidth: '3rem', height: 1, background: 'var(--color-primary)', opacity: 0.4 }} />
          </div>
          <h2 style={{
            fontFamily: 'var(--font-display)', fontSize: 'var(--text-2xl)',
            fontWeight: 500, letterSpacing: '-0.03em', lineHeight: 1.08,
          }}>The Toolkit</h2>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(min(280px, 100%), 1fr))',
          gap: 'var(--space-4)',
        }}>
          {skills.map(s => (
            <div key={s.name} style={{
              background: 'var(--color-surface)', border: '1px solid var(--color-border)',
              borderRadius: 'var(--radius-lg)', padding: 'var(--space-5) var(--space-6)',
            }}>
              <div style={{
                fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)',
                letterSpacing: '0.08em', textTransform: 'uppercase',
                color: 'var(--color-text-faint)', marginBottom: 'var(--space-2)',
              }}>{s.layer}</div>
              <h3 style={{
                fontFamily: 'var(--font-mono)', fontSize: 'var(--text-base)',
                fontWeight: 500, color: s.color, marginBottom: 'var(--space-3)',
              }}>{s.name}</h3>
              <p style={{
                fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)', lineHeight: 1.72,
              }}>{s.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
