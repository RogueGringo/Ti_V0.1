const axioms = [
  {
    id: 1, name: 'NO_AVERAGING', short: 'Preserve Variance',
    desc: 'Raw probes never averaged before filtration. The variance structure IS the topological signal. Premature summarization is a category error.',
    color: 'var(--color-red)',
  },
  {
    id: 2, name: 'UPWARD_FLOW', short: 'No Layer Skipping',
    desc: 'L0 → L1 → L2 → L3. No implementation without design. No design without approaches. No approaches without raw context.',
    color: 'var(--color-primary)',
  },
  {
    id: 3, name: 'WAYPOINT_ROUTING', short: 'Phase Transition Gates',
    desc: 'Every routing decision is a topological phase transition. Transitions fire when persistence diagrams exhibit qualitative change — not on a timer.',
    color: 'var(--color-teal)',
  },
  {
    id: 4, name: 'SHAPE_OVER_COUNT', short: 'Gini > Betti',
    desc: 'The Gini trajectory of topological evolution matters more than feature count. 3 coherent sections outperform 12 scattered ones. r = 0.935 validated.',
    color: 'var(--color-success)',
  },
  {
    id: 5, name: 'ADAPTIVE_SCALE', short: 'Data-Driven Thresholds',
    desc: 'epsilon_max is always extracted from the data\'s own geometry (95th percentile of pairwise distances). Never fixed by the user.',
    color: 'var(--color-primary)',
  },
]

export default function Axioms() {
  return (
    <section id="axioms" style={{ padding: 'clamp(3rem, 8vw, 6rem) 0' }}>
      <div style={{ maxWidth: 'var(--content-wide)', margin: '0 auto', padding: '0 var(--space-6)' }}>
        <div style={{ marginBottom: 'var(--space-12)' }}>
          <div style={{
            fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)',
            letterSpacing: '0.12em', textTransform: 'uppercase',
            color: 'var(--color-primary)', marginBottom: 'var(--space-4)',
            display: 'flex', alignItems: 'center', gap: 'var(--space-3)',
          }}>
            Governing Axioms
            <span style={{ flex: 1, maxWidth: '3rem', height: 1, background: 'var(--color-primary)', opacity: 0.4 }} />
          </div>
          <h2 style={{
            fontFamily: 'var(--font-display)', fontSize: 'var(--text-2xl)',
            fontWeight: 500, letterSpacing: '-0.03em', lineHeight: 1.08,
          }}>The Field Equations</h2>
          <p style={{
            fontSize: 'var(--text-base)', color: 'var(--color-text-muted)',
            marginTop: 'var(--space-4)', maxWidth: '58ch', lineHeight: 1.75,
          }}>
            These are not guidelines. They are waypoint constraints on the space of all
            possible cognitive trajectories. A process satisfying all five is on-shell.
          </p>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fill, minmax(min(300px, 100%), 1fr))',
          gap: 'var(--space-6)',
        }}>
          {axioms.map(a => (
            <div key={a.id} style={{
              background: 'var(--color-surface)', border: '1px solid var(--color-border)',
              borderLeft: `3px solid ${a.color}`, borderRadius: 'var(--radius-xl)',
              padding: 'var(--space-6) var(--space-8)',
            }}>
              <div style={{
                fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)',
                letterSpacing: '0.08em', textTransform: 'uppercase',
                color: 'var(--color-text-faint)', marginBottom: 'var(--space-2)',
              }}>Axiom {a.id}</div>
              <h3 style={{
                fontFamily: 'var(--font-display)', fontSize: 'var(--text-lg)',
                fontWeight: 500, marginBottom: 'var(--space-2)',
              }}>{a.name}</h3>
              <div style={{
                fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)',
                color: a.color, marginBottom: 'var(--space-4)',
              }}>{a.short}</div>
              <p style={{
                fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)', lineHeight: 1.72,
              }}>{a.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
