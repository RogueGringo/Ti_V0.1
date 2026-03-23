import { useRef, useEffect, useState } from 'react'

/**
 * Persistence barcode visualization.
 * Horizontal bars showing birth/death of each topological feature.
 * Long bars = real structure, short bars = noise.
 *
 * Reads from FilteredTopology artifact or uses demo data.
 */

const DEMO_BARCODE = [
  { birth: 0.0, death: 0.001, dimension: 0 },
  { birth: 0.0, death: 0.007, dimension: 0 },
  { birth: 0.0, death: 0.020, dimension: 0 },
  { birth: 0.0, death: 0.048, dimension: 0 },
  { birth: 0.0, death: 0.050, dimension: 0 },
  { birth: 0.0, death: 0.082, dimension: 0 },
  { birth: 0.0, death: 0.155, dimension: 0 },
  { birth: 0.0, death: 0.210, dimension: 0 },
  { birth: 0.0, death: 0.319, dimension: 0 },
  { birth: 0.0, death: Infinity, dimension: 0 },
  { birth: 0.0, death: Infinity, dimension: 0 },
]

export default function Barcode() {
  const canvasRef = useRef(null)
  const [barcode, setBarcode] = useState(DEMO_BARCODE)

  useEffect(() => {
    fetch('/api/artifact/filtered')
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (d && d.barcode) setBarcode(d.barcode) })
      .catch(() => {})
  }, [])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    const dpr = window.devicePixelRatio || 1
    const w = canvas.offsetWidth
    const h = canvas.offsetHeight
    canvas.width = w * dpr
    canvas.height = h * dpr
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    const cs = getComputedStyle(document.documentElement)
    const colors = {
      bg: cs.getPropertyValue('--color-surface').trim() || '#16140d',
      gold: cs.getPropertyValue('--color-primary').trim() || '#d08a28',
      teal: cs.getPropertyValue('--color-teal').trim() || '#45a8b0',
      gray: cs.getPropertyValue('--color-text-faint').trim() || '#544f3e',
      text: cs.getPropertyValue('--color-text-muted').trim() || '#817a66',
      border: cs.getPropertyValue('--color-border').trim() || '#35311e',
    }

    ctx.fillStyle = colors.bg
    ctx.fillRect(0, 0, w, h)

    // Filter out infinite bars for scale, but draw them
    const finite = barcode.filter(b => isFinite(b.death))
    const maxDeath = finite.length > 0 ? Math.max(...finite.map(b => b.death)) : 1
    const medianLife = finite.length > 0
      ? [...finite].sort((a, b) => (a.death - a.birth) - (b.death - b.birth))[Math.floor(finite.length / 2)].death
      : maxDeath / 2

    const pad = { top: 30, bottom: 30, left: 60, right: 20 }
    const cw = w - pad.left - pad.right
    const ch = h - pad.top - pad.bottom
    const barHeight = Math.min(14, ch / barcode.length - 2)
    const barGap = 2

    // Sort by lifetime (longest at top)
    const sorted = [...barcode].sort((a, b) => {
      const la = isFinite(a.death) ? a.death - a.birth : Infinity
      const lb = isFinite(b.death) ? b.death - b.birth : Infinity
      return lb - la
    })

    // X scale
    const xScale = (val) => pad.left + (Math.min(val, maxDeath * 1.1) / (maxDeath * 1.1)) * cw

    // Median threshold line
    ctx.strokeStyle = colors.gray
    ctx.lineWidth = 1
    ctx.setLineDash([4, 4])
    const mx = xScale(medianLife)
    ctx.beginPath()
    ctx.moveTo(mx, pad.top)
    ctx.lineTo(mx, h - pad.bottom)
    ctx.stroke()
    ctx.setLineDash([])

    ctx.fillStyle = colors.gray
    ctx.font = '10px "JetBrains Mono", monospace'
    ctx.textAlign = 'center'
    ctx.fillText('median', mx, pad.top - 8)

    // Draw bars
    sorted.forEach((bar, i) => {
      const y = pad.top + i * (barHeight + barGap)
      const x0 = xScale(bar.birth)
      const lifetime = bar.death - bar.birth
      const isLong = isFinite(lifetime) ? lifetime > medianLife : true
      const x1 = isFinite(bar.death) ? xScale(bar.death) : w - pad.right

      // Bar color: long bars = gold, short bars = gray, infinite = teal
      let color
      if (!isFinite(bar.death)) color = colors.teal
      else if (isLong) color = colors.gold
      else color = colors.gray

      ctx.fillStyle = color
      ctx.globalAlpha = isLong ? 0.9 : 0.4
      ctx.fillRect(x0, y, Math.max(x1 - x0, 2), barHeight)
      ctx.globalAlpha = 1

      // Arrow for infinite bars
      if (!isFinite(bar.death)) {
        ctx.fillStyle = colors.teal
        ctx.beginPath()
        ctx.moveTo(w - pad.right, y + barHeight / 2)
        ctx.lineTo(w - pad.right - 6, y + 1)
        ctx.lineTo(w - pad.right - 6, y + barHeight - 1)
        ctx.fill()
      }

      // Label
      ctx.fillStyle = colors.text
      ctx.font = '9px "JetBrains Mono", monospace'
      ctx.textAlign = 'right'
      ctx.fillText(`H${bar.dimension}`, pad.left - 8, y + barHeight - 2)
    })

    // X axis
    ctx.strokeStyle = colors.border
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(pad.left, h - pad.bottom)
    ctx.lineTo(w - pad.right, h - pad.bottom)
    ctx.stroke()

    ctx.fillStyle = colors.text
    ctx.font = '10px "JetBrains Mono", monospace'
    ctx.textAlign = 'center'
    const ticks = 5
    for (let i = 0; i <= ticks; i++) {
      const val = (i / ticks) * maxDeath * 1.1
      ctx.fillText(val.toFixed(2), xScale(val), h - pad.bottom + 16)
    }
    ctx.fillText('ε (filtration parameter)', w / 2, h - 4)
  }, [barcode])

  return (
    <section style={{ padding: 'var(--space-12) 0' }}>
      <div style={{ maxWidth: 'var(--content-wide)', margin: '0 auto', padding: '0 var(--space-6)' }}>
        <p style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)', letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--color-teal)', marginBottom: 'var(--space-4)' }}>
          Persistence Barcode
        </p>
        <h3 style={{ fontFamily: 'var(--font-display)', fontSize: 'var(--text-xl)', fontWeight: 500, letterSpacing: '-0.02em', marginBottom: 'var(--space-4)' }}>
          What Survived Filtration
        </h3>
        <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)', marginBottom: 'var(--space-6)', maxWidth: '55ch', lineHeight: 1.8 }}>
          Each bar is a topological feature. <span style={{ color: 'var(--color-primary)' }}>Long gold bars</span> = real structure (persistent clusters). <span style={{ color: 'var(--color-text-faint)' }}>Short gray bars</span> = noise (filtered out). <span style={{ color: 'var(--color-teal)' }}>Teal arrows</span> = components that never die (connected components of the whole).
        </p>
        <div style={{
          width: '100%',
          height: 280,
          borderRadius: 'var(--radius-lg)',
          border: '1px solid var(--color-border)',
          overflow: 'hidden',
        }}>
          <canvas
            ref={canvasRef}
            style={{ width: '100%', height: '100%' }}
            aria-label="Persistence barcode showing birth and death of topological features"
          />
        </div>
      </div>
    </section>
  )
}
