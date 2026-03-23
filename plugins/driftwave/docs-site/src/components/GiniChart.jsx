import { useRef, useEffect, useState } from 'react'

/**
 * Gini trajectory chart — shows hierarchy evolution during L2 synthesis.
 * Rising = hierarchifying (good). Falling = flattening (investigate).
 * Stable = hold. The shape matters more than the count.
 */

const DEMO_TRAJECTORY = [0.25, 0.28, 0.31, 0.35, 0.38, 0.41, 0.43, 0.44, 0.45]
const DEMO_LABELS = ['init', 'scan', 'cluster_0', 'cluster_1', 'cluster_2', 'merge', 'refine', 'synthesize', 'review']

export default function GiniChart() {
  const canvasRef = useRef(null)
  const [trajectory, setTrajectory] = useState(DEMO_TRAJECTORY)
  const [labels, setLabels] = useState(DEMO_LABELS)

  useEffect(() => {
    fetch('/api/artifact/synthesis')
      .then(r => r.ok ? r.json() : null)
      .then(d => {
        if (d && d.trajectory && d.trajectory.length > 1) {
          setTrajectory(d.trajectory)
          setLabels(d.sections ? d.sections.map(s => s.title.slice(0, 12)) : d.trajectory.map((_, i) => `step_${i}`))
        }
      })
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
      green: cs.getPropertyValue('--color-success').trim() || '#6daa45',
      red: cs.getPropertyValue('--color-red').trim() || '#e74c3c',
      text: cs.getPropertyValue('--color-text-muted').trim() || '#817a66',
      border: cs.getPropertyValue('--color-border').trim() || '#35311e',
      faint: cs.getPropertyValue('--color-text-faint').trim() || '#544f3e',
    }

    ctx.fillStyle = colors.bg
    ctx.fillRect(0, 0, w, h)

    const pad = { top: 30, bottom: 50, left: 50, right: 20 }
    const cw = w - pad.left - pad.right
    const ch = h - pad.top - pad.bottom

    const yMin = Math.min(...trajectory) * 0.9
    const yMax = Math.max(...trajectory) * 1.1
    const n = trajectory.length

    function xScale(i) { return pad.left + (i / (n - 1)) * cw }
    function yScale(v) { return pad.top + ch - ((v - yMin) / (yMax - yMin)) * ch }

    // Grid
    ctx.strokeStyle = colors.border
    ctx.lineWidth = 0.5
    for (let g = 0; g < 5; g++) {
      const gy = pad.top + (g / 4) * ch
      ctx.beginPath()
      ctx.moveTo(pad.left, gy)
      ctx.lineTo(w - pad.right, gy)
      ctx.stroke()
    }

    // Slope zones
    for (let i = 1; i < n; i++) {
      const slope = trajectory[i] - trajectory[i - 1]
      const x0 = xScale(i - 1)
      const x1 = xScale(i)
      const y0 = yScale(trajectory[i - 1])
      const y1 = yScale(trajectory[i])

      // Fill zone under segment
      ctx.globalAlpha = 0.08
      ctx.fillStyle = slope >= 0 ? colors.green : colors.red
      ctx.beginPath()
      ctx.moveTo(x0, y0)
      ctx.lineTo(x1, y1)
      ctx.lineTo(x1, pad.top + ch)
      ctx.lineTo(x0, pad.top + ch)
      ctx.closePath()
      ctx.fill()
      ctx.globalAlpha = 1
    }

    // Main line
    ctx.strokeStyle = colors.gold
    ctx.lineWidth = 2.5
    ctx.beginPath()
    for (let i = 0; i < n; i++) {
      const px = xScale(i)
      const py = yScale(trajectory[i])
      if (i === 0) ctx.moveTo(px, py)
      else ctx.lineTo(px, py)
    }
    ctx.stroke()

    // Dots + slope indicators
    for (let i = 0; i < n; i++) {
      const px = xScale(i)
      const py = yScale(trajectory[i])

      // Dot
      ctx.beginPath()
      ctx.arc(px, py, 4, 0, Math.PI * 2)
      if (i > 0) {
        const slope = trajectory[i] - trajectory[i - 1]
        ctx.fillStyle = slope >= 0 ? colors.green : colors.red
      } else {
        ctx.fillStyle = colors.gold
      }
      ctx.fill()

      // Slope arrow
      if (i > 0) {
        const slope = trajectory[i] - trajectory[i - 1]
        ctx.fillStyle = slope >= 0 ? colors.green : colors.red
        ctx.font = '12px sans-serif'
        ctx.textAlign = 'center'
        ctx.fillText(slope >= 0.01 ? '▲' : slope <= -0.01 ? '▼' : '─', px, py - 10)
      }
    }

    // X axis labels
    ctx.fillStyle = colors.faint
    ctx.font = '9px "JetBrains Mono", monospace'
    ctx.textAlign = 'center'
    for (let i = 0; i < n; i++) {
      if (n > 10 && i % 2 !== 0 && i !== n - 1) continue
      ctx.save()
      ctx.translate(xScale(i), h - pad.bottom + 14)
      ctx.rotate(-0.4)
      ctx.fillText(labels[i] || `${i}`, 0, 0)
      ctx.restore()
    }

    // Y axis
    ctx.fillStyle = colors.text
    ctx.font = '10px "JetBrains Mono", monospace'
    ctx.textAlign = 'right'
    for (let g = 0; g < 5; g++) {
      const val = yMin + (g / 4) * (yMax - yMin)
      ctx.fillText(val.toFixed(2), pad.left - 8, pad.top + ch - (g / 4) * ch + 4)
    }

    // Y axis label
    ctx.save()
    ctx.translate(12, pad.top + ch / 2)
    ctx.rotate(-Math.PI / 2)
    ctx.textAlign = 'center'
    ctx.fillStyle = colors.text
    ctx.font = '10px "JetBrains Mono", monospace'
    ctx.fillText('Gini coefficient', 0, 0)
    ctx.restore()

    // Status label
    const lastSlope = n > 1 ? trajectory[n - 1] - trajectory[n - 2] : 0
    const status = lastSlope >= 0.01 ? 'HIERARCHIFYING ▲' : lastSlope <= -0.01 ? 'FLATTENING ▼' : 'STABLE ─'
    const statusColor = lastSlope >= 0.01 ? colors.green : lastSlope <= -0.01 ? colors.red : colors.gold
    ctx.fillStyle = statusColor
    ctx.font = 'bold 11px "JetBrains Mono", monospace'
    ctx.textAlign = 'right'
    ctx.fillText(status, w - pad.right, pad.top - 8)

  }, [trajectory, labels])

  return (
    <section style={{ padding: 'var(--space-12) 0' }}>
      <div style={{ maxWidth: 'var(--content-wide)', margin: '0 auto', padding: '0 var(--space-6)' }}>
        <p style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)', letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--color-success)', marginBottom: 'var(--space-4)' }}>
          Gini Trajectory
        </p>
        <h3 style={{ fontFamily: 'var(--font-display)', fontSize: 'var(--text-xl)', fontWeight: 500, letterSpacing: '-0.02em', marginBottom: 'var(--space-4)' }}>
          Shape Over Count
        </h3>
        <p style={{ fontSize: 'var(--text-sm)', color: 'var(--color-text-muted)', marginBottom: 'var(--space-6)', maxWidth: '55ch', lineHeight: 1.8 }}>
          The Gini coefficient tracks whether dominant features are emerging (<span style={{ color: 'var(--color-success)' }}>▲ hierarchifying</span>) or all features are equally weighted (<span style={{ color: 'var(--color-red)' }}>▼ flattening</span>). A rising trajectory means the design is converging. A falling one means it's losing focus.
        </p>
        <div style={{
          width: '100%',
          height: 250,
          borderRadius: 'var(--radius-lg)',
          border: '1px solid var(--color-border)',
          overflow: 'hidden',
        }}>
          <canvas
            ref={canvasRef}
            style={{ width: '100%', height: '100%' }}
            aria-label="Gini trajectory chart showing hierarchy evolution during design synthesis"
          />
        </div>
      </div>
    </section>
  )
}
