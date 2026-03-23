import { useRef, useEffect, useState } from 'react'
import * as THREE from 'three'

/**
 * Force-directed 3D graph of artifact clusters.
 * Nodes = files/artifacts, edges = similarity above threshold.
 * Colors: gold = cluster members, teal = cross-cluster edges, gray = noise.
 *
 * Reads FilteredTopology artifact from /tmp/dw-artifacts/filtered.json
 * Falls back to demo data if no artifact exists.
 */

const DEMO_DATA = {
  clusters: [
    { id: 0, label: 'Topology Engine', members: ['sheaf_laplacian.py', 'transport_maps.py', 'torch_sheaf.py', 'spectral.py', 'persistence.py'], bar_length: 0.85 },
    { id: 1, label: 'Experiment Pipeline', members: ['phase3c.py', 'phase3d.py', 'sweep.py'], bar_length: 0.72 },
    { id: 2, label: 'Visualization', members: ['plots.py', 'dashboard.py'], bar_length: 0.45 },
  ],
  noise: ['scratch.py', 'old_test.py'],
  distances: null
}

const COLORS = {
  gold: 0xd08a28,
  teal: 0x45a8b0,
  gray: 0x544f3e,
  bg: 0x0f0d08,
  edge: 0x35311e,
  glow: 0xeda040,
}

function createNode(x, y, z, color, size = 0.15) {
  const geo = new THREE.SphereGeometry(size, 16, 16)
  const mat = new THREE.MeshPhongMaterial({ color, emissive: color, emissiveIntensity: 0.3 })
  const mesh = new THREE.Mesh(geo, mat)
  mesh.position.set(x, y, z)
  return mesh
}

function createEdge(from, to, color, opacity = 0.3) {
  const points = [from.position.clone(), to.position.clone()]
  const geo = new THREE.BufferGeometry().setFromPoints(points)
  const mat = new THREE.LineBasicMaterial({ color, transparent: true, opacity })
  return new THREE.Line(geo, mat)
}

export default function ForceGraph() {
  const mountRef = useRef(null)
  const [data, setData] = useState(DEMO_DATA)
  const [status, setStatus] = useState('demo')

  // Try to load live artifact
  useEffect(() => {
    fetch('/api/artifact/filtered')
      .then(r => r.ok ? r.json() : null)
      .then(d => { if (d && d.clusters) { setData(d); setStatus('live') } })
      .catch(() => {}) // stay on demo
  }, [])

  useEffect(() => {
    if (!mountRef.current) return
    const container = mountRef.current
    const w = container.clientWidth
    const h = 400

    // Scene
    const scene = new THREE.Scene()
    scene.background = new THREE.Color(COLORS.bg)

    const camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 100)
    camera.position.set(0, 0, 8)

    const renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(w, h)
    renderer.setPixelRatio(window.devicePixelRatio)
    container.innerHTML = ''
    container.appendChild(renderer.domElement)

    // Lights
    scene.add(new THREE.AmbientLight(0x444444))
    const point = new THREE.PointLight(COLORS.gold, 1, 20)
    point.position.set(3, 3, 5)
    scene.add(point)

    // Build nodes
    const nodes = []
    const clusterColors = [COLORS.gold, COLORS.teal, COLORS.glow, 0x6daa45]

    data.clusters.forEach((cluster, ci) => {
      const color = clusterColors[ci % clusterColors.length]
      const angle0 = (ci / data.clusters.length) * Math.PI * 2
      const radius = 2.5

      cluster.members.forEach((member, mi) => {
        const spread = 0.8
        const a = angle0 + (mi - cluster.members.length / 2) * 0.3
        const x = Math.cos(a) * radius + (Math.random() - 0.5) * spread
        const y = Math.sin(a) * radius + (Math.random() - 0.5) * spread
        const z = (Math.random() - 0.5) * spread
        const size = 0.1 + cluster.bar_length * 0.1

        const node = createNode(x, y, z, color, size)
        node.userData = { label: member, cluster: ci }
        nodes.push(node)
        scene.add(node)
      })
    })

    // Noise nodes (small, gray)
    ;(data.noise || []).forEach(member => {
      const x = (Math.random() - 0.5) * 6
      const y = (Math.random() - 0.5) * 6
      const z = (Math.random() - 0.5) * 2
      const node = createNode(x, y, z, COLORS.gray, 0.06)
      node.userData = { label: member, cluster: -1 }
      nodes.push(node)
      scene.add(node)
    })

    // Edges — connect nodes in same cluster
    const edges = []
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const ci = nodes[i].userData.cluster
        const cj = nodes[j].userData.cluster
        if (ci === -1 || cj === -1) continue
        const dist = nodes[i].position.distanceTo(nodes[j].position)

        if (ci === cj && dist < 3) {
          // Intra-cluster: gold/teal
          const color = clusterColors[ci % clusterColors.length]
          const edge = createEdge(nodes[i], nodes[j], color, 0.4)
          edges.push(edge)
          scene.add(edge)
        } else if (ci !== cj && dist < 2) {
          // Cross-cluster: dim
          const edge = createEdge(nodes[i], nodes[j], COLORS.edge, 0.15)
          edges.push(edge)
          scene.add(edge)
        }
      }
    }

    // Simple force simulation
    const velocities = nodes.map(() => new THREE.Vector3())
    let frame = 0

    function simulate() {
      const dt = 0.016
      const repulsion = 0.5
      const attraction = 0.02
      const damping = 0.92
      const centering = 0.01

      for (let i = 0; i < nodes.length; i++) {
        const pi = nodes[i].position
        const vi = velocities[i]

        // Centering force
        vi.x -= pi.x * centering
        vi.y -= pi.y * centering
        vi.z -= pi.z * centering

        // Repulsion from all other nodes
        for (let j = 0; j < nodes.length; j++) {
          if (i === j) continue
          const dx = pi.x - nodes[j].position.x
          const dy = pi.y - nodes[j].position.y
          const dz = pi.z - nodes[j].position.z
          const dist2 = dx * dx + dy * dy + dz * dz + 0.01
          const force = repulsion / dist2
          vi.x += dx * force * dt
          vi.y += dy * force * dt
          vi.z += dz * force * dt
        }

        // Attraction to same-cluster nodes
        for (let j = 0; j < nodes.length; j++) {
          if (i === j) continue
          if (nodes[i].userData.cluster === nodes[j].userData.cluster && nodes[i].userData.cluster !== -1) {
            const dx = nodes[j].position.x - pi.x
            const dy = nodes[j].position.y - pi.y
            const dz = nodes[j].position.z - pi.z
            vi.x += dx * attraction * dt
            vi.y += dy * attraction * dt
            vi.z += dz * attraction * dt
          }
        }

        vi.multiplyScalar(damping)
      }

      // Apply velocities
      for (let i = 0; i < nodes.length; i++) {
        nodes[i].position.add(velocities[i])
      }

      // Update edges
      edges.forEach(edge => {
        const positions = edge.geometry.attributes.position
        if (positions) {
          // Find connected nodes by proximity (edges don't store refs)
          // For simplicity, edges are static after initial layout
        }
      })
    }

    // Animation loop
    let animId
    function animate() {
      animId = requestAnimationFrame(animate)
      frame++

      // Run simulation for first 300 frames
      if (frame < 300) simulate()

      // Gentle rotation
      scene.rotation.y += 0.002

      // Breathing on cluster nodes
      nodes.forEach(n => {
        if (n.userData.cluster >= 0) {
          const breath = 1 + 0.05 * Math.sin(frame * 0.02 + n.userData.cluster)
          n.scale.setScalar(breath)
        }
      })

      renderer.render(scene, camera)
    }
    animate()

    // Resize handler
    const onResize = () => {
      const newW = container.clientWidth
      camera.aspect = newW / h
      camera.updateProjectionMatrix()
      renderer.setSize(newW, h)
    }
    window.addEventListener('resize', onResize)

    return () => {
      cancelAnimationFrame(animId)
      window.removeEventListener('resize', onResize)
      renderer.dispose()
    }
  }, [data])

  return (
    <section style={{ padding: 'var(--space-16) 0' }}>
      <div style={{ maxWidth: 'var(--content-wide)', margin: '0 auto', padding: '0 var(--space-6)' }}>
        <p style={{ fontFamily: 'var(--font-mono)', fontSize: 'var(--text-xs)', letterSpacing: '0.12em', textTransform: 'uppercase', color: 'var(--color-primary)', marginBottom: 'var(--space-4)' }}>
          Topology Visualization
        </p>
        <h2 style={{ fontFamily: 'var(--font-display)', fontSize: 'var(--text-2xl)', fontWeight: 500, letterSpacing: '-0.03em', marginBottom: 'var(--space-4)' }}>
          Artifact Space
        </h2>
        <p style={{ fontSize: 'var(--text-base)', color: 'var(--color-text-muted)', marginBottom: 'var(--space-8)', maxWidth: '58ch' }}>
          Each node is a file. Clusters form from persistent homology — the same math that finds structure in zeta zeros, applied to your codebase. {status === 'live' ? '🟢 Live data from pipeline.' : '⚪ Demo data — run /wavefront to see your project.'}
        </p>
        <div
          ref={mountRef}
          style={{
            width: '100%',
            height: 400,
            borderRadius: 'var(--radius-xl)',
            border: '1px solid var(--color-border)',
            overflow: 'hidden',
          }}
        />
        <div style={{ display: 'flex', gap: 'var(--space-4)', marginTop: 'var(--space-4)', flexWrap: 'wrap' }}>
          {data.clusters.map((c, i) => (
            <span key={c.id} style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 'var(--text-xs)',
              color: ['var(--color-primary)', 'var(--color-teal)', '#eda040', 'var(--color-success)'][i % 4],
              padding: '2px 8px',
              border: '1px solid',
              borderRadius: 'var(--radius-full)',
            }}>
              {c.label} ({c.members.length})
            </span>
          ))}
          {data.noise?.length > 0 && (
            <span style={{
              fontFamily: 'var(--font-mono)',
              fontSize: 'var(--text-xs)',
              color: 'var(--color-text-faint)',
              padding: '2px 8px',
              border: '1px solid var(--color-border)',
              borderRadius: 'var(--radius-full)',
            }}>
              noise ({data.noise.length})
            </span>
          )}
        </div>
      </div>
    </section>
  )
}
