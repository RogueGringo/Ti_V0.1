import Nav from './components/Nav'
import Hero from './components/Hero'
import Pipeline from './components/Pipeline'
import Axioms from './components/Axioms'
import Skills from './components/Skills'
import Install from './components/Install'
import Footer from './components/Footer'

export default function App() {
  return (
    <>
      <Nav />
      <main>
        <Hero />
        <div style={{ height: 1, background: 'var(--color-divider)' }} />
        <Pipeline />
        <div style={{ height: 1, background: 'var(--color-divider)' }} />
        <Axioms />
        <div style={{ height: 1, background: 'var(--color-divider)' }} />
        <Skills />
        <div style={{ height: 1, background: 'var(--color-divider)' }} />
        <Install />
      </main>
      <Footer />
    </>
  )
}
