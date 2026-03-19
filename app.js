/* ═══════════════════════════════════════════
   B. Jones Portfolio — app.js
   Theme toggle, tabs, counters, scroll reveals,
   mobile menu, BibTeX copy
   ═══════════════════════════════════════════ */

(function () {
  'use strict';

  /* ─── Theme Toggle ─── */
  const root = document.documentElement;
  const toggle = document.getElementById('themeToggle');
  let theme = root.getAttribute('data-theme') || 'dark';
  root.setAttribute('data-theme', theme);

  const sunSVG = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>';
  const moonSVG = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>';

  function updateToggleIcon() {
    if (!toggle) return;
    toggle.innerHTML = theme === 'dark' ? sunSVG : moonSVG;
    toggle.setAttribute('aria-label', 'Switch to ' + (theme === 'dark' ? 'light' : 'dark') + ' mode');
  }
  updateToggleIcon();

  if (toggle) {
    toggle.addEventListener('click', function () {
      theme = theme === 'dark' ? 'light' : 'dark';
      root.setAttribute('data-theme', theme);
      updateToggleIcon();
    });
  }

  /* ─── Mobile Menu ─── */
  const hamburger = document.getElementById('hamburgerBtn');
  const mobileNav = document.getElementById('mobileNav');

  if (hamburger && mobileNav) {
    hamburger.addEventListener('click', function () {
      mobileNav.classList.toggle('open');
      hamburger.setAttribute('aria-expanded', mobileNav.classList.contains('open'));
    });

    // Close on link click
    mobileNav.querySelectorAll('a').forEach(function (link) {
      link.addEventListener('click', function () {
        mobileNav.classList.remove('open');
        hamburger.setAttribute('aria-expanded', 'false');
      });
    });
  }

  /* ─── Tab Switching ─── */
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabPanels = document.querySelectorAll('.tab-panel');

  tabBtns.forEach(function (btn) {
    btn.addEventListener('click', function () {
      var targetTab = btn.getAttribute('data-tab');

      tabBtns.forEach(function (b) {
        b.classList.remove('active');
        b.setAttribute('aria-selected', 'false');
      });
      tabPanels.forEach(function (p) {
        p.classList.remove('active');
      });

      btn.classList.add('active');
      btn.setAttribute('aria-selected', 'true');

      var panel = document.getElementById('tab-' + targetTab);
      if (panel) panel.classList.add('active');
    });

    // Keyboard navigation for tabs
    btn.addEventListener('keydown', function (e) {
      var btns = Array.from(tabBtns);
      var idx = btns.indexOf(btn);
      var next;

      if (e.key === 'ArrowRight') {
        next = btns[(idx + 1) % btns.length];
      } else if (e.key === 'ArrowLeft') {
        next = btns[(idx - 1 + btns.length) % btns.length];
      } else if (e.key === 'Home') {
        next = btns[0];
      } else if (e.key === 'End') {
        next = btns[btns.length - 1];
      }

      if (next) {
        e.preventDefault();
        next.focus();
        next.click();
      }
    });
  });

  /* ─── Animated Counters ─── */
  function animateCounter(el) {
    var target = parseInt(el.getAttribute('data-count'), 10);
    var prefix = el.getAttribute('data-prefix') || '';
    var suffix = el.getAttribute('data-suffix') || '';
    var duration = 1800;
    var start = performance.now();

    function easeOutExpo(t) {
      return t === 1 ? 1 : 1 - Math.pow(2, -10 * t);
    }

    function step(now) {
      var elapsed = now - start;
      var progress = Math.min(elapsed / duration, 1);
      var eased = easeOutExpo(progress);
      var current = Math.round(eased * target);

      el.textContent = prefix + current.toLocaleString() + suffix;

      if (progress < 1) {
        requestAnimationFrame(step);
      }
    }

    requestAnimationFrame(step);
  }

  // Observe hero stats
  var statValues = document.querySelectorAll('.stat-value[data-count]');
  var countersAnimated = false;

  if (statValues.length > 0 && 'IntersectionObserver' in window) {
    var counterObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting && !countersAnimated) {
          countersAnimated = true;
          statValues.forEach(animateCounter);
          counterObserver.disconnect();
        }
      });
    }, { threshold: 0.3 });

    statValues.forEach(function (el) {
      counterObserver.observe(el);
    });
  }

  /* ─── Scroll Reveals ─── */
  var reveals = document.querySelectorAll('.reveal');

  if (reveals.length > 0 && 'IntersectionObserver' in window) {
    var revealObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          revealObserver.unobserve(entry.target);
        }
      });
    }, {
      threshold: 0.08,
      rootMargin: '0px 0px -40px 0px'
    });

    reveals.forEach(function (el) {
      revealObserver.observe(el);
    });
  } else {
    // Fallback: show all
    reveals.forEach(function (el) {
      el.classList.add('visible');
    });
  }

  /* ─── BibTeX Copy ─── */
  var copyBtn = document.getElementById('copyBibtex');
  var bibtexBlock = document.getElementById('bibtex');

  if (copyBtn && bibtexBlock) {
    copyBtn.addEventListener('click', function () {
      // Get just the BibTeX text, not the button text
      var text = bibtexBlock.textContent.replace('Copy', '').trim();

      if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(function () {
          copyBtn.textContent = 'Copied!';
          setTimeout(function () {
            copyBtn.textContent = 'Copy';
          }, 2000);
        }).catch(function () {
          fallbackCopy(text);
        });
      } else {
        fallbackCopy(text);
      }
    });
  }

  function fallbackCopy(text) {
    var ta = document.createElement('textarea');
    ta.value = text;
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    try {
      document.execCommand('copy');
      if (copyBtn) {
        copyBtn.textContent = 'Copied!';
        setTimeout(function () {
          copyBtn.textContent = 'Copy';
        }, 2000);
      }
    } catch (e) {
      // Silently fail
    }
    document.body.removeChild(ta);
  }

  /* ─── Active Nav Highlight (optional) ─── */
  var navLinks = document.querySelectorAll('.nav-links a');
  var sections = [];

  navLinks.forEach(function (link) {
    var href = link.getAttribute('href');
    if (href && href.startsWith('#')) {
      var section = document.getElementById(href.slice(1));
      if (section) {
        sections.push({ link: link, section: section });
      }
    }
  });

  if (sections.length > 0 && 'IntersectionObserver' in window) {
    var navObserver = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          var id = entry.target.id;
          navLinks.forEach(function (l) {
            l.style.color = '';
          });
          sections.forEach(function (s) {
            if (s.section.id === id) {
              s.link.style.color = 'var(--color-text)';
            }
          });
        }
      });
    }, {
      threshold: 0.15,
      rootMargin: '-80px 0px -50% 0px'
    });

    sections.forEach(function (s) {
      navObserver.observe(s.section);
    });
  }

})();
