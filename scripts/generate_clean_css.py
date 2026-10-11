"""
Generate style.css matching the ultra-clean, modern SaaS / AI design in the reference image:
- Light luminous background with vibrant blue/azure/cobalt gradient accents.
- Pure frosted glass cards with subtle borders and refined drop-shadows.
- Zero emojis: Clean SVG icons throughout.
- Precise, scientific model naming: ResNet-18, Baseline LBP + GLCM + SVM, ResNet-50, ConvNeXt-Tiny.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "hf_space_static"

css_content = """/* Modern Clean AI Platform Design System */
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;1,400&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
  /* Luminous Light Palette */
  --bg-page: #f8fafc;
  --bg-surface: #ffffff;
  --bg-card: rgba(255, 255, 255, 0.82);
  --bg-card-elevated: #ffffff;
  --bg-muted: #f1f5f9;
  
  /* Vibrant Electric Azure & Deep Cobalt Accent (from reference image) */
  --primary-blue: #1d4ed8;
  --electric-azure: #0ea5e9;
  --sky-glow: #38bdf8;
  --deep-navy: #0f172a;
  
  /* Text colors */
  --text-primary: #0f172a;
  --text-secondary: #475569;
  --text-muted: #64748b;
  --text-subtle: #94a3b8;
  
  /* Borders and Glass Effects */
  --border-light: rgba(226, 232, 240, 0.85);
  --border-glass: rgba(255, 255, 255, 0.9);
  --border-blue: rgba(14, 165, 233, 0.25);
  
  /* Radii */
  --radius-sm: 10px;
  --radius-md: 16px;
  --radius-lg: 24px;
  --radius-full: 9999px;
  
  /* Shadows */
  --shadow-subtle: 0 4px 6px -1px rgba(15, 23, 42, 0.04), 0 2px 4px -2px rgba(15, 23, 42, 0.03);
  --shadow-card: 0 20px 40px -12px rgba(2, 132, 199, 0.08), 0 1px 3px 0 rgba(15, 23, 42, 0.04);
  --shadow-blue-glow: 0 12px 30px -5px rgba(29, 78, 216, 0.25);
  
  --font-main: 'Be Vietnam Pro', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
  --font-mono: 'JetBrains Mono', monospace;
}

* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  background-color: var(--bg-page);
  color: var(--text-primary);
  font-family: var(--font-main);
  line-height: 1.6;
  min-height: 100vh;
  overflow-x: hidden;
  background-image: 
    radial-gradient(circle at 10% 5%, rgba(14, 165, 233, 0.12) 0%, transparent 45%),
    radial-gradient(circle at 90% 12%, rgba(29, 78, 216, 0.10) 0%, transparent 50%),
    radial-gradient(circle at 50% 60%, rgba(56, 189, 248, 0.06) 0%, transparent 55%),
    radial-gradient(circle at 20% 90%, rgba(2, 132, 199, 0.08) 0%, transparent 45%);
  background-attachment: fixed;
}

.container {
  max-width: 1240px;
  margin: 0 auto;
  padding: 0 24px;
}

/* Modern Minimal Navbar */
.navbar {
  position: sticky;
  top: 0;
  z-index: 100;
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  background: rgba(255, 255, 255, 0.78);
  border-bottom: 1px solid var(--border-light);
  padding: 14px 0;
}

.nav-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.brand {
  display: flex;
  align-items: center;
  gap: 12px;
  text-decoration: none;
  color: var(--text-primary);
}

.brand-icon-wrap {
  width: 42px;
  height: 42px;
  border-radius: 12px;
  background: linear-gradient(135deg, #1d4ed8 0%, #0ea5e9 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #ffffff;
  box-shadow: 0 6px 16px -2px rgba(14, 165, 233, 0.35);
}

.brand-title {
  font-weight: 800;
  font-size: 1.15rem;
  letter-spacing: -0.02em;
}

.brand-pill {
  font-size: 0.72rem;
  font-weight: 700;
  padding: 2px 8px;
  border-radius: var(--radius-full);
  background: rgba(14, 165, 233, 0.1);
  color: #0284c7;
  border: 1px solid rgba(14, 165, 233, 0.25);
  margin-left: 6px;
}

.nav-links {
  display: flex;
  gap: 22px;
  align-items: center;
}

.nav-links a {
  color: var(--text-secondary);
  text-decoration: none;
  font-size: 0.9rem;
  font-weight: 600;
  transition: color 0.2s ease;
}

.nav-links a:hover {
  color: var(--primary-blue);
}

.btn-space-link {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 16px;
  border-radius: var(--radius-full);
  background: var(--deep-navy);
  color: #ffffff;
  text-decoration: none;
  font-size: 0.85rem;
  font-weight: 600;
  transition: all 0.25s ease;
  box-shadow: 0 4px 12px rgba(15, 23, 42, 0.15);
}

.btn-space-link:hover {
  background: var(--primary-blue);
  transform: translateY(-1px);
  box-shadow: var(--shadow-blue-glow);
}

/* Hero Section */
.hero {
  padding: 55px 0 35px;
}

.hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid var(--border-light);
  box-shadow: var(--shadow-subtle);
  padding: 6px 14px;
  border-radius: var(--radius-full);
  font-size: 0.82rem;
  font-weight: 700;
  color: var(--primary-blue);
  margin-bottom: 20px;
}

.hero-title {
  font-size: 3rem;
  font-weight: 800;
  line-height: 1.15;
  letter-spacing: -0.03em;
  color: var(--text-primary);
  max-width: 820px;
  margin-bottom: 16px;
}

.hero-title .gradient-text {
  background: linear-gradient(135deg, #1d4ed8 0%, #0ea5e9 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

.hero-subtitle {
  font-size: 1.12rem;
  color: var(--text-secondary);
  max-width: 720px;
  line-height: 1.6;
  margin-bottom: 40px;
}

/* KPI Floating Stats Bar (inspired by bottom stats in reference image) */
.kpi-row {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 16px;
  margin-bottom: 45px;
}

.kpi-card {
  background: var(--bg-card);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid var(--border-glass);
  border-radius: var(--radius-md);
  padding: 18px 20px;
  box-shadow: var(--shadow-card);
  display: flex;
  align-items: center;
  gap: 16px;
  transition: all 0.25s ease;
}

.kpi-card:hover {
  transform: translateY(-2px);
  border-color: var(--border-blue);
  box-shadow: 0 20px 35px -10px rgba(14, 165, 233, 0.15);
}

.kpi-icon-pill {
  width: 44px;
  height: 44px;
  border-radius: 12px;
  background: rgba(14, 165, 233, 0.1);
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--primary-blue);
  flex-shrink: 0;
}

.kpi-num {
  font-size: 1.75rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: var(--text-primary);
  line-height: 1.1;
}

.kpi-desc {
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--text-muted);
  margin-top: 2px;
}

/* Main Section */
.section {
  padding: 30px 0;
}

.section-tag {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  color: #0284c7;
  font-size: 0.8rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 8px;
}

.section-title {
  font-size: 1.95rem;
  font-weight: 800;
  letter-spacing: -0.025em;
  color: var(--text-primary);
}

.section-desc {
  color: var(--text-secondary);
  font-size: 0.98rem;
  margin-top: 6px;
  max-width: 760px;
}

/* Mode Notification Banner */
.matrix-mode-banner {
  background: linear-gradient(135deg, rgba(29, 78, 216, 0.06) 0%, rgba(14, 165, 233, 0.08) 100%);
  border: 1px solid rgba(14, 165, 233, 0.25);
  border-radius: var(--radius-md);
  padding: 16px 22px;
  margin: 25px 0 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 14px;
}

.mode-banner-left {
  display: flex;
  align-items: center;
  gap: 14px;
}

.mode-icon-circle {
  width: 40px;
  height: 40px;
  border-radius: 50%;
  background: #ffffff;
  color: var(--primary-blue);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: var(--shadow-subtle);
  flex-shrink: 0;
}

.mode-title {
  font-weight: 800;
  font-size: 0.98rem;
  color: var(--text-primary);
}

.mode-sub {
  font-size: 0.84rem;
  color: var(--text-muted);
  margin-top: 2px;
}

.mode-status-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 0.78rem;
  font-weight: 700;
  padding: 5px 12px;
  border-radius: var(--radius-full);
  background: #ffffff;
  color: #0284c7;
  border: 1px solid rgba(14, 165, 233, 0.3);
  box-shadow: var(--shadow-subtle);
}

/* Clean Source Tabs (Library vs Upload) */
.source-tabs-bar {
  display: flex;
  gap: 10px;
  margin-bottom: 22px;
}

.source-tab-btn {
  background: rgba(255, 255, 255, 0.85);
  border: 1px solid var(--border-light);
  color: var(--text-secondary);
  padding: 10px 20px;
  border-radius: var(--radius-full);
  font-size: 0.88rem;
  font-weight: 700;
  cursor: pointer;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  transition: all 0.2s ease;
  box-shadow: var(--shadow-subtle);
}

.source-tab-btn:hover {
  background: #ffffff;
  color: var(--text-primary);
}

.source-tab-btn.active {
  background: var(--deep-navy);
  color: #ffffff;
  border-color: var(--deep-navy);
  box-shadow: 0 6px 18px rgba(15, 23, 42, 0.18);
}

/* Species Selector Chips */
.species-chips-track {
  display: flex;
  gap: 10px;
  overflow-x: auto;
  padding-bottom: 12px;
  margin-bottom: 22px;
  scrollbar-width: thin;
  scrollbar-color: var(--border-light) transparent;
}

.species-chip {
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  gap: 10px;
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-full);
  padding: 5px 16px 5px 6px;
  color: var(--text-secondary);
  cursor: pointer;
  font-size: 0.86rem;
  font-weight: 600;
  transition: all 0.2s ease;
  box-shadow: var(--shadow-subtle);
}

.species-chip img {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  object-fit: cover;
  border: 1px solid rgba(0, 0, 0, 0.08);
}

.species-chip:hover {
  background: #ffffff;
  color: var(--text-primary);
  border-color: rgba(14, 165, 233, 0.4);
}

.species-chip.active {
  background: #ffffff;
  border-color: #0284c7;
  color: #0284c7;
  font-weight: 700;
  box-shadow: 0 4px 14px rgba(14, 165, 233, 0.25);
}

/* Explorer Top Grid (Viewer + Botanical details) */
.explorer-top-grid {
  display: grid;
  grid-template-columns: 1fr 1.15fr;
  gap: 24px;
  background: var(--bg-card);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid var(--border-glass);
  border-radius: var(--radius-lg);
  padding: 26px;
  box-shadow: var(--shadow-card);
  margin-bottom: 28px;
}

@media (max-width: 900px) {
  .explorer-top-grid {
    grid-template-columns: 1fr;
  }
}

/* Image Viewer Frame */
.image-viewer-column {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.dual-image-frame {
  position: relative;
  width: 100%;
  aspect-ratio: 1 / 1;
  max-height: 420px;
  border-radius: var(--radius-md);
  overflow: hidden;
  border: 1px solid var(--border-light);
  background: #0f172a;
}

.viewer-base-img,
.viewer-cam-img {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.viewer-pill-badge {
  position: absolute;
  top: 14px;
  left: 14px;
  background: rgba(15, 23, 42, 0.75);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border: 1px solid rgba(255, 255, 255, 0.2);
  padding: 5px 12px;
  border-radius: var(--radius-full);
  font-size: 0.76rem;
  font-weight: 700;
  color: #ffffff;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  z-index: 5;
}

.opacity-slider-box {
  background: #ffffff;
  border: 1px solid var(--border-light);
  border-radius: var(--radius-md);
  padding: 12px 18px;
  display: flex;
  flex-direction: column;
  gap: 8px;
  box-shadow: var(--shadow-subtle);
}

.slider-info-row {
  display: flex;
  justify-content: space-between;
  font-size: 0.82rem;
  font-weight: 700;
  color: var(--text-secondary);
}

.slider-input {
  width: 100%;
  accent-color: var(--primary-blue);
  cursor: pointer;
}

/* Botanical Facts Column */
.facts-column {
  display: flex;
  flex-direction: column;
  gap: 18px;
}

.species-title-group {
  border-bottom: 1px solid var(--border-light);
  padding-bottom: 14px;
}

.type-pill {
  font-size: 0.74rem;
  font-weight: 800;
  text-transform: uppercase;
  color: #0284c7;
  letter-spacing: 0.06em;
}

.species-heading {
  font-size: 1.85rem;
  font-weight: 800;
  letter-spacing: -0.025em;
  color: var(--text-primary);
  margin-top: 2px;
}

.species-sub-sci {
  font-size: 0.95rem;
  font-style: italic;
  color: var(--text-muted);
}

.facts-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
}

.fact-tile {
  background: #ffffff;
  border: 1px solid var(--border-light);
  border-radius: var(--radius-sm);
  padding: 12px 14px;
  box-shadow: var(--shadow-subtle);
}

.fact-tile-full {
  grid-column: 1 / -1;
}

.fact-tile-label {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  font-weight: 800;
  color: var(--text-muted);
}

.fact-tile-val {
  font-size: 0.9rem;
  font-weight: 700;
  color: var(--text-primary);
  margin-top: 3px;
}

/* CONSENSUS BANNER */
.consensus-card {
  background: #ffffff;
  border: 1px solid var(--border-light);
  border-radius: var(--radius-md);
  padding: 14px 20px;
  margin-bottom: 18px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 12px;
  box-shadow: var(--shadow-subtle);
}

.consensus-left {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 0.92rem;
  color: var(--text-secondary);
}

.consensus-count-highlight {
  color: var(--primary-blue);
  font-weight: 800;
}

.consensus-badge-pill {
  font-size: 0.82rem;
  font-weight: 800;
  padding: 5px 14px;
  border-radius: var(--radius-full);
  background: rgba(14, 165, 233, 0.1);
  color: #0284c7;
  border: 1px solid rgba(14, 165, 233, 0.25);
}

/* 4-COLUMN SIMULTANEOUS MULTI-MODEL GRID */
.multi-model-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  margin-bottom: 40px;
}

@media (max-width: 1100px) {
  .multi-model-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

@media (max-width: 650px) {
  .multi-model-grid {
    grid-template-columns: 1fr;
  }
}

.model-column-card {
  background: var(--bg-card);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid var(--border-light);
  border-radius: var(--radius-lg);
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 14px;
  box-shadow: var(--shadow-card);
  transition: all 0.25s ease;
  position: relative;
}

.model-column-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 25px 45px -12px rgba(2, 132, 199, 0.15);
  border-color: rgba(14, 165, 233, 0.35);
}

.model-column-card.primary-highlight {
  border: 2px solid rgba(29, 78, 216, 0.5);
  background: #ffffff;
  box-shadow: 0 20px 40px -10px rgba(29, 78, 216, 0.12);
}

.model-header-row {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 8px;
}

.model-exact-name {
  font-size: 1.15rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  color: var(--text-primary);
}

.model-category-sub {
  font-size: 0.74rem;
  font-weight: 700;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-top: 2px;
}

.latency-pill {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 0.72rem;
  font-family: var(--font-mono);
  font-weight: 600;
  padding: 3px 8px;
  border-radius: var(--radius-full);
  background: var(--bg-muted);
  color: var(--text-secondary);
  border: 1px solid var(--border-light);
}

.prediction-highlight-box {
  background: #f8fafc;
  border: 1px solid var(--border-light);
  border-radius: var(--radius-md);
  padding: 12px 14px;
}

.pred-label-tiny {
  font-size: 0.7rem;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--text-muted);
}

.pred-wood-name {
  font-size: 1.1rem;
  font-weight: 800;
  color: var(--text-primary);
  margin-top: 2px;
}

.pred-conf-rate {
  font-size: 0.86rem;
  font-weight: 700;
  color: var(--primary-blue);
  margin-top: 3px;
}

.bars-block {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.bar-row {
  display: flex;
  flex-direction: column;
  gap: 3px;
}

.bar-labels {
  display: flex;
  justify-content: space-between;
  font-size: 0.76rem;
  font-weight: 600;
  color: var(--text-secondary);
}

.bar-track {
  width: 100%;
  height: 6px;
  background: #e2e8f0;
  border-radius: var(--radius-full);
  overflow: hidden;
}

.bar-fill {
  height: 100%;
  border-radius: var(--radius-full);
  background: linear-gradient(90deg, #1d4ed8 0%, #0ea5e9 100%);
  transition: width 0.35s ease;
}

.model-explanation-note {
  font-size: 0.8rem;
  color: var(--text-secondary);
  line-height: 1.45;
  margin-top: auto;
  border-top: 1px solid var(--border-light);
  padding-top: 10px;
}

/* CUSTOM UPLOAD VIEW */
.upload-view-card {
  display: none;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  background: var(--bg-card);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid var(--border-glass);
  border-radius: var(--radius-lg);
  padding: 26px;
  box-shadow: var(--shadow-card);
  margin-bottom: 28px;
}

@media (max-width: 900px) {
  .upload-view-card {
    grid-template-columns: 1fr;
  }
}

.upload-dropzone {
  border: 2px dashed rgba(14, 165, 233, 0.4);
  border-radius: var(--radius-md);
  padding: 40px 20px;
  text-align: center;
  cursor: pointer;
  transition: all 0.25s ease;
  background: #ffffff;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 280px;
}

.upload-dropzone:hover,
.upload-dropzone.dragover {
  border-color: var(--primary-blue);
  background: rgba(14, 165, 233, 0.04);
}

.upload-icon-circle {
  width: 60px;
  height: 60px;
  border-radius: 50%;
  background: rgba(14, 165, 233, 0.1);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #0284c7;
  margin-bottom: 14px;
}

.upload-title {
  font-size: 1.15rem;
  font-weight: 800;
  color: var(--text-primary);
  margin-bottom: 6px;
}

.upload-hint {
  font-size: 0.84rem;
  color: var(--text-muted);
  margin-bottom: 18px;
  max-width: 320px;
}

.btn-select-file {
  background: var(--deep-navy);
  color: #ffffff;
  padding: 9px 20px;
  border-radius: var(--radius-full);
  font-size: 0.86rem;
  font-weight: 700;
  border: none;
  cursor: pointer;
  transition: all 0.2s ease;
}

.btn-select-file:hover {
  background: var(--primary-blue);
}

.btn-run-all-models {
  width: 100%;
  margin-top: 14px;
  background: linear-gradient(135deg, #1d4ed8 0%, #0ea5e9 100%);
  color: #ffffff;
  border: none;
  border-radius: var(--radius-md);
  padding: 14px;
  font-size: 0.98rem;
  font-weight: 800;
  cursor: pointer;
  box-shadow: var(--shadow-blue-glow);
  transition: all 0.25s ease;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.btn-run-all-models:hover {
  transform: translateY(-2px);
  box-shadow: 0 16px 35px -5px rgba(29, 78, 216, 0.4);
}

/* Benchmark Section */
.figures-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
  gap: 22px;
  margin-top: 25px;
}

.figure-card {
  background: var(--bg-card);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid var(--border-glass);
  border-radius: var(--radius-md);
  overflow: hidden;
  box-shadow: var(--shadow-card);
  transition: transform 0.2s ease;
}

.figure-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 25px 45px -12px rgba(2, 132, 199, 0.15);
}

.figure-img-wrap {
  width: 100%;
  background: #ffffff;
  border-bottom: 1px solid var(--border-light);
  cursor: pointer;
}

.figure-img-wrap img {
  width: 100%;
  height: auto;
  display: block;
}

.figure-caption {
  padding: 16px 20px;
}

.figure-caption h4 {
  font-size: 1.05rem;
  font-weight: 800;
  color: var(--text-primary);
  margin-bottom: 6px;
}

.figure-caption p {
  font-size: 0.86rem;
  color: var(--text-secondary);
  line-height: 1.5;
}

/* Comparison Table */
.table-container {
  overflow-x: auto;
  background: var(--bg-card);
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  border: 1px solid var(--border-glass);
  border-radius: var(--radius-md);
  margin-top: 25px;
  box-shadow: var(--shadow-card);
}

table {
  width: 100%;
  border-collapse: collapse;
  text-align: left;
}

th, td {
  padding: 14px 20px;
  border-bottom: 1px solid var(--border-light);
  font-size: 0.9rem;
}

th {
  background: #f8fafc;
  font-weight: 800;
  color: var(--text-muted);
  text-transform: uppercase;
  font-size: 0.76rem;
  letter-spacing: 0.05em;
}

tr:last-child td {
  border-bottom: none;
}

tr.highlight-row {
  background: rgba(14, 165, 233, 0.06);
}

tr.highlight-row td {
  font-weight: 700;
  color: var(--primary-blue);
}

/* Modal */
.modal {
  display: none;
  position: fixed;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  background: rgba(15, 23, 42, 0.85);
  z-index: 1000;
  backdrop-filter: blur(12px);
  align-items: center;
  justify-content: center;
  padding: 20px;
}

.modal.active {
  display: flex;
}

.modal img {
  max-width: 90vw;
  max-height: 90vh;
  border-radius: var(--radius-md);
  box-shadow: 0 25px 60px rgba(0, 0, 0, 0.5);
}

/* Footer */
footer {
  border-top: 1px solid var(--border-light);
  padding: 40px 0;
  margin-top: 60px;
  text-align: center;
  color: var(--text-muted);
  font-size: 0.88rem;
  background: rgba(255, 255, 255, 0.6);
}

footer a {
  color: var(--primary-blue);
  text-decoration: none;
  font-weight: 600;
}

footer a:hover {
  text-decoration: underline;
}
"""

(STATIC_DIR / "style.css").write_text(css_content, encoding="utf-8")
print("[*] Generated updated style.css with clean azure SaaS design.")
