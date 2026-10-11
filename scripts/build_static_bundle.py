"""
Build static bundle: app.js, README.md with HF metadata frontmatter.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "hf_space_static"

def build():
    data_path = STATIC_DIR / "data.json"
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Generate app.js with embedded data and fallback fetch
    js_template = f"""// Wood Grain Classifier Interactive Showcase Script
const EMBEDDED_DATA = {json.dumps(data, ensure_ascii=False, indent=2)};

let appData = EMBEDDED_DATA;
let currentSampleIndex = 0;

document.addEventListener('DOMContentLoaded', () => {{
  initApp();
}});

function initApp() {{
  const container = document.getElementById('species-selector-container');
  if (!container || !appData.samples) return;

  container.innerHTML = '';
  appData.samples.forEach((sample, idx) => {{
    const chip = document.createElement('button');
    chip.className = `species-chip ${{idx === 0 ? 'active' : ''}}`;
    chip.id = `chip-${{sample.id}}`;
    chip.innerHTML = `
      <img src="assets/samples/${{sample.filename}}" alt="${{sample.species.common_vi}}">
      <span>${{sample.id}}. ${{sample.species.common_vi}}</span>
    `;
    chip.addEventListener('click', () => {{
      selectSpecies(idx);
    }});
    container.appendChild(chip);
  }});

  // Slider listener
  const slider = document.getElementById('cam-slider');
  const sliderVal = document.getElementById('slider-val');
  const camImg = document.getElementById('viewer-cam');
  
  if (slider && camImg) {{
    slider.addEventListener('input', (e) => {{
      const val = e.target.value;
      sliderVal.textContent = `${{val}}%`;
      camImg.style.opacity = val / 100;
    }});
  }}

  // Load first species by default
  selectSpecies(0);
}}

function selectSpecies(idx) {{
  currentSampleIndex = idx;
  const sample = appData.samples[idx];
  if (!sample) return;

  // Update active chip
  document.querySelectorAll('.species-chip').forEach((c, i) => {{
    if (i === idx) c.classList.add('active');
    else c.classList.remove('active');
  }});

  // Update viewer images
  const baseImg = document.getElementById('viewer-base');
  const camImg = document.getElementById('viewer-cam');
  if (baseImg) baseImg.src = `assets/samples/${{sample.filename}}`;
  if (camImg) camImg.src = `assets/cams/${{sample.cam_filename}}`;

  // Update text details
  const typeBadge = document.getElementById('detail-type-badge');
  const title = document.getElementById('detail-title');
  const scientific = document.getElementById('detail-scientific');
  const porosity = document.getElementById('fact-porosity');
  const density = document.getElementById('fact-density');
  const grain = document.getElementById('fact-grain');
  const usage = document.getElementById('fact-usage');

  if (typeBadge) typeBadge.textContent = sample.species.type;
  if (title) title.textContent = `${{sample.id}}. ${{sample.species.common_vi}}`;
  if (scientific) scientific.textContent = `${{sample.species.scientific}} (${{sample.species.common_en}})`;
  if (porosity) porosity.textContent = sample.species.porosity;
  if (density) density.textContent = sample.species.density;
  if (grain) grain.textContent = sample.species.grain;
  if (usage) usage.textContent = sample.species.usage;

  // Update probabilities
  const probsContainer = document.getElementById('probs-list');
  if (probsContainer) {{
    probsContainer.innerHTML = '';
    sample.predictions.slice(0, 3).forEach((pred) => {{
      const pct = (pred.probability * 100).toFixed(1);
      const row = document.createElement('div');
      row.className = 'prob-row';
      row.innerHTML = `
        <div class="prob-info">
          <span>${{pred.class_id}}. ${{pred.name}}</span>
          <span>${{pct}}%</span>
        </div>
        <div class="prob-bar-bg">
          <div class="prob-bar-fill" style="width: ${{pct}}%"></div>
        </div>
      `;
      probsContainer.appendChild(row);
    }});
  }}
}}

function openModal(imgSrc) {{
  const modal = document.getElementById('image-modal');
  const modalImg = document.getElementById('modal-img');
  if (modal && modalImg) {{
    modalImg.src = imgSrc;
    modal.classList.add('active');
  }}
}}

function closeModal() {{
  const modal = document.getElementById('image-modal');
  if (modal) modal.classList.remove('active');
}}
"""
    (STATIC_DIR / "app.js").write_text(js_template, encoding="utf-8")
    print("  [+] Generated app.js")

    # 2. Generate README.md with Hugging Face static metadata frontmatter
    readme_content = """---
title: Wood Grain Species Classification INT3401
emoji: 🪵
colorFrom: green
colorTo: yellow
sdk: static
pinned: false
license: mit
---

# 🪵 Hệ Thống Nhận Diện Loài Gỗ Cận Cảnh (Wood Grain Classification)

Showcase tương tác kết quả **Bài tập lớn môn Học Máy (INT3401) - Lớp 11**, Trường Đại học Công nghệ (VNU-UET).

## 🔬 Thông Tin Nghiên Cứu
- **Mô hình đề xuất:** ResNet-18 Deep Transfer Learning (Two-Stage Training: Head Warm-up + Fine-tuning)
- **Tăng cường dữ liệu:** Thích ứng biến đổi màu sắc (Color Augmentation) chống trượt phân phối ánh sáng
- **Giải thích quyết định:** Grad-CAM (Gradient-weighted Class Activation Mapping) trên lớp tích chập cuối `layer4`
- **Kết quả thực nghiệm:**
  - **Test Accuracy:** 92.74%
  - **Macro-F1:** 92.57%
  - **Độ trễ suy luận (CPU):** 14.73 ms / ảnh
- **Tập dữ liệu:** Zenodo DOI 10.5281/zenodo.2545611 (8,544 ảnh RGB thớ gỗ, 12 loài gỗ châu Âu)
"""
    (STATIC_DIR / "README.md").write_text(readme_content, encoding="utf-8")
    print("  [+] Generated README.md")

if __name__ == "__main__":
    build()
