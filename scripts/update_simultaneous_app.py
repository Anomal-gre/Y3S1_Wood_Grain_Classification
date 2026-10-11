"""
Regenerate index.html and app.js with SIMULTANEOUS MULTI-MODEL INFERENCE.
Runs ResNet-18, Baseline SVM, ResNet-50, and ConvNeXt-Tiny concurrently side-by-side.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "hf_space_static"

def build():
    # Load data
    data_path = STATIC_DIR / "data.json"
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Update index.html
    html_content = """<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Nhận Diện Loài Gỗ Cận Cảnh & Chạy Đồng Thời Các Mô Hình | INT3401</title>
  <meta name="description" content="Chạy đồng thời và so sánh khả năng nhận diện 12 loài gỗ của ResNet-18, Baseline SVM, ResNet-50 và ConvNeXt-Tiny. BTL Học Máy INT3401 Lớp 11 - VNU UET.">
  <link rel="stylesheet" href="style.css">
  <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220%22%3E<text y=%2226%22 font-size=%2224%22>🪵</text></svg>">
</head>
<body>

  <!-- Navigation -->
  <header class="navbar">
    <div class="container nav-content">
      <a href="#" class="brand">
        <div class="brand-icon">🪵</div>
        <div>
          <span class="brand-title">WoodGrain-AI</span>
          <span class="brand-badge">INT3401 Lớp 11</span>
        </div>
      </a>
      <nav class="nav-links">
        <a href="#explorer">Khám Phá & So Sánh Mô Hình</a>
        <a href="#gradcam">Grad-CAM XAI</a>
        <a href="#benchmarks">Bảng Đối Chuẩn</a>
        <a href="https://huggingface.co/spaces/AnomalGreatad/wood-grain-classification" target="_blank" class="btn-hf-badge">
          <span>🤗 Space Live</span>
        </a>
      </nav>
    </div>
  </header>

  <!-- Hero Section -->
  <main class="container">
    <section class="hero">
      <div class="hero-tag">
        <span>🔬 Deep Transfer Learning & Multi-Model Comparative Evaluation</span>
      </div>
      <h1 class="hero-title">Đánh Giá & Nhận Diện Đồng Thời Các Mô Hình</h1>
      <p class="hero-subtitle">
        Hệ thống chạy <strong>song song đồng thời 4 mô hình</strong> (ResNet-18, Baseline LBP+SVM, ResNet-50, ConvNeXt-Tiny) trên cùng ảnh vân gỗ để so sánh độ chính xác, độ tự tin và thời gian phản hồi.
      </p>

      <!-- KPI Metrics Counter -->
      <div class="kpi-grid">
        <div class="kpi-card">
          <div class="kpi-value highlight" id="kpi-acc">92.74%</div>
          <div class="kpi-label">Độ Chính Xác (ResNet-18)</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-value highlight" id="kpi-f1">92.57%</div>
          <div class="kpi-label">Macro-F1 (Tập Test Độc Lập)</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-value">4 Mô Hình</div>
          <div class="kpi-label">Chạy Song Song Đồng Thời</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-value">12 Loài</div>
          <div class="kpi-label">Gỗ Tự Nhiên Châu Âu</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-value">8,544</div>
          <div class="kpi-label">Ảnh Thớ Gỗ (WOOD-AUTH)</div>
        </div>
      </div>
    </section>

    <!-- Interactive Explorer Section -->
    <section id="explorer" class="section">
      <div class="section-header">
        <div class="section-tag">Simultaneous Model Evaluation</div>
        <h2 class="section-title">Chạy Đồng Thời Tất Cả Mô Hình Nhận Diện</h2>
        <p class="section-desc">Chọn ảnh mẫu có sẵn hoặc tải ảnh từ máy tính — cả 4 mô hình sẽ cùng phân tích và đưa ra kết quả song song ngay bên dưới.</p>
      </div>

      <!-- Simultaneous Status Banner -->
      <div class="model-control-panel">
        <div style="display: flex; align-items: center; gap: 14px; flex-wrap: wrap;">
          <span style="font-size: 1.5rem;">⚡</span>
          <div>
            <div style="font-weight: 800; font-size: 1rem; color: var(--text-main);">Chế Độ: Chạy Đồng Thời Tất Cả 4 Mô Hình Nhận Diện</div>
            <div style="font-size: 0.85rem; color: var(--text-muted); margin-top: 2px;">
              Mỗi bức ảnh được phân tích đồng thời bởi: 
              <strong style="color: #34d399;">ResNet-18 (Chính)</strong> • 
              <strong style="color: #fbbf24;">Baseline LBP+SVM</strong> • 
              <strong style="color: #38bdf8;">ResNet-50</strong> • 
              <strong style="color: #c084fc;">ConvNeXt-Tiny</strong>
            </div>
          </div>
        </div>
        <div class="model-badge-info">
          <span>🟢 4 Mô Hình Đang Kích Hoạt Song Song</span>
        </div>
      </div>

      <!-- Source Navigation Tabs (Library vs Custom Upload) -->
      <div class="source-tabs-container">
        <button class="tab-btn active" id="tab-library-btn" onclick="switchSourceTab('library')">
          <span>📁 Thư Viện 12 Loài Kiểm Thử (Library Samples)</span>
        </button>
        <button class="tab-btn" id="tab-upload-btn" onclick="switchSourceTab('upload')">
          <span>📤 Tải Ảnh Của Bạn Lên (Upload Custom Image)</span>
        </button>
      </div>

      <!-- TAB 1: LIBRARY SAMPLES CONTAINER -->
      <div id="library-container">
        <!-- Species Selector Chips -->
        <div class="species-selector" id="species-selector-container">
          <!-- Populated dynamically via JS -->
        </div>

        <!-- Upper Grid: Image Viewer + Botanical Details -->
        <div class="explorer-grid">
          <!-- Left: Dual Image Viewer -->
          <div class="viewer-container">
            <div class="dual-image-frame" id="dual-frame">
              <span class="viewer-badge" id="viewer-badge">Ảnh Gốc ↔ Heatmap Grad-CAM</span>
              <img src="assets/samples/sample_1.jpg" alt="Wood Grain Base" class="viewer-base-img" id="viewer-base">
              <img src="assets/cams/cam_1.jpg" alt="Grad-CAM Overlay" class="viewer-cam-img" id="viewer-cam" style="opacity: 0.5;">
            </div>
            
            <div class="viewer-controls">
              <div class="slider-label">
                <span>Độ trong suốt Grad-CAM Heatmap</span>
                <span id="slider-val">50%</span>
              </div>
              <input type="range" min="0" max="100" value="50" class="slider-input" id="cam-slider">
            </div>
          </div>

          <!-- Right: Botanical Details -->
          <div class="details-container">
            <div>
              <div class="species-header-badge" id="detail-type-badge">Gỗ cứng (Hardwood)</div>
              <h3 class="species-title" id="detail-title">Gỗ Dẻ Gai Châu Âu</h3>
              <p class="species-scientific" id="detail-scientific">Fagus sylvatica (European Beech)</p>
            </div>

            <!-- Botanical & Anatomical Details -->
            <div class="facts-grid">
              <div class="fact-item">
                <div class="fact-label">Cấu trúc lỗ mạch</div>
                <div class="fact-val" id="fact-porosity">Lỗ mạch phân tán (Diffuse-porous)</div>
              </div>
              <div class="fact-item">
                <div class="fact-label">Khối lượng riêng</div>
                <div class="fact-val" id="fact-density">710 kg/m³</div>
              </div>
              <div class="fact-item fact-full">
                <div class="fact-label">Đặc trưng thớ gỗ vi mô</div>
                <div class="fact-val" id="fact-grain">Thớ mịn, thẳng, tia gỗ ngắn dạng đốm nâu li ti phân bố đều.</div>
              </div>
              <div class="fact-item fact-full">
                <div class="fact-label">Ứng dụng chế tác mộc</div>
                <div class="fact-val" id="fact-usage">Nội thất cao cấp, sàn gỗ, đồ chơi mộc, cán công cụ.</div>
              </div>
            </div>
          </div>
        </div>

        <!-- LOWER SECTION: SIMULTANEOUS 4-MODEL PREDICTION MATRIX FOR LIBRARY SAMPLE -->
        <div style="margin-top: 30px;">
          <!-- Consensus Banner -->
          <div class="consensus-banner">
            <div class="consensus-title">
              <span>🎯 ĐỘ ĐỒNG THUẬN GIỮA CÁC MÔ HÌNH:</span>
              <strong id="lib-consensus-text">4/4 Mô hình đồng thuận</strong>
            </div>
            <div class="consensus-badge" id="lib-consensus-species">Gỗ Dẻ Gai Châu Âu</div>
          </div>

          <!-- 4-Column Grid -->
          <div class="multi-model-grid" id="lib-multi-model-grid">
            <!-- Dynamically populated via JS -->
          </div>
        </div>
      </div>

      <!-- TAB 2: CUSTOM IMAGE UPLOAD CONTAINER -->
      <div id="upload-container" class="upload-container" style="display: none;">
        <!-- Left: Upload Area & Canvas -->
        <div class="viewer-container">
          <div class="upload-dropzone" id="dropzone" onclick="document.getElementById('custom-file-input').click()">
            <div class="upload-icon">📷</div>
            <div class="upload-title">Kéo thả ảnh vân gỗ vào đây</div>
            <div class="upload-hint">Hỗ trợ các định dạng JPG, JPEG, PNG, WEBP (Khuyên dùng ảnh chụp macro cận cảnh)</div>
            <button type="button" class="btn-browse">Duyệt File Từ Máy Tính</button>
            <input type="file" id="custom-file-input" accept="image/*" style="display: none;" onchange="handleFileUpload(event)">
          </div>

          <!-- Preview & Attention Heatmap Canvas -->
          <div class="dual-image-frame" id="upload-preview-frame" style="display: none;">
            <span class="viewer-badge" id="upload-badge">Ảnh Đã Tải Lên</span>
            <img id="upload-preview-img" class="viewer-base-img" src="" alt="Uploaded Wood">
            <canvas id="upload-cam-canvas" class="viewer-cam-img" style="opacity: 0.55; display: none;"></canvas>
          </div>

          <button type="button" class="btn-analyze" id="btn-run-inference" style="display: none;" onclick="analyzeUploadedImage()">
            <span>⚡ Chạy Đồng Thời Tất Cả Mô Hình Nhận Diện</span>
          </button>
        </div>

        <!-- Right: Upload Recognition Results -->
        <div class="details-container" id="upload-results-panel">
          <div id="upload-placeholder-msg" style="text-align: center; padding: 40px 20px; color: var(--text-muted);">
            <div style="font-size: 2.5rem; margin-bottom: 10px;">🪵</div>
            <h4 style="color: var(--text-main); margin-bottom: 6px;">Chưa có ảnh nào được tải lên</h4>
            <p style="font-size: 0.9rem;">Vui lòng chọn hoặc kéo thả một bức ảnh thớ gỗ từ máy tính để cả 4 mô hình cùng nhận diện song song.</p>
          </div>

          <!-- Active Result Content (Hidden until analysis) -->
          <div id="upload-active-results" style="display: none;">
            <div>
              <div class="species-header-badge" id="up-type-badge">Gỗ cứng (Hardwood)</div>
              <h3 class="species-title" id="up-title">Kết Quả Nhận Diện</h3>
              <p class="species-scientific" id="up-scientific">Tên khoa học</p>
            </div>

            <!-- Botanical Facts for Uploaded Match -->
            <div class="facts-grid" style="margin-top: 15px;">
              <div class="fact-item">
                <div class="fact-label">Cấu trúc lỗ mạch</div>
                <div class="fact-val" id="up-porosity">-</div>
              </div>
              <div class="fact-item">
                <div class="fact-label">Khối lượng riêng</div>
                <div class="fact-val" id="up-density">-</div>
              </div>
              <div class="fact-item fact-full">
                <div class="fact-label">Đặc trưng thớ gỗ nhận diện</div>
                <div class="fact-val" id="up-grain">-</div>
              </div>
              <div class="fact-item fact-full">
                <div class="fact-label">Ứng dụng chế tác lâm sản</div>
                <div class="fact-val" id="up-usage">-</div>
              </div>
            </div>
          </div>
        </div>

        <!-- LOWER SECTION FOR UPLOAD VIEW: SIMULTANEOUS 4-MODEL GRID -->
        <div id="upload-models-wrapper" style="grid-column: 1 / -1; display: none; margin-top: 20px;">
          <!-- Consensus Banner -->
          <div class="consensus-banner">
            <div class="consensus-title">
              <span>🎯 ĐỘ ĐỒNG THUẬN GIỮA CÁC MÔ HÌNH:</span>
              <strong id="up-consensus-text">Đang phân tích...</strong>
            </div>
            <div class="consensus-badge" id="up-consensus-species">-</div>
          </div>

          <!-- 4-Column Grid for Uploaded Image -->
          <div class="multi-model-grid" id="up-multi-model-grid">
            <!-- Dynamically populated via JS -->
          </div>
        </div>
      </div>
    </section>

    <!-- Grad-CAM Deep Dive -->
    <section id="gradcam" class="section">
      <div class="section-header">
        <div class="section-tag">Interpretability (XAI)</div>
        <h2 class="section-title">Giải Thích Quyết Định Bằng Grad-CAM</h2>
        <p class="section-desc">Mô hình ResNet-18 đã học được những đặc trưng sinh học nào? Grad-CAM trên lớp <code>layer4</code> chiếu sáng chính xác các tia gỗ (medullary rays) và mạch gỗ (vessels).</p>
      </div>

      <div class="figure-card">
        <div class="figure-img-wrap" onclick="openModal('assets/figures/gradcam_gallery.png')">
          <img src="assets/figures/gradcam_gallery.png" alt="Grad-CAM Gallery" loading="lazy">
        </div>
        <div class="figure-caption">
          <h4>Trực Quan Hóa Grad-CAM Trên 8 Mẫu Thử Nghiệm Đại Diện</h4>
          <p>Hàng trên hiển thị các mẫu phân loại chính xác: vùng đỏ rực tập trung dọc theo các tia gỗ phản quang (Gỗ Ti trăn, Sồi) và ranh giới sinh trưởng mùa xuân/thu (Gỗ Thông Douglas). Hàng dưới minh họa các trường hợp nhầm lẫn giữa các loài có cấu trúc vi mô tương đồng cao (Dẻ gai vs Sồi khi thiếu tia gỗ).</p>
        </div>
      </div>
    </section>

    <!-- Benchmarks & Evaluation Section -->
    <section id="benchmarks" class="section">
      <div class="section-header">
        <div class="section-tag">Empirical Results</div>
        <h2 class="section-title">Bảng Đối Chuẩn Toàn Diện Các Mô Hình Đã Huấn Luyện</h2>
        <p class="section-desc">Đánh giá thực nghiệm khách quan trên 2.836 ảnh Test độc lập không rò rỉ dữ liệu (Stratified Non-leaking Split).</p>
      </div>

      <!-- Comparative Figures -->
      <div class="figures-grid">
        <div class="figure-card">
          <div class="figure-img-wrap" onclick="openModal('assets/figures/resnet18_color_aug_confusion_matrix.png')">
            <img src="assets/figures/resnet18_color_aug_confusion_matrix.png" alt="Confusion Matrix" loading="lazy">
          </div>
          <div class="figure-caption">
            <h4>Ma Trận Nhầm Lẫn (Confusion Matrix - 12 Loài)</h4>
            <p>Đường chéo chính chiếm đa số tuyệt đối với độ chính xác trung bình 92.74%. Hầu hết các loài thông (Scots Pine, Douglas Fir, Larch) đạt độ nhạy trên 95%.</p>
          </div>
        </div>

        <div class="figure-card">
          <div class="figure-img-wrap" onclick="openModal('assets/figures/resnet18_color_aug_robustness_test.png')">
            <img src="assets/figures/resnet18_color_aug_robustness_test.png" alt="Robustness Benchmark" loading="lazy">
          </div>
          <div class="figure-caption">
            <h4>Kiểm Thử Độ Bền Vững (Color Shift Robustness)</h4>
            <p>Đánh giá tính bền bỉ trước 7 điều kiện ánh sáng môi trường thực tế (Grayscale, Desaturate, Hue ±0.2, Dark, Bright). Mô hình duy trì F1 > 78% ngay cả khi lệch màu mạnh.</p>
          </div>
        </div>

        <div class="figure-card">
          <div class="figure-img-wrap" onclick="openModal('assets/figures/resnet18_color_aug_learning_curves.png')">
            <img src="assets/figures/resnet18_color_aug_learning_curves.png" alt="Learning Curves" loading="lazy">
          </div>
          <div class="figure-caption">
            <h4>Đường Cong Học Hai Giai Đoạn (Two-Stage Training)</h4>
            <p>Giai đoạn 1 (Warm-up 5 epochs, đóng băng backbone) giúp hội tụ đầu phân loại ổn định. Giai đoạn 2 (Fine-tuning 10 epochs với lr thấp) tối ưu hóa toàn diện biểu diễn thớ gỗ.</p>
          </div>
        </div>
      </div>

      <!-- Full Comparison Table -->
      <div class="table-container">
        <table>
          <thead>
            <tr>
              <th>Phương Pháp / Mô Hình</th>
              <th>Chiến Lược Huấn Luyện</th>
              <th>Độ Chính Xác (Accuracy)</th>
              <th>Macro-F1 Score</th>
              <th>Độ Trễ CPU</th>
              <th>Đặc Điểm & Nhận Xét</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Baseline ML Cổ Điển: LBP + GLCM + RBF-SVM</strong></td>
              <td>Trích xuất 64 chiều texture + RBF Kernel</td>
              <td>80.99%</td>
              <td>78.92%</td>
              <td><strong>0.36 ms/ảnh</strong></td>
              <td>Tốc độ siêu nhanh nhưng độ chính xác thấp hơn -13.65% F1 do đặc trưng thủ công dễ bão hòa.</td>
            </tr>
            <tr>
              <td><strong>ResNet-18 (Giai đoạn 1 - Frozen Backbone)</strong></td>
              <td>Chỉ tinh chỉnh Head, giữ nguyên ImageNet</td>
              <td>86.10%</td>
              <td>85.40%</td>
              <td>14.73 ms/ảnh</td>
              <td>Trọng số ImageNet ban đầu chỉ nhận diện vật thể lớn, cần fine-tune để học cấu trúc vi mô.</td>
            </tr>
            <tr class="highlight-row">
              <td><strong>ResNet-18 Hai Giai Đoạn (Mô Hình Chính Đề Xuất)</strong></td>
              <td>Two-Stage + Color Augmentation</td>
              <td><strong>92.74%</strong></td>
              <td><strong>92.57%</strong></td>
              <td><strong>14.73 ms/ảnh</strong></td>
              <td><strong>Điểm tối ưu Pareto:</strong> Độ chính xác cao, kích thước nhẹ (43 MB), tốc độ thời gian thực trên CPU.</td>
            </tr>
            <tr>
              <td><strong>ResNet-50 (Thử nghiệm mạng sâu 50 layers)</strong></td>
              <td>Two-Stage + Color Augmentation</td>
              <td>93.41%</td>
              <td>93.15%</td>
              <td>32.40 ms/ảnh</td>
              <td>F1 tăng nhẹ +0.58% nhưng độ trễ CPU tăng hơn 2.2 lần (32.4 ms so với 14.7 ms).</td>
            </tr>
            <tr>
              <td><strong>ConvNeXt-Tiny (CNN Hiện Đại Thập Niên 2020)</strong></td>
              <td>Depthwise Sep Conv + 7x7 Kernels</td>
              <td>93.85%</td>
              <td>93.52%</td>
              <td>38.10 ms/ảnh</td>
              <td>Độ chính xác cao nhất (+0.95% F1) nhưng mô hình nặng và độ trễ CPU cao hơn 2.6 lần.</td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>
  </main>

  <!-- Image Zoom Modal -->
  <div class="modal" id="image-modal" onclick="closeModal()">
    <img src="" id="modal-img" alt="Zoomed View">
  </div>

  <!-- Footer -->
  <footer>
    <div class="container">
      <p><strong>BTL Học Máy (INT3401) – Lớp 11</strong> | Nhóm 11: Nghiên cứu phân loại kết cấu vân gỗ bằng Deep Transfer Learning.</p>
      <p style="margin-top: 6px;">Trường Đại học Công nghệ, Đại học Quốc gia Hà Nội (VNU - UET). Dữ liệu nghiên cứu: <a href="https://doi.org/10.5281/zenodo.2545611" target="_blank">WOOD-AUTH Benchmark (Barmpoutis 2018)</a>.</p>
    </div>
  </footer>

  <script src="app.js"></script>
</body>
</html>
"""
    (STATIC_DIR / "index.html").write_text(html_content, encoding="utf-8")
    print("  [+] Generated updated index.html")

if __name__ == "__main__":
    build()
