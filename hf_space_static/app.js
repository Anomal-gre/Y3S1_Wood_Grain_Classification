const names = ['Dẻ gai', 'Keo dậu hoa trắng', 'Anh đào dại', 'Dẻ ngọt', 'Thông Douglas', 'Thông rụng lá châu Âu', 'Sồi Anh', 'Thông Scots', 'Tiêu huyền phương Đông', 'Dương lá rung', 'Cây thiên đường', 'Óc chó'];
const tabs = [...document.querySelectorAll('.app-tab')];
function selectTab(index, focus = false) {
  tabs.forEach((tab, i) => { const active = i === index; tab.classList.toggle('active', active); tab.setAttribute('aria-selected', String(active)); tab.tabIndex = active ? 0 : -1; document.getElementById(tab.getAttribute('aria-controls')).hidden = !active; });
  if (focus) tabs[index].focus();
  window.scrollTo({top: 0, behavior: 'smooth'});
}
tabs.forEach((tab, i) => { tab.addEventListener('click', () => selectTab(i)); tab.addEventListener('keydown', e => { if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') { e.preventDefault(); selectTab(1 - i, true); } }); });
const grid = document.getElementById('sample-grid');
names.forEach((name, i) => { const card = document.createElement('div'); card.className = 'sample-card'; const img = document.createElement('img'); img.src = `assets/samples/sample_${i + 1}.jpg`; img.alt = `Vân gỗ ${name}`; img.loading = 'lazy'; const caption = document.createElement('span'); caption.textContent = name; card.append(img, caption); grid.append(card); });
function openModal(src) { document.getElementById('modal-img').src = src; document.getElementById('image-modal').classList.add('active'); }
function closeModal() { document.getElementById('image-modal').classList.remove('active'); }
const input = document.getElementById('wood-image'), drop = document.getElementById('drop-area'), preview = document.getElementById('image-preview'), status = document.getElementById('result-status'), results = document.getElementById('result-list');
let sessionPromise, selection = 0;
function session() { if (!window.ort) return Promise.reject(new Error('Không tải được thư viện nhận diện. Hãy kiểm tra kết nối mạng.')); if (!sessionPromise) sessionPromise = ort.InferenceSession.create('assets/wood_resnet18.onnx', { executionProviders: ['wasm'] }).catch(error => { sessionPromise = null; throw error; }); return sessionPromise; }
async function classify(file) {
  const id = ++selection; results.replaceChildren(); document.getElementById('model-top-name').textContent = 'Chưa có kết quả'; document.getElementById('model-top-score').textContent = 'Đang chờ dự đoán'; document.getElementById('model-predictions').textContent = 'Đang chờ dự đoán.';
  if (!file || !['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) { status.textContent = 'Vui lòng chọn ảnh JPG, PNG hoặc WEBP.'; return; }
  status.textContent = 'Đang tải mô hình và phân tích ảnh. Lần đầu có thể mất thêm thời gian…';
  const url = URL.createObjectURL(file); const image = new Image();
  try {
    await new Promise((resolve, reject) => { image.onload = resolve; image.onerror = () => reject(new Error('Không đọc được ảnh. Vui lòng chọn ảnh khác.')); image.src = url; });
    if (id !== selection) return;
    preview.src = url; document.getElementById('preview-wrap').hidden = false; drop.hidden = true;
    const canvas = document.createElement('canvas'); canvas.width = canvas.height = 256; const ctx = canvas.getContext('2d', {willReadFrequently: true}); ctx.drawImage(image, 0, 0, 256, 256);
    const pixels = ctx.getImageData(16, 16, 224, 224).data, data = new Float32Array(3 * 224 * 224), mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225];
    for (let i = 0; i < 224 * 224; i++) for (let c = 0; c < 3; c++) data[c * 224 * 224 + i] = (pixels[i * 4 + c] / 255 - mean[c]) / std[c];
    const model = await session(); const output = await model.run({[model.inputNames[0]]: new ort.Tensor('float32', data, [1, 3, 224, 224])}); if (id !== selection) return;
    const logits = Array.from(output[model.outputNames[0]].data), max = Math.max(...logits), scores = logits.map(x => Math.exp(x - max)), sum = scores.reduce((a,b) => a+b, 0);
    const ranked = scores.map((value, i) => ({name:names[i], value:value/sum})).sort((a,b) => b.value-a.value).slice(0,3);
    document.getElementById('model-top-name').textContent = ranked[0].name; document.getElementById('model-top-score').textContent = `Điểm dự đoán: ${(ranked[0].value * 100).toFixed(1)}%`; document.getElementById('model-predictions').replaceChildren(); document.getElementById('result-title').textContent = ranked[0].name; status.textContent = '3 dự đoán có điểm cao nhất · ResNet-18';
    ranked.forEach(item => { const row = document.createElement('div'); row.className = 'prediction-row'; const label = document.createElement('div'); label.className = 'prediction-label'; const name = document.createElement('span'); name.textContent = item.name; const score = document.createElement('strong'); score.textContent = `${(item.value*100).toFixed(1)}%`; label.append(name,score); const bar = document.createElement('div'); bar.className = 'prediction-bar'; const fill = document.createElement('div'); fill.style.width = `${item.value*100}%`; bar.append(fill); row.append(label,bar); results.append(row); document.getElementById('model-predictions').append(row.cloneNode(true)); });
  } catch (error) { if (id === selection) { status.textContent = `Không thể nhận diện ảnh: ${error.message}`; document.getElementById('model-top-score').textContent = 'Không thể phân tích ảnh.'; document.getElementById('model-predictions').textContent = 'Chưa có dự đoán.'; } }
}
input.addEventListener('change', () => classify(input.files[0]));
['dragenter','dragover'].forEach(type => drop.addEventListener(type, e => { e.preventDefault(); drop.classList.add('dragging'); }));
['dragleave','drop'].forEach(type => drop.addEventListener(type, e => { e.preventDefault(); drop.classList.remove('dragging'); }));
drop.addEventListener('drop', e => classify(e.dataTransfer.files[0]));

drop.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
input.addEventListener('click', () => { input.value = ''; });
