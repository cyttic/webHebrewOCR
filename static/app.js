let selectedFile = null;
let selectedExample = null;

const fileInput     = document.getElementById('fileInput');
const modelSelect   = document.getElementById('modelSelect');
const beamSelect    = document.getElementById('beamSelect');
const runBtn        = document.getElementById('runBtn');
const examplesGrid  = document.getElementById('examplesGrid');
const imageWrap     = document.getElementById('imageWrap');
const previewCol    = document.querySelector('.preview-col');
const resultText    = document.getElementById('resultText');
const drawCanvas    = document.getElementById('drawCanvas');
const clearBtn      = document.getElementById('clearBtn');
const analyzeBtn    = document.getElementById('analyzeBtn');

async function init() {
  // models
  const m = await fetch('/api/models').then(r => r.json());
  modelSelect.innerHTML = '';
  if (!m.models.length) {
    modelSelect.appendChild(new Option('(no models found)', ''));
  } else {
    m.models.forEach(name => modelSelect.appendChild(new Option(name, name)));
  }

  // examples
  const e = await fetch('/api/examples').then(r => r.json());
  examplesGrid.innerHTML = '';
  e.examples.forEach(fn => {
    const img = document.createElement('img');
    img.src = '/images/' + encodeURIComponent(fn);
    img.title = fn;
    img.className = 'example-thumb';
    img.addEventListener('click', () => selectExample(fn, img));
    examplesGrid.appendChild(img);
  });
}

function showImage(src) {
  imageWrap.innerHTML = '';
  const img = document.createElement('img');
  img.src = src;
  imageWrap.appendChild(img);
}

function clearExampleHighlight() {
  document.querySelectorAll('.example-thumb.selected')
          .forEach(t => t.classList.remove('selected'));
}

function setFile(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) { alert('Please choose an image file.'); return; }
  selectedFile = file;
  selectedExample = null;
  fileInput.value = '';
  clearExampleHighlight();
  showImage(URL.createObjectURL(file));
  resultText.textContent = '—';
}

fileInput.addEventListener('change', () => setFile(fileInput.files[0]));

// drag-and-drop onto the preview column
['dragenter', 'dragover'].forEach(ev =>
  previewCol.addEventListener(ev, e => {
    e.preventDefault();
    previewCol.classList.add('dragover');
  }));
['dragleave', 'dragend', 'drop'].forEach(ev =>
  previewCol.addEventListener(ev, e => {
    e.preventDefault();
    previewCol.classList.remove('dragover');
  }));
previewCol.addEventListener('drop', e => {
  const files = e.dataTransfer && e.dataTransfer.files;
  if (files && files.length) setFile(files[0]);
});

function selectExample(fn, thumb) {
  selectedExample = fn;
  selectedFile = null;
  fileInput.value = '';
  clearExampleHighlight();
  thumb.classList.add('selected');
  showImage('/images/' + encodeURIComponent(fn));
  resultText.textContent = '—';
}

runBtn.addEventListener('click', async () => {
  if (!selectedFile && !selectedExample) { alert('Choose an image or an example first.'); return; }
  const model = modelSelect.value;
  if (!model) { alert('No model available.'); return; }

  const fd = new FormData();
  fd.append('model', model);
  fd.append('beams', beamSelect.value);
  if (selectedFile) fd.append('file', selectedFile);
  else fd.append('example', selectedExample);

  runBtn.disabled = true;
  resultText.textContent = 'Running…';
  try {
    const res = await fetch('/api/ocr', { method: 'POST', body: fd });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      resultText.textContent = 'Error: ' + (err.detail || res.status);
    } else {
      const data = await res.json();
      resultText.textContent = data.text || '(empty)';
    }
  } catch (err) {
    resultText.textContent = 'Error: ' + err.message;
  } finally {
    runBtn.disabled = false;
  }
});

// ── simple drawing canvas ─────────────────────────────────────────────
const ctx = drawCanvas.getContext('2d');
let drawing = false;

function initCanvas() {
  // internal resolution matches displayed size; white background (RGB-safe for the model)
  drawCanvas.width  = drawCanvas.clientWidth;
  drawCanvas.height = drawCanvas.clientHeight;
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
  ctx.strokeStyle = '#000';
  ctx.lineWidth = 5;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
}

function canvasPos(pt) {
  const r = drawCanvas.getBoundingClientRect();
  return {
    x: (pt.clientX - r.left) * (drawCanvas.width  / r.width),
    y: (pt.clientY - r.top)  * (drawCanvas.height / r.height),
  };
}

function startDraw(pt) { drawing = true; const p = canvasPos(pt); ctx.beginPath(); ctx.moveTo(p.x, p.y); }
function moveDraw(pt)  { if (!drawing) return; const p = canvasPos(pt); ctx.lineTo(p.x, p.y); ctx.stroke(); }
function endDraw()     { drawing = false; }

drawCanvas.addEventListener('mousedown', e => startDraw(e));
drawCanvas.addEventListener('mousemove', e => moveDraw(e));
window.addEventListener('mouseup', endDraw);
drawCanvas.addEventListener('touchstart', e => { e.preventDefault(); startDraw(e.touches[0]); });
drawCanvas.addEventListener('touchmove',  e => { e.preventDefault(); moveDraw(e.touches[0]); });
window.addEventListener('touchend', endDraw);

clearBtn.addEventListener('click', initCanvas);

// move the drawing into the preview, ready for "Run OCR"
analyzeBtn.addEventListener('click', () => {
  drawCanvas.toBlob(blob => {
    if (!blob) return;
    const file = new File([blob], 'drawing.png', { type: 'image/png' });
    setFile(file);   // same path as upload/drop -> shows in preview, ready to Run
  }, 'image/png');
});

initCanvas();
init();
