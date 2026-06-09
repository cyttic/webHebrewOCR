const fileInput     = document.getElementById('fileInput');
const modelSelect   = document.getElementById('modelSelect');
const beamSelect    = document.getElementById('beamSelect');
const runBtn        = document.getElementById('runBtn');
const examplesGrid  = document.getElementById('examplesGrid');
const previewCol    = document.querySelector('.preview-col');
const resultText    = document.getElementById('resultText');
const drawCanvas    = document.getElementById('drawCanvas');
const canvasHint    = document.getElementById('canvasHint');
const clearBtn      = document.getElementById('clearBtn');
const penColor      = document.getElementById('penColor');
const penSize       = document.getElementById('penSize');

const ctx = drawCanvas.getContext('2d');
let drawing = false;
let hasContent = false;   // true once an image is loaded or a stroke is drawn
let _sourceFile    = null;   // File object if user uploaded a file
let _sourceExample = null;   // example filename if user picked an example
let _drawBounds    = null;   // {minX, minY, maxX, maxY} of strokes drawn on the canvas

function resetDrawBounds() { _drawBounds = null; }

function expandDrawBounds(p) {
  const half = parseInt(penSize.value, 10) / 2;
  const x0 = p.x - half, y0 = p.y - half, x1 = p.x + half, y1 = p.y + half;
  if (!_drawBounds) {
    _drawBounds = { minX: x0, minY: y0, maxX: x1, maxY: y1 };
  } else {
    _drawBounds.minX = Math.min(_drawBounds.minX, x0);
    _drawBounds.minY = Math.min(_drawBounds.minY, y0);
    _drawBounds.maxX = Math.max(_drawBounds.maxX, x1);
    _drawBounds.maxY = Math.max(_drawBounds.maxY, y1);
  }
}

async function init() {
  // models
  const MODEL_LABELS = {
    'first_iteration':              'exp 1, human',
    'trocr-hebrew-synthetic-cont':  'exp 2, synt',
    'pre_finetune_unfrozen':        'exp 3, synt',
    'trocr-hebrew-matan-exp7':      'exp 7, matan',
  };
  const MODEL_ORDER = ['first_iteration', 'trocr-hebrew-synthetic-cont', 'pre_finetune_unfrozen', 'trocr-hebrew-matan-exp7'];

  const m = await fetch('/api/models').then(r => r.json());
  modelSelect.innerHTML = '';
  if (!m.models.length) {
    modelSelect.appendChild(new Option('(no models found)', ''));
  } else {
    const known  = MODEL_ORDER.filter(name => m.models.includes(name));
    const others = m.models.filter(name => !MODEL_ORDER.includes(name));
    [...known, ...others].forEach(name =>
      modelSelect.appendChild(new Option(MODEL_LABELS[name] || name, name)));
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

// ── canvas: shared preview + drawing surface ──────────────────────────
function paintWhite() {
  drawCanvas.width  = drawCanvas.clientWidth;
  drawCanvas.height = drawCanvas.clientHeight;
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
}

function clearCanvas() {
  paintWhite();
  hasContent = false;
  _sourceFile    = null;
  _sourceExample = null;
  resetDrawBounds();
  if (canvasHint) canvasHint.style.display = '';   // show placeholder again
  resultText.textContent = '—';
}

// draw a selected/dropped image as the canvas background (contain-fit, centered)
function loadImageOntoCanvas(src) {
  const img = new Image();
  img.onload = () => {
    paintWhite();
    const cw = drawCanvas.width, ch = drawCanvas.height;
    const scale = Math.min(cw / img.width, ch / img.height);
    const dw = img.width * scale, dh = img.height * scale;
    ctx.drawImage(img, (cw - dw) / 2, (ch - dh) / 2, dw, dh);
    hasContent = true;
    if (canvasHint) canvasHint.style.display = 'none';
    resultText.textContent = '—';
  };
  img.src = src;
}

function clearExampleHighlight() {
  document.querySelectorAll('.example-thumb.selected')
          .forEach(t => t.classList.remove('selected'));
}

function setFile(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) { alert('Please choose an image file.'); return; }
  fileInput.value = '';
  clearExampleHighlight();
  _sourceFile    = file;
  _sourceExample = null;
  resetDrawBounds();
  loadImageOntoCanvas(URL.createObjectURL(file));
}

function selectExample(fn, thumb) {
  fileInput.value = '';
  clearExampleHighlight();
  thumb.classList.add('selected');
  _sourceFile    = null;
  _sourceExample = fn;
  resetDrawBounds();
  loadImageOntoCanvas('/images/' + encodeURIComponent(fn));
}

fileInput.addEventListener('change', () => setFile(fileInput.files[0]));

// drag-and-drop onto the preview area
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

// ── drawing (only while the pen is pressed down) ──────────────────────
function canvasPos(pt) {
  const r = drawCanvas.getBoundingClientRect();
  return {
    x: (pt.clientX - r.left) * (drawCanvas.width  / r.width),
    y: (pt.clientY - r.top)  * (drawCanvas.height / r.height),
  };
}

function startDraw(pt) {
  drawing = true;
  hasContent = true;
  _sourceFile    = null;
  _sourceExample = null;
  if (canvasHint) canvasHint.style.display = 'none';
  ctx.strokeStyle = penColor.value;
  ctx.lineWidth = parseInt(penSize.value, 10);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  const p = canvasPos(pt);
  ctx.beginPath();
  ctx.moveTo(p.x, p.y);
  expandDrawBounds(p);
}
function moveDraw(pt) {
  if (!drawing) return;
  const p = canvasPos(pt);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
  expandDrawBounds(p);
}
function endDraw()    { drawing = false; }

drawCanvas.addEventListener('mousedown', e => startDraw(e));
drawCanvas.addEventListener('mousemove', e => moveDraw(e));
window.addEventListener('mouseup', endDraw);
drawCanvas.addEventListener('touchstart', e => { e.preventDefault(); startDraw(e.touches[0]); });
drawCanvas.addEventListener('touchmove',  e => { e.preventDefault(); moveDraw(e.touches[0]); });
window.addEventListener('touchend', endDraw);

clearBtn.addEventListener('click', () => { clearExampleHighlight(); clearCanvas(); });

// ── run OCR on whatever is on the canvas (image + drawing) ────────────
runBtn.addEventListener('click', () => {
  if (!hasContent) { alert('Draw something or choose an image first.'); return; }
  const model = modelSelect.value;
  if (!model) { alert('No model available.'); return; }

  async function sendOcr(fd) {
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
  }

  if (_sourceExample) {
    const fd = new FormData();
    fd.append('model', model);
    fd.append('beams', beamSelect.value);
    fd.append('example', _sourceExample);
    sendOcr(fd);
  } else if (_sourceFile) {
    const fd = new FormData();
    fd.append('model', model);
    fd.append('beams', beamSelect.value);
    fd.append('file', _sourceFile);
    sendOcr(fd);
  } else {
    let sourceCanvas = drawCanvas;
    if (_drawBounds) {
      const pad = 6;
      const x = Math.max(0, Math.floor(_drawBounds.minX - pad));
      const y = Math.max(0, Math.floor(_drawBounds.minY - pad));
      const w = Math.min(drawCanvas.width,  Math.ceil(_drawBounds.maxX + pad)) - x;
      const h = Math.min(drawCanvas.height, Math.ceil(_drawBounds.maxY + pad)) - y;
      if (w > 0 && h > 0) {
        const cropped = document.createElement('canvas');
        cropped.width = w;
        cropped.height = h;
        cropped.getContext('2d').drawImage(drawCanvas, x, y, w, h, 0, 0, w, h);
        sourceCanvas = cropped;
      }
    }
    sourceCanvas.toBlob((blob) => {
      if (!blob) return;
      const fd = new FormData();
      fd.append('model', model);
      fd.append('beams', beamSelect.value);
      fd.append('file', new File([blob], 'canvas.png', { type: 'image/png' }));
      sendOcr(fd);
    }, 'image/png');
  }
});

clearCanvas();
init();
