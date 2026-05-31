let selectedFile = null;
let selectedExample = null;

const fileInput     = document.getElementById('fileInput');
const modelSelect   = document.getElementById('modelSelect');
const beamSelect    = document.getElementById('beamSelect');
const runBtn        = document.getElementById('runBtn');
const examplesGrid  = document.getElementById('examplesGrid');
const imageWrap     = document.getElementById('imageWrap');
const resultText    = document.getElementById('resultText');

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

fileInput.addEventListener('change', () => {
  if (!fileInput.files.length) return;
  selectedFile = fileInput.files[0];
  selectedExample = null;
  clearExampleHighlight();
  showImage(URL.createObjectURL(selectedFile));
  resultText.textContent = '—';
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

init();
