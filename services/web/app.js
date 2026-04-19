let events = [];
let cursor = 0;
let timer = null;

const editorEl = document.getElementById('editor');
const stdoutEl = document.getElementById('stdout');
const localsEl = document.getElementById('locals');
const stackEl = document.getElementById('call-stack');
const currentEventEl = document.getElementById('current-event');
const codeViewEl = document.getElementById('code-view');

function defaultCode(language) {
  if (language === 'javascript') {
    return `function add(a, b) {\n  const result = a + b;\n  console.log(result);\n}\nadd(2, 3);`;
  }
  return `def add(a, b):\n    result = a + b\n    print(result)\n\nadd(2, 3)`;
}

function escapeHtml(value) {
  return value
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}

function renderCodeView(activeLine) {
  const lines = editorEl.value.split('\n');
  codeViewEl.innerHTML = lines
    .map((line, index) => {
      const lineNo = index + 1;
      const activeClass = lineNo === activeLine ? 'code-line active' : 'code-line';
      return `<span class="${activeClass}">${lineNo.toString().padStart(2, '0')} | ${escapeHtml(line)}</span>`;
    })
    .join('');
}

function renderEvent(event) {
  currentEventEl.textContent = JSON.stringify(event, null, 2);
  renderCodeView(event.line || 0);
  if (event.call_stack) {
    stackEl.innerHTML = '';
    event.call_stack.forEach((item) => {
      const li = document.createElement('li');
      li.textContent = item;
      stackEl.appendChild(li);
    });
  }
  if (event.locals) localsEl.textContent = JSON.stringify(event.locals, null, 2);
  if (event.event === 'stdout' && event.stdout) stdoutEl.textContent += event.stdout + '\n';
  if (event.event === 'error' && event.message) stdoutEl.textContent += `[error] ${event.message}\n`;
}

function step() {
  if (cursor >= events.length) {
    if (timer) clearInterval(timer);
    timer = null;
    return;
  }
  renderEvent(events[cursor]);
  cursor += 1;
}

document.getElementById('language').addEventListener('change', (e) => {
  editorEl.value = defaultCode(e.target.value);
  renderCodeView(0);
});

document.getElementById('run').addEventListener('click', async () => {
  const language = document.getElementById('language').value;
  const code = editorEl.value;
  stdoutEl.textContent = '';
  localsEl.textContent = '{}';
  stackEl.innerHTML = '';
  currentEventEl.textContent = 'Running...';

  const res = await fetch(`${window.__API_BASE__ || 'http://localhost:8000'}/execute`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ language, code }),
  });
  const data = await res.json();
  events = data.events || [];
  cursor = 0;
  if (events.length) renderEvent(events[0]);
});

document.getElementById('step').addEventListener('click', step);
document.getElementById('play').addEventListener('click', () => {
  if (timer) return;
  timer = setInterval(step, 500);
});
document.getElementById('pause').addEventListener('click', () => {
  if (!timer) return;
  clearInterval(timer);
  timer = null;
});

editorEl.value = defaultCode('python');
renderCodeView(0);
