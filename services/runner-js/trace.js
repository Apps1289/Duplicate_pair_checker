const fs = require('fs');
const vm = require('vm');

function cloneVars(sandbox) {
  const vars = {};
  for (const [k, v] of Object.entries(sandbox)) {
    if (k.startsWith('__') || k === 'console') continue;
    if (typeof v === 'function') {
      vars[k] = `[Function ${v.name || 'anonymous'}]`;
    } else {
      try {
        vars[k] = JSON.stringify(v);
      } catch {
        vars[k] = String(v);
      }
    }
  }
  return vars;
}

function instrument(code) {
  const lines = code.split('\n');
  return lines
    .map((line, idx) => {
      const n = idx + 1;
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith('//')) return line;
      const indent = line.match(/^\s*/)[0];
      return `${indent}__trace.line(${n});\n${line}`;
    })
    .join('\n');
}

const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const events = [];
const sandbox = {
  console: {
    log: (...args) => {
      events.push({ event: 'stdout', stdout: args.map(String).join(' ') });
    },
  },
};

let previous = {};
sandbox.__trace = {
  line: (lineNumber) => {
    const stackText = new Error().stack || '';
    const frames = stackText
      .split('\n')
      .slice(2)
      .map((v) => v.trim().replace(/^at\s+/, ''));
    const locals = cloneVars(sandbox);
    events.push({ event: 'line_exec', line: lineNumber, call_stack: frames, locals });
    for (const [name, value] of Object.entries(locals)) {
      if (previous[name] !== value) {
        events.push({ event: 'var_update', line: lineNumber, locals: { [name]: value } });
      }
    }
    previous = locals;
  },
};

try {
  vm.createContext(sandbox);
  vm.runInContext(instrument(input.code || ''), sandbox, { timeout: 2000 });
  events.push({ event: 'end', message: 'Execution finished' });
} catch (err) {
  events.push({ event: 'error', message: String(err.message || err) });
}

process.stdout.write(JSON.stringify({ events }));
