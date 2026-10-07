import { MATERIAL_LIST } from '../sim/materials';
import type { SimConfig } from '../sim/types';

interface VesselDef {
  id: string;
  label: string;
}

const VESSELS: VesselDef[] = [
  { id: 'liner', label: 'Liner' },
  { id: 'freighter', label: 'Freighter' },
  { id: 'tug', label: 'Tug' },
  { id: 'barge', label: 'Barge' },
  { id: 'raft', label: 'Raft' },
  { id: 'craneBarge', label: 'Crane' },
  { id: 'iceberg', label: 'Iceberg' },
];

const PARTICLE_MIN = 1000;
const PARTICLE_MAX = 120_000;

function section(title: string): HTMLElement {
  const sec = document.createElement('section');
  sec.className = 'panel-section';

  const heading = document.createElement('h2');
  heading.className = 'panel-section-title';
  heading.textContent = title;
  sec.append(heading);
  return sec;
}

function labeledRange(
  labelText: string,
  min: number,
  max: number,
  step: number,
  value: number,
  format: (v: number) => string,
  onInput: (v: number) => void,
): { root: HTMLElement; input: HTMLInputElement; valueEl: HTMLElement } {
  const root = document.createElement('label');
  root.className = 'panel-control';

  const row = document.createElement('div');
  row.className = 'panel-control-row';

  const name = document.createElement('span');
  name.className = 'panel-control-label';
  name.textContent = labelText;

  const valueEl = document.createElement('span');
  valueEl.className = 'panel-control-value';
  valueEl.textContent = format(value);

  row.append(name, valueEl);

  const input = document.createElement('input');
  input.type = 'range';
  input.className = 'panel-slider';
  input.min = String(min);
  input.max = String(max);
  input.step = String(step);
  input.value = String(value);
  input.setAttribute('aria-label', labelText);

  input.addEventListener('input', () => {
    const v = Number(input.value);
    valueEl.textContent = format(v);
    onInput(v);
  });

  root.append(row, input);
  return { root, input, valueEl };
}

function toggle(
  labelText: string,
  checked: boolean,
  onChange: (on: boolean) => void,
): { root: HTMLElement; input: HTMLInputElement } {
  const root = document.createElement('label');
  root.className = 'panel-toggle';

  const input = document.createElement('input');
  input.type = 'checkbox';
  input.className = 'panel-checkbox';
  input.checked = checked;
  input.setAttribute('aria-label', labelText);

  const text = document.createElement('span');
  text.textContent = labelText;

  input.addEventListener('change', () => onChange(input.checked));
  root.append(input, text);
  return { root, input };
}

export function mountPanel(
  el: HTMLElement,
  opts: {
    onSpawn: (vesselId: string) => void;
    onDensity: (v: number) => void;
    onStrength: (v: number) => void;
    onParticleTarget: (n: number) => void;
    onToggleGpu: (on: boolean) => void;
    onToggleStress: (on: boolean) => void;
    onPause: () => void;
    onStep: () => void;
    onReset: () => void;
    onBlueprint: (file: File) => void;
    onWind?: (v: number) => void;
    getConfig: () => SimConfig & { densityScale: number; strengthScale: number };
  },
): { refresh: () => void } {
  el.replaceChildren();
  el.classList.add('panel');

  const cfg0 = opts.getConfig();

  // --- Vessels ---
  const vesselSec = section('Vessels');
  const vesselGrid = document.createElement('div');
  vesselGrid.className = 'vessel-grid';
  vesselGrid.setAttribute('role', 'group');
  vesselGrid.setAttribute('aria-label', 'Spawn vessel');

  for (const v of VESSELS) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'vessel-btn';
    btn.textContent = v.label;
    btn.setAttribute('aria-label', `Spawn ${v.label}`);
    btn.addEventListener('click', () => opts.onSpawn(v.id));
    vesselGrid.append(btn);
  }
  vesselSec.append(vesselGrid);
  el.append(vesselSec);

  // --- Structure ---
  const structSec = section('Structure');
  const density = labeledRange(
    'Density',
    0.25,
    3,
    0.05,
    cfg0.densityScale,
    (v) => v.toFixed(2),
    opts.onDensity,
  );
  const strength = labeledRange(
    'Strength',
    0.25,
    3,
    0.05,
    cfg0.strengthScale,
    (v) => v.toFixed(2),
    opts.onStrength,
  );
  structSec.append(density.root, strength.root);
  el.append(structSec);

  // --- Water ---
  const waterSec = section('Water');
  const particles = labeledRange(
    'Particles',
    PARTICLE_MIN,
    PARTICLE_MAX,
    1000,
    cfg0.particleTarget,
    (v) => `${Math.round(v / 1000)}k`,
    (n) => opts.onParticleTarget(Math.round(n)),
  );
  waterSec.append(particles.root);
  el.append(waterSec);

  // --- Environment (optional mild wind) ---
  let windCtrl: ReturnType<typeof labeledRange> | null = null;
  if (opts.onWind) {
    const envSec = section('Environment');
    windCtrl = labeledRange(
      'Wind',
      -1,
      1,
      0.05,
      cfg0.wind,
      (v) => v.toFixed(2),
      opts.onWind,
    );
    envSec.append(windCtrl.root);
    el.append(envSec);
  }

  // --- Display ---
  const displaySec = section('Display');
  const gpu = toggle('GPU (WebGPU)', cfg0.useGpu, opts.onToggleGpu);
  const stress = toggle('Stress overlay', cfg0.stressOverlay, opts.onToggleStress);
  displaySec.append(gpu.root, stress.root);
  el.append(displaySec);

  // --- Simulation ---
  const simSec = section('Simulation');
  const actions = document.createElement('div');
  actions.className = 'panel-actions';

  const pauseBtn = document.createElement('button');
  pauseBtn.type = 'button';
  pauseBtn.className = 'panel-btn';
  pauseBtn.setAttribute('aria-label', cfg0.paused ? 'Resume' : 'Pause');
  pauseBtn.textContent = cfg0.paused ? 'Resume' : 'Pause';
  pauseBtn.addEventListener('click', () => {
    opts.onPause();
    refresh();
  });

  const stepBtn = document.createElement('button');
  stepBtn.type = 'button';
  stepBtn.className = 'panel-btn';
  stepBtn.textContent = 'Step';
  stepBtn.setAttribute('aria-label', 'Step one frame');
  stepBtn.addEventListener('click', () => opts.onStep());

  const resetBtn = document.createElement('button');
  resetBtn.type = 'button';
  resetBtn.className = 'panel-btn panel-btn-danger';
  resetBtn.textContent = 'Reset';
  resetBtn.setAttribute('aria-label', 'Reset simulation');
  resetBtn.addEventListener('click', () => opts.onReset());

  actions.append(pauseBtn, stepBtn, resetBtn);
  simSec.append(actions);
  el.append(simSec);

  // --- Blueprint ---
  const bpSec = section('Blueprint');
  const fileLabel = document.createElement('label');
  fileLabel.className = 'panel-file';

  const fileText = document.createElement('span');
  fileText.textContent = 'Load PNG blueprint';

  const fileInput = document.createElement('input');
  fileInput.type = 'file';
  fileInput.accept = 'image/png,.png';
  fileInput.className = 'panel-file-input';
  fileInput.setAttribute('aria-label', 'Blueprint PNG file');
  fileInput.addEventListener('change', () => {
    const file = fileInput.files?.[0];
    if (file) opts.onBlueprint(file);
    fileInput.value = '';
  });

  fileLabel.append(fileText, fileInput);
  bpSec.append(fileLabel);
  el.append(bpSec);

  // --- Materials legend ---
  const matSec = section('Materials');
  const legend = document.createElement('ul');
  legend.className = 'materials-legend';
  legend.setAttribute('aria-label', 'Material colors');

  for (const mat of MATERIAL_LIST) {
    const item = document.createElement('li');
    item.className = 'materials-legend-item';

    const swatch = document.createElement('span');
    swatch.className = 'material-swatch';
    swatch.style.backgroundColor = mat.color;
    swatch.setAttribute('aria-hidden', 'true');
    swatch.title = mat.color;

    const name = document.createElement('span');
    name.className = 'material-name';
    name.textContent = mat.name;

    item.append(swatch, name);
    legend.append(item);
  }
  matSec.append(legend);
  el.append(matSec);

  function refresh(): void {
    const cfg = opts.getConfig();
    density.input.value = String(cfg.densityScale);
    density.valueEl.textContent = cfg.densityScale.toFixed(2);
    strength.input.value = String(cfg.strengthScale);
    strength.valueEl.textContent = cfg.strengthScale.toFixed(2);
    particles.input.value = String(cfg.particleTarget);
    particles.valueEl.textContent = `${Math.round(cfg.particleTarget / 1000)}k`;
    gpu.input.checked = cfg.useGpu;
    stress.input.checked = cfg.stressOverlay;
    pauseBtn.textContent = cfg.paused ? 'Resume' : 'Pause';
    pauseBtn.setAttribute('aria-label', cfg.paused ? 'Resume' : 'Pause');
    if (windCtrl) {
      windCtrl.input.value = String(cfg.wind);
      windCtrl.valueEl.textContent = cfg.wind.toFixed(2);
    }
  }

  return { refresh };
}
