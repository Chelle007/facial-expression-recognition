const videoEl   = document.getElementById('webcamVideo');
const canvasEl  = document.getElementById('faceCanvas');
const ctx       = canvasEl.getContext('2d');
const startButton  = document.getElementById('startButton');
const stopButton   = document.getElementById('stopButton');
const statusEl  = document.getElementById('status');
const errorMsg  = document.getElementById('errorMsg');
const modelSel  = document.getElementById('modelSelect');

let currentStream = null;
let rafId = null;

const EMOTIONS = ['Surprise','Fear','Disgust','Happiness','Sadness','Anger','Neutral'];
let currentModel = 'face-api';
let ortSessions = {};
let faceapiReady = false;

const MIRROR_UI = true;

let trackBox = null;  
let lastSeen = 0;
const SMOOTH = 0.25;     
const LOST_TIMEOUT = 500;

const now = () => performance.now();
function lerp(a, b, t){ return a + (b - a) * t; }
function smoothTo(target){
  if (!trackBox) { trackBox = { ...target }; return; }
  trackBox.x = lerp(trackBox.x, target.x, SMOOTH);
  trackBox.y = lerp(trackBox.y, target.y, SMOOTH);
  trackBox.width  = lerp(trackBox.width,  target.width,  SMOOTH);
  trackBox.height = lerp(trackBox.height, target.height, SMOOTH);
}

function setRunning(isOn) {
  startButton.classList.toggle('hidden',  isOn);
  stopButton.classList.toggle('hidden',  !isOn);
  startButton.disabled = isOn;
  stopButton.disabled  = !isOn;
}

const softmax = (arr) => {
  const m = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - m));
  const sum = exps.reduce((a,b)=>a+b,0);
  return exps.map(v => v/sum);
};
function setStatus(msg) { statusEl.textContent = msg; }
function setError(msg)  { errorMsg.textContent = msg; errorMsg.classList.remove('hidden'); }
function clearError()   { errorMsg.textContent = ''; errorMsg.classList.add('hidden'); }

function clamp(v, lo, hi){ return Math.max(lo, Math.min(hi, v)); }
function padBox(b, k=0.12){
  const cw = videoEl.clientWidth;
  const ch = videoEl.clientHeight;
  const cx = b.x + b.width/2;
  const cy = b.y + b.height/2;
  const w  = b.width  * (1 + k);
  const h  = b.height * (1 + k);
  const x  = clamp(cx - w/2, 0, Math.max(0, cw - w));
  const y  = clamp(cy - h/2 - 0.05*h, 0, Math.max(0, ch - h)); 
  return { x, y, width:w, height:h };
}

function highlightEmotion(activeEmotion) {
  EMOTIONS.forEach(em => {
    const badge = document.getElementById(em);
    if (!badge) return;
    if (em === activeEmotion) {
      badge.classList.add('active-emotion');
    } else {
      badge.classList.remove('active-emotion');
    }
  });
}

async function startCamera() {
  clearError();
  setRunning(true);

  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error('Your browser does not support camera access.');
    }
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user' }, audio: false
    });
    currentStream = stream;
    videoEl.srcObject = stream;
    await new Promise(r => (videoEl.onloadedmetadata = r));
    resizeCanvas();
    await ensureDetectors();
    trackLoop();            
  } catch (err) {
    setRunning(false);
    setError(err.message || 'Error starting camera.');
    stopCamera();             
  }
}

function stopCamera() {
  if (rafId) { cancelAnimationFrame(rafId); rafId = null; }

  if (currentStream) {
    currentStream.getTracks().forEach(t => t.stop());
    videoEl.srcObject = null;
    currentStream = null;
  }
  setRunning(false);      
  setStatus('Idle.');
  trackBox = null;         
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  highlightEmotion(null);
}

function resizeCanvas() {
  const dpr = devicePixelRatio || 1;
  const w = videoEl.clientWidth;
  const h = videoEl.clientHeight;
  canvasEl.width  = Math.round(w * dpr);
  canvasEl.height = Math.round(h * dpr);
  canvasEl.style.width = w + 'px';
  canvasEl.style.height = h + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener('resize', resizeCanvas);

async function ensureDetectors() {
  try {
    await faceapi.nets.tinyFaceDetector.loadFromUri('./models/');
    await faceapi.nets.faceExpressionNet.loadFromUri('./models/');
    faceapiReady = true;
    setStatus('face-api ready.');
  } catch {
    faceapiReady = false;
    setStatus('Using FaceDetector API fallback (face-api models not found).');
  }
}

async function detectFaceBox() {
  const opts = faceapiReady
    ? new faceapi.TinyFaceDetectorOptions({ inputSize: 256, scoreThreshold: 0.4 })
    : null;

  if (faceapiReady) {
    const det = await faceapi.detectSingleFace(videoEl, opts);
    if (det) {
      const resized = faceapi.resizeResults(det, {
        width:  videoEl.clientWidth,
        height: videoEl.clientHeight
      });
      const b = resized.box;
      return padBox({ x: b.x, y: b.y, width: b.width, height: b.height });
    }
  }

  if ('FaceDetector' in window) {
    try {
      if (!detectFaceBox._fd) detectFaceBox._fd = new window.FaceDetector({ fastMode: true, maxDetectedFaces: 1 });
      const faces = await detectFaceBox._fd.detect(videoEl);
      if (faces.length) {
        const b = faces[0].boundingBox;
        return padBox({ x:b.x, y:b.y, width:b.width, height:b.height });
      }
    } catch { /* ignore */ }
  }
  return null;
}

async function ensureOrtSession(key) {
  if (ortSessions[key]) return ortSessions[key];
  let path;

  if (key === 'resnet18')       path = 'ResNet18.onnx';
  if (key === 'mobilenetv2')    path = 'MobileNetV2.onnx';
  if (key === 'emotion_cnn')    path = 'emotion_cnn.onnx';
  if (key === 'efficientnetb0') path = 'EfficientNetB0.onnx';

  if (!path) throw new Error('Unknown ONNX model key: ' + key);

  setStatus(`Loading ${path} ...`);
  const session = await ort.InferenceSession.create(path, { executionProviders: ['wasm','webgl','webgpu'] });
  ortSessions[key] = session;
  setStatus(`${path} loaded.`);
  return session;
}

function toTensorFromBox(box, size = 224) {
  const vw = videoEl.videoWidth || videoEl.clientWidth;
  const vh = videoEl.videoHeight || videoEl.clientHeight;
  const cw = videoEl.clientWidth;
  const ch = videoEl.clientHeight;

  const bx = MIRROR_UI ? (cw - (box.x + box.width)) : box.x;
  const by = box.y;
  const bw = box.width;
  const bh = box.height;

  const sx = clamp(Math.round(bx * (vw / cw)), 0, vw);
  const sy = clamp(Math.round(by * (vh / ch)), 0, vh);
  const sw = clamp(Math.round(bw * (vw / cw)), 1, vw - sx);
  const sh = clamp(Math.round(bh * (vh / ch)), 1, vh - sy);

  const off = toTensorFromBox._c || (toTensorFromBox._c = document.createElement('canvas'));
  off.width = size; off.height = size;
  const octx = off.getContext('2d', { willReadFrequently: true });
  octx.clearRect(0, 0, size, size);
  octx.drawImage(videoEl, sx, sy, sw, sh, 0, 0, size, size);

  const { data } = octx.getImageData(0, 0, size, size);
  const chw = new Float32Array(3 * size * size);
  let p = 0, rOff = 0, gOff = size * size, bOff = 2 * size * size;
  for (let i = 0; i < data.length; i += 4) {
    const r = data[i]   / 255;
    const g = data[i+1] / 255;
    const b = data[i+2] / 255;
    chw[rOff + p] = (r - 0.5) / 0.5;
    chw[gOff + p] = (g - 0.5) / 0.5;
    chw[bOff + p] = (b - 0.5) / 0.5;
    p++;
  }
  return new ort.Tensor('float32', chw, [1, 3, size, size]);
}

async function runFaceApiExpressions() {
  if (!faceapiReady) return null;
  const det = await faceapi
    .detectSingleFace(videoEl, new faceapi.TinyFaceDetectorOptions({ inputSize: 256, scoreThreshold: 0.4 }))
    .withFaceExpressions();

  if (!det?.expressions) return null;

  const mapOrder = ['surprised','fearful','disgusted','happy','sad','angry','neutral'];
  const scores = mapOrder.map(k => det.expressions[k] ?? 0);
  return softmax(scores);
}

async function runOnnxEmotion(which) {
  const session = await ensureOrtSession(which);
  const inputName  = session.inputNames?.[0]  || 'input';
  const outputName = session.outputNames?.[0] || 'output';
  const tensor = trackBox
    ? toTensorFromBox(trackBox, 224)
    : toTensorFromBox({x:0,y:0,width:videoEl.clientWidth,height:videoEl.clientHeight}, 224);
  const results = await session.run({ [inputName]: tensor });
  const out = results[outputName] || Object.values(results)[0];
  const logits = Array.from(out.data);
  return softmax(logits).slice(0, 7);
}

function drawBoxAndLabel(box, label, prob) {
  ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
  if (!box) return;

  const dpr = devicePixelRatio || 1;
  const canvasCssW = canvasEl.width / dpr;
  const drawX = MIRROR_UI ? (canvasCssW - (box.x + box.width)) : box.x;

  ctx.lineWidth = 4;
  ctx.strokeStyle = '#00e676';
  ctx.fillStyle = 'rgba(0,0,0,0.6)';
  ctx.strokeRect(drawX, box.y, box.width, box.height);

  const text = `${label} ${(prob*100).toFixed(1)}% [${currentModel}]`;
  ctx.font = '16px system-ui, -apple-system, Segoe UI, Roboto';
  const pad = 6;
  const tw = ctx.measureText(text).width;
  const th = 22;
  const tx = Math.max(4, drawX);
  const ty = Math.max(th + 4, box.y);
  ctx.fillRect(tx - pad, ty - th, tw + pad*2, th);
  ctx.fillStyle = '#fff';
  ctx.fillText(text, tx, ty - 6);
}

async function trackLoop() {
  if (!currentStream) return;

  const detected = await detectFaceBox();
  const t = now();
  if (detected) {
    smoothTo(detected);
    lastSeen = t;
  } else {
    if (t - lastSeen > LOST_TIMEOUT) trackBox = null;
  }

  let probs = null;
  if (currentModel === 'face-api' && faceapiReady) {
    probs = await runFaceApiExpressions();
  } else if (
    currentModel === 'resnet18' ||
    currentModel === 'mobilenetv2' ||
    currentModel === 'emotion_cnn' ||
    currentModel === 'efficientnetb0'
  ) {
    try { probs = await runOnnxEmotion(currentModel); }
    catch (e) { setStatus(`ONNX error: ${e.message}`); }
  }

  if (trackBox && probs) {
    let maxI = 0, maxV = probs[0];
    for (let i = 1; i < probs.length; i++) if (probs[i] > maxV) { maxV = probs[i]; maxI = i; }
    drawBoxAndLabel(trackBox, EMOTIONS[maxI], maxV);
    setStatus(`Detected: ${EMOTIONS[maxI]} ${(maxV*100).toFixed(1)}%`);
    highlightEmotion(EMOTIONS[maxI]);
  } else if (trackBox) {
    drawBoxAndLabel(trackBox, 'Detecting...', 0);
    setStatus('Face found, running model...');
  } else {
    ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    setStatus('Looking for a face...');
    highlightEmotion(null);
  }

  setTimeout(trackLoop, 200);
}

startButton.addEventListener('click', startCamera);
stopButton.addEventListener('click', stopCamera);
modelSel.addEventListener('change', () => {
  currentModel = modelSel.value;
  setStatus(`Model switched to: ${currentModel}`);
});
currentModel = modelSel.value;

setStatus('Idle. Click the camera button to start.');
