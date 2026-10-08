// Lightning from Thunder: viewer for the simulated strike library.
// Data: viewer/bolts/index.json plus <id>.json and <id>.mp3 per strike, written by
// scripts/make_viewer_data.py from the project's pipeline. Scene units are kilometres;
// three.js axes: x = east, y = up, z = south.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { CSS2DRenderer, CSS2DObject } from "three/addons/renderers/CSS2DRenderer.js";
import { LineSegments2 } from "three/addons/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/addons/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/addons/lines/LineMaterial.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

const $ = (id) => document.getElementById(id);
const KM = 1e-3;
const toV = (p) => new THREE.Vector3(p[0] * KM, p[2] * KM, -p[1] * KM);
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const narrow = () => innerWidth <= 820;
const fail = (msg) => { $("loading").textContent = msg; throw new Error(msg); };

let LIB;
try { LIB = await (await fetch("viewer/bolts/index.json")).json(); }
catch { fail("Couldn't load the strike library. Check your connection and reload the page."); }
const NS = LIB.strikes.length;
$("storm-n").textContent = String(NS);

// ---------------------------------------------------------------- palette (matches style.css)
const C = (h) => new THREE.Color(h);
const TRUTH = [C("#a8c4e8"), C("#8ea5cc"), C("#b3a6d8")];  // main channel, branch, in-cloud
const HEARD = C("#7fcfbf"), FLASH = C("#f2f6ff"), LEADER = C("#c9d4ee"), AMBER = C("#efb560");
const ERR = ["#fbe7b5", "#efb560", "#e08a4f", "#c95c45"].map(C);
function errColor(e) {  // 0 m pale gold → 15 m and beyond rust
  const x = Math.min(Math.max((e ?? 15) / 15, 0), 1) * 3, i = Math.min(Math.floor(x), 2);
  return ERR[i].clone().lerp(ERR[i + 1], x - i);
}

// ---------------------------------------------------------------- renderer, camera, post-processing
let renderer;
try { renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: "high-performance" }); }
catch { fail("This page needs WebGL, which is turned off or unavailable in this browser."); }
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.75));
renderer.setSize(innerWidth, innerHeight);
$("scene").appendChild(renderer.domElement);
const labels = new CSS2DRenderer();
labels.setSize(innerWidth, innerHeight);
Object.assign(labels.domElement.style, { position: "absolute", top: "0", pointerEvents: "none" });
$("scene").appendChild(labels.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x151c2a, 0.025);
const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.01, 300);
camera.position.set(8, 3, 10);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.maxPolarAngle = Math.PI * 0.495;
controls.minDistance = 0.3;
controls.maxDistance = 45;
controls.autoRotateSpeed = 0.3;

// Render through a multisampled target so the thin channel lines stay smooth under bloom.
const composer = new EffectComposer(renderer, new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 }));
composer.setSize(innerWidth, innerHeight);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 0.7, 0.45, 0.28);
composer.addPass(bloom);
composer.addPass(new OutputPass());

// Render only when something changes: playback, the intro, a camera move or a UI action.
let dirty = true;
const invalidate = () => { dirty = true; };
controls.addEventListener("change", invalidate);
addEventListener("pointerdown", invalidate);
addEventListener("keydown", invalidate);

function frame() {
  const w = innerWidth, h = innerHeight;
  camera.aspect = w / h;
  // Wide screens: the story panel is on the left, so centre the scene in the space beside it.
  // Narrow screens: the panel covers the bottom, so lift the scene.
  if (!narrow()) camera.setViewOffset(w, h, -90, 0, w, h);
  else camera.setViewOffset(w, h, 0, 0.2 * h, w, h);
  camera.updateProjectionMatrix();
}
frame();
addEventListener("resize", () => {
  frame();
  renderer.setSize(innerWidth, innerHeight);
  labels.setSize(innerWidth, innerHeight);
  composer.setSize(innerWidth, innerHeight);
  chanMat.resolution.set(innerWidth, innerHeight);
  stormMat.resolution.set(innerWidth, innerHeight);
  invalidate();
});

let tween = null;
function flyTo(pos, target, dur = 1.6) {
  tween = { p0: camera.position.clone(), q0: controls.target.clone(), p1: pos, q1: target, start: performance.now(), dur: dur * 1000 };
}
controls.addEventListener("start", () => {
  tween = null; controls.autoRotate = false;
  $("hint").style.opacity = "0";
});

// ---------------------------------------------------------------- textures
function canvasTex(size, draw) {
  const c = document.createElement("canvas"); c.width = c.height = size;
  draw(c.getContext("2d"), size);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}
const dotTex = canvasTex(64, (g) => {
  const r = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  r.addColorStop(0, "rgba(255,255,255,1)"); r.addColorStop(0.45, "rgba(255,255,255,1)");
  r.addColorStop(0.62, "rgba(255,255,255,0.3)"); r.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = r; g.fillRect(0, 0, 64, 64);
});
const glowTex = canvasTex(128, (g) => {
  const r = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  r.addColorStop(0, "rgba(255,255,255,0.8)"); r.addColorStop(0.3, "rgba(210,220,255,0.25)"); r.addColorStop(1, "rgba(200,210,255,0)");
  g.fillStyle = r; g.fillRect(0, 0, 128, 128);
});
let seed = 7;
const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);  // seeded, so the sky looks the same each visit
const cloudTex = [0, 1, 2].map(() => canvasTex(128, (g) => {
  for (let i = 0; i < 24; i++) {
    const a = rand() * Math.PI * 2, d = Math.sqrt(rand()) * 34, x = 64 + Math.cos(a) * d, y = 64 + Math.sin(a) * d * 0.6;
    const rr = 18 + rand() * 22, r = g.createRadialGradient(x, y, 0, x, y, rr);
    r.addColorStop(0, "rgba(255,255,255,0.2)"); r.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = r; g.fillRect(0, 0, 128, 128);
  }
}));

// ---------------------------------------------------------------- environment
const skyMat = new THREE.ShaderMaterial({
  side: THREE.BackSide, depthWrite: false, fog: false,
  uniforms: { flash: { value: 0 }, flashDir: { value: new THREE.Vector3(0, 1, 0) } },
  vertexShader: `varying vec3 vDir; void main() { vDir = normalize(position);
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`,
  fragmentShader: `uniform float flash; uniform vec3 flashDir; varying vec3 vDir;
    void main() {
      vec3 d = normalize(vDir);
      // sRGB colours, converted to linear at the end (the output pass encodes back to sRGB).
      vec3 zenith = vec3(0.043, 0.063, 0.106), horizon = vec3(0.114, 0.149, 0.212), below = vec3(0.07, 0.09, 0.13);
      vec3 c = mix(horizon, zenith, smoothstep(0.0, 0.5, d.y));
      c = mix(c, below, smoothstep(0.0, -0.1, d.y));
      float f = flash * (0.3 + 0.7 * pow(max(dot(d, flashDir), 0.0), 4.0));
      c += vec3(0.55, 0.6, 0.8) * f * 0.6;
      gl_FragColor = vec4(pow(c, vec3(2.2)), 1.0); }`,
});
const sky = new THREE.Mesh(new THREE.SphereGeometry(120, 32, 16), skyMat);
scene.add(sky);

const groundMat = new THREE.MeshBasicMaterial({ color: 0x0f141f });
const ground = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), groundMat);
ground.rotation.x = -Math.PI / 2; ground.position.y = -0.003;
scene.add(ground);
scene.add(new THREE.GridHelper(40, 40, 0x273146, 0x19202e));
const groundGlow = new THREE.Mesh(new THREE.PlaneGeometry(1, 1),
  new THREE.MeshBasicMaterial({ map: glowTex, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, opacity: 0 }));
groundGlow.rotation.x = -Math.PI / 2; groundGlow.position.y = 0.002;
scene.add(groundGlow);

function label(text, pos, cls = "tag") {
  const el = document.createElement("div"); el.className = cls; el.textContent = text;
  const o = new CSS2DObject(el); o.position.copy(pos); return o;
}
const RINGS = [1, 2, 4, 6];  // range rings around the microphones
const ringLabels = RINGS.map((r) => {
  const pts = [];
  for (let i = 0; i <= 128; i++) { const a = (i / 128) * Math.PI * 2; pts.push(new THREE.Vector3(Math.cos(a) * r, 0.001, Math.sin(a) * r)); }
  scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), new THREE.LineBasicMaterial({ color: 0x2c374c })));
  const l = label(`${r} km`, new THREE.Vector3(r * 0.707, 0, r * 0.707), "tag ghost");
  scene.add(l); return l;
});
// Put the ring labels on the far side of the array from the strike, clear of the other labels.
function placeRingLabels(dir) {
  const d = dir.clone().applyAxisAngle(new THREE.Vector3(0, 1, 0), 2.4);
  ringLabels.forEach((l, i) => l.position.copy(d).multiplyScalar(RINGS[i]));
}
scene.add(label("N", new THREE.Vector3(0, 0, -6.4), "tag ghost"));

// The array is 50 m across: one marker at this scale, with a ring that follows the sound level.
const micSprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: dotTex, color: 0xe6e9ef, sizeAttenuation: false, depthTest: false, transparent: true }));
micSprite.scale.setScalar(0.01); micSprite.position.set(0, 0.002, 0); micSprite.renderOrder = 10;
scene.add(micSprite);
const micRing = new THREE.Mesh(new THREE.RingGeometry(0.93, 1, 64),
  new THREE.MeshBasicMaterial({ color: HEARD, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
micRing.rotation.x = -Math.PI / 2; micRing.position.y = 0.004;
scene.add(micRing);
scene.add(label("5 microphones", new THREE.Vector3(0, 0.22, 0)));

// A thin cloud deck at the channel top (drawn there so the whole channel stays in view).
let clouds = [];
const cloudBase = C("#222b3d"), cloudLit = C("#b9c6e6");
function buildClouds(cx, cz, radius, height, count) {
  clouds.forEach((c) => { scene.remove(c); c.material.dispose(); });
  clouds = [];
  for (let i = 0; i < count; i++) {
    const a = rand() * Math.PI * 2, d = Math.sqrt(rand()) * radius;
    const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: cloudTex[i % 3], color: cloudBase, transparent: true,
      opacity: 0.35 + rand() * 0.3, depthWrite: false }));
    s.position.set(cx + Math.cos(a) * d, height + 0.3 + rand() * 1.1, cz + Math.sin(a) * d);
    const k = 3 + rand() * 3.5; s.scale.set(k * 1.7, k, 1);
    s.material.rotation = rand() * Math.PI;
    scene.add(s); clouds.push(s);
  }
}
function lightClouds(f, at) {
  for (const s of clouds) {
    const w = at ? f * Math.exp(-s.position.distanceTo(at) / 3.5) : 0;
    s.material.color.copy(cloudBase).lerp(cloudLit, Math.min(w, 1));
  }
}

// ---------------------------------------------------------------- shared materials
const chanMat = new LineMaterial({ vertexColors: true, linewidth: 2, resolution: new THREE.Vector2(innerWidth, innerHeight) });
const stormMat = new LineMaterial({ vertexColors: true, linewidth: 1.3, resolution: new THREE.Vector2(innerWidth, innerHeight) });
const ptMat = (size) => new THREE.PointsMaterial({ size, map: dotTex, vertexColors: true, sizeAttenuation: false,
  transparent: true, depthWrite: false, alphaTest: 0.05 });
const shellGeom = new THREE.SphereGeometry(1, 48, 24);
const shellMat = new THREE.ShaderMaterial({
  uniforms: { opacity: { value: 0 } },
  vertexShader: `varying float rim; void main() { vec4 mv = modelViewMatrix * vec4(position, 1.0);
    rim = 1.0 - abs(dot(normalize(normalMatrix * normal), normalize(-mv.xyz))); gl_Position = projectionMatrix * mv; }`,
  fragmentShader: `uniform float opacity; varying float rim; void main() { gl_FragColor = vec4(0.5, 0.62, 0.8, opacity * pow(rim, 4.0)); }`,
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
});

// ---------------------------------------------------------------- one strike
const cache = new Map();
function fetchStrike(id) {
  if (!cache.has(id)) cache.set(id, fetch(`viewer/bolts/${id}.json`).then((r) => {
    if (!r.ok) throw new Error(`strike ${id}: HTTP ${r.status}`);
    return r.json();
  }).catch((e) => { cache.delete(id); throw e; }));
  return cache.get(id);
}

function buildStrike(D) {
  const S = { D, A: D.about, M: D.metrics };
  const group = new THREE.Group(); S.group = group;
  const nodes = D.channel.nodes.map(toV);
  // Sort segments by distance along the channel from its origin, so "the leader has reached
  // here" becomes a draw count.
  const raw = D.channel;
  const order = raw.segments.map((_, i) => i).sort((a, b) => raw.leader_m[a] - raw.leader_m[b]);
  S.n = order.length;
  S.kind = order.map((i) => raw.kind[i]);
  S.arrival = order.map((i) => raw.arrival_s[i]);
  S.leader = order.map((i) => raw.leader_m[i]);
  S.maxLeader = Math.max(1, S.leader[S.n - 1]);
  const pos = new Float32Array(S.n * 6), mains = [];
  order.forEach((i, k) => {
    const [a, b] = raw.segments[i];
    nodes[a].toArray(pos, k * 6); nodes[b].toArray(pos, k * 6 + 3);
    if (S.kind[k] === 0) mains.push(nodes[a].clone().lerp(nodes[b], 0.5));
  });
  S.col = new Float32Array(S.n * 6);
  S.geom = new LineSegmentsGeometry().setPositions(pos).setColors(S.col);
  S.colBuf = S.geom.attributes.instanceColorStart.data;
  S.line = new LineSegments2(S.geom, chanMat);
  group.add(S.line);
  S.glow = new Float32Array(S.n);

  const box = new THREE.Box3().setFromPoints(nodes);
  S.box = box;
  S.strike = toV(raw.strike);
  S.top = nodes.reduce((a, b) => (b.y > a.y ? b : a), nodes[0]);
  const heard = S.arrival.filter((a) => a !== null);
  S.firstArrival = Math.min(...heard);
  S.lastArrival = Math.max(...heard);
  S.dur = D.waveform.duration_s;

  // Rebuilt points (time-ordered), the newest drawn larger; 2-sigma bars; rays from the array.
  const P = D.points; S.P = P; S.np = P.t.length;
  S.ptVec = P.xyz.map(toV);
  S.recon = new THREE.Group(); group.add(S.recon);
  S.ptGeom = new THREE.BufferGeometry().setFromPoints(S.ptVec);
  S.ptGeom.setAttribute("color", new THREE.Float32BufferAttribute(P.error_m.flatMap((e) => errColor(e).toArray()), 3));
  S.points = new THREE.Points(S.ptGeom, ptMat(8));
  S.newGeom = S.ptGeom.clone(); S.newGeom.setDrawRange(0, 0);
  S.points.add(new THREE.Points(S.newGeom, ptMat(17)));
  S.recon.add(S.points);
  const sg = [];
  P.sigma_m.forEach((s, i) => {
    const p = S.ptVec[i];
    [[2 * s[0] * KM, 0, 0], [0, 2 * s[2] * KM, 0], [0, 0, 2 * s[1] * KM]].forEach(([dx, dy, dz]) =>
      sg.push(p.x - dx, p.y - dy, p.z - dz, p.x + dx, p.y + dy, p.z + dz));
  });
  const sgGeom = new THREE.BufferGeometry(); sgGeom.setAttribute("position", new THREE.Float32BufferAttribute(sg, 3));
  S.sigma = new THREE.LineSegments(sgGeom, new THREE.LineBasicMaterial({ color: 0xf3d9a6, transparent: true, opacity: 0.45 }));
  S.sigma.visible = false; S.recon.add(S.sigma);
  S.rayGeom = new THREE.BufferGeometry();
  S.rayGeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(60), 3));
  S.rayGeom.setAttribute("color", new THREE.BufferAttribute(new Float32Array(60), 3));
  S.rays = new THREE.LineSegments(S.rayGeom, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, blending: THREE.AdditiveBlending }));
  S.rays.frustumCulled = false;
  group.add(S.rays);

  // Sound shells from a few points along the main channel (illustrative).
  S.shells = [];
  for (let k = 0; k < mains.length; k += Math.max(1, Math.floor(mains.length / 8))) {
    const m = new THREE.Mesh(shellGeom, shellMat.clone());
    m.position.copy(mains[k]); m.visible = false; group.add(m); S.shells.push(m);
  }

  // CSS2D labels ignore their parent's visibility, so keep handles to hide them in the storm view.
  S.labels = [label("strike point", S.strike.clone().add(new THREE.Vector3(0, -0.12, 0)))];
  S.labels.forEach((l) => group.add(l));

  // Camera: fit the strike and the array into the free screen area, side-on to the line
  // between them and a little from above.
  const ctr = box.getCenter(new THREE.Vector3());
  S.along = new THREE.Vector3(ctr.x, 0, ctr.z).normalize();
  const dir = new THREE.Vector3(-S.along.z, 0, S.along.x).addScaledVector(S.along, -0.3).normalize();
  dir.y = 0.2; dir.normalize();
  const sphere = box.clone().expandByPoint(new THREE.Vector3()).getBoundingSphere(new THREE.Sphere());
  S.focus = sphere.center.clone();
  const vfov = THREE.MathUtils.degToRad(camera.fov), aspect = innerWidth / innerHeight;
  const hfit = 2 * Math.atan(Math.tan(vfov / 2) * aspect * (narrow() ? 0.95 : 0.58));
  const vfit = narrow() ? 2 * Math.atan(Math.tan(vfov / 2) * 0.55) : vfov * 0.82;
  S.home = S.focus.clone().addScaledVector(dir, sphere.radius / Math.sin(Math.min(hfit, vfit) / 2));
  S.sideOffset = S.along.clone().multiplyScalar(-(0.8 * Math.hypot(box.max.x - box.min.x, box.max.z - box.min.z) + 0.8));
  S.recT0 = Math.max(0, Math.min(P.t[0] ?? S.firstArrival, S.firstArrival) - 0.6);
  S.recT1 = Math.min(S.dur, Math.max(P.t[S.np - 1] ?? S.lastArrival, S.lastArrival) + 0.6);
  S.strikeEnd = Math.min(S.lastArrival + 1, S.dur);
  return S;
}

function disposeStrike(S) {
  scene.remove(S.group);
  S.group.traverse((o) => {
    if (o.isCSS2DObject) o.element.remove();
    if (o.geometry && o.geometry !== shellGeom) o.geometry.dispose();
    if (o.material && o.material !== chanMat) o.material.dispose();
  });
}

// Channel colour: base × level, or the leader during the intro; white in the flash; teal where heard.
function paintStrike(S, { level = 1, leaderL = Infinity, flash = 0 }) {
  const c = new THREE.Color();
  let shown = S.n;
  for (let k = 0; k < S.n; k++) {
    if (S.leader[k] > leaderL) { shown = k; break; }
    if (leaderL < Infinity) {
      const tip = Math.max(0, 1 - (leaderL - S.leader[k]) / 300);
      c.copy(LEADER).multiplyScalar(0.4 + 0.6 * tip);
    } else {
      c.copy(TRUTH[S.kind[k]]).multiplyScalar(level);
      if (S.glow[k] > 0) c.lerp(HEARD, Math.min(S.glow[k], 1));
      if (flash > 0) c.lerp(FLASH, Math.min(flash, 1));
    }
    c.toArray(S.col, k * 6); c.toArray(S.col, k * 6 + 3);
  }
  S.geom.instanceCount = shown;
  S.colBuf.array.set(S.col); S.colBuf.needsUpdate = true;
}
const countUpTo = (S, t) => { let k = 0; while (k < S.np && S.P.t[k] <= t) k++; return k; };
function showPoints(S, t, pop) {
  const k = countUpTo(S, t), j = countUpTo(S, t - pop);
  S.ptGeom.setDrawRange(0, k); S.newGeom.setDrawRange(j, k - j);
  return k;
}
function drawRays(S, t) {
  const pos = S.rayGeom.attributes.position.array, col = S.rayGeom.attributes.color.array;
  pos.fill(0); col.fill(0);
  let r = 0;
  for (let i = S.np - 1; i >= 0 && r < 10 && t >= 0; i--) {
    if (S.P.t[i] > t) continue;
    const age = t - S.P.t[i];
    if (age > 1) break;
    S.ptVec[i].toArray(pos, r * 6 + 3);
    const c = AMBER.clone().multiplyScalar(0.5 * (1 - age));
    c.toArray(col, r * 6); c.toArray(col, r * 6 + 3);
    r++;
  }
  S.rayGeom.attributes.position.needsUpdate = true; S.rayGeom.attributes.color.needsUpdate = true;
}
function heardGlow(S, t, half) {
  for (let k = 0; k < S.n; k++) {
    const a = S.arrival[k];
    const g = a !== null && Math.abs(t - a) < half ? 1 - Math.abs(t - a) / half : 0;
    S.glow[k] = Math.max(g, S.glow[k] * 0.86);
  }
}

// ---------------------------------------------------------------- player
const audio = $("audio");
const player = { mode: "strike", t: 0, playing: false, speed: 2, introStart: -1 };
let S = null, soundOn = true, audioBroken = false, finalHold = false, stormOn = false, truthLevel = 0.5;
const LEADER_S = 1.5, FLASH_S = 1.2;  // intro: leader descent, then the return-stroke flash (seconds)
const inIntro = (now) => player.introStart >= 0 && now - player.introStart < (LEADER_S + FLASH_S) * 1000;
const ICON_PLAY = '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><path d="M2.5 1.2v9.6L10.6 6z" fill="currentColor"/></svg>';
const ICON_PAUSE = '<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true"><rect x="2" y="1.5" width="3" height="9" rx=".6" fill="currentColor"/><rect x="7" y="1.5" width="3" height="9" rx=".6" fill="currentColor"/></svg>';

function setSpeed(s) {
  player.speed = s;
  document.querySelectorAll("[data-speed]").forEach((b) => b.setAttribute("aria-pressed", String(parseFloat(b.dataset.speed) === s)));
  syncAudio();
}
function setPlaying(on) {
  player.playing = on;
  if (on) audioBroken = false;  // retry audio on each play: autoplay may have been blocked before a click
  $("play").innerHTML = on ? ICON_PAUSE : ICON_PLAY;
  $("play").setAttribute("aria-label", on ? "Pause" : "Play");
  syncAudio();
  invalidate();
}
function syncAudio() {
  const want = player.playing && soundOn && !inIntro(performance.now()) && !stormOn && !audioBroken;
  if (!want) { if (!audio.paused) audio.pause(); return; }
  audio.playbackRate = player.speed;
  audio.preservesPitch = false;
  if (Math.abs(audio.currentTime - player.t) > 0.08) audio.currentTime = player.t;
  if (audio.paused) audio.play().catch(() => { audioBroken = true; });
}
function seek(t) {
  player.t = Math.max(0, Math.min(S.dur, t));
  player.introStart = -1;
  finalHold = false;
  S.glow.fill(0);
  syncAudio();
  invalidate();
}
function playStrike(intro = true) {
  finalHold = false;
  player.mode = "strike"; player.t = 0; S.glow.fill(0);
  setSpeed(2);
  player.introStart = intro && !reduceMotion ? performance.now() : -1;
  setPlaying(true);
}
function playRecon() {
  finalHold = false;
  player.mode = "recon"; player.introStart = -1; S.glow.fill(0);
  player.t = S.recT0;
  setSpeed(1);
  setPlaying(true);
}

// ---------------------------------------------------------------- waveform
const wave = $("wave"), wctx = wave.getContext("2d");
let view = { t0: 0, t1: 1 };
function drawWave() {
  if (!S) return;
  const W = S.D.waveform, t = player.t;
  const dpr = Math.min(devicePixelRatio, 2), w = wave.clientWidth, h = wave.clientHeight;
  if (wave.width !== Math.round(w * dpr) || wave.height !== Math.round(h * dpr)) { wave.width = Math.round(w * dpr); wave.height = Math.round(h * dpr); }
  wctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  wctx.clearRect(0, 0, w, h);
  const x = (tt) => ((tt - view.t0) / (view.t1 - view.t0)) * w;
  wctx.fillStyle = "rgba(239,181,96,0.07)";
  wctx.fillRect(0, 0, Math.max(0, x(t)), h);
  wctx.lineWidth = 1;
  for (const played of [false, true]) {
    wctx.strokeStyle = played ? "#a8c4e8" : "#4a5772";
    wctx.beginPath();
    for (let i = 0; i < W.min.length; i++) {
      const tt = i * W.dt_s;
      if (tt < view.t0 || tt > view.t1 || (tt <= t) !== played) continue;
      const xx = x(tt);
      wctx.moveTo(xx, h / 2 - W.max[i] * h * 0.46); wctx.lineTo(xx, h / 2 - W.min[i] * h * 0.46 + 0.5);
    }
    wctx.stroke();
  }
  if (player.mode === "recon") {  // one tick per rebuilt point, coloured by its error
    for (let i = 0; i < S.np && S.P.t[i] <= t; i++) {
      if (S.P.t[i] < view.t0) continue;
      wctx.fillStyle = errColor(S.P.error_m[i]).getStyle();
      wctx.fillRect(x(S.P.t[i]) - 0.6, h - 5, 1.3, 5);
    }
  }
  wctx.fillStyle = "#efb560";
  wctx.fillRect(Math.round(x(t)) - 1, 0, 2, h);
  $("tlabel").textContent = t.toFixed(1);
}
wave.addEventListener("click", (e) => {
  if (!S || stormOn) return;
  const r = wave.getBoundingClientRect();
  seek(view.t0 + ((e.clientX - r.left) / r.width) * (view.t1 - view.t0));
});

// ---------------------------------------------------------------- facts card & status
const COMPASS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"];
const SHORT = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
const dirIdx = (deg) => Math.round((((deg % 360) + 360) % 360) / 45) % 8;
const TYPE = { branched: "Branched strike", with_incloud: "Strike with an in-cloud channel", tortuous: "Unbranched strike" };
const pct = (x) => (x === null || x === undefined ? "–" : `${Math.round(x * 100)}%`);
const fmt = (x, d = 1) => (x === null || x === undefined ? "–" : x.toFixed(d));
const rows = (r) => r.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");

function factsStrike() {
  const A = S.A;
  $("f-title").textContent = TYPE[A.preset] ?? "Strike";
  $("f-id").textContent = `${A.id + 1}/${NS}`;
  const r = [
    ["Distance", `${A.distance_km.toFixed(1)} km ${SHORT[dirIdx(A.azimuth_deg)]}`],
    ["Starts at", `${A.top_km.toFixed(1)} km`],
    ["Length", `${A.length_km.toFixed(1)} km`],
    ["Branches", String(A.n_branches)],
    ["Air", `${A.temperature_c.toFixed(0)} °C`],
    ["Wind", `${A.wind_mps.toFixed(1)} m/s ${SHORT[dirIdx(A.wind_from_deg)]}`],
    ["Heard", `${fmt(S.firstArrival)}–${fmt(S.lastArrival)} s`],
  ];
  if (cur >= 2) r.push(["Rebuilt", `${A.n_points} pts`], ["Median error", `${fmt(A.median_error_m)} m`]);
  $("f-list").innerHTML = rows(r);
}
function factsStorm() {
  const s = LIB.summary, d = LIB.strikes.map((x) => x.distance_km);
  $("f-title").textContent = "All strikes";
  $("f-id").textContent = String(NS);
  $("f-list").innerHTML = rows([
    ["Distance", `${Math.min(...d).toFixed(1)}–${Math.max(...d).toFixed(1)} km`],
    ["Points", s.total_points.toLocaleString("en-US")],
    ["Median error", `${s.median_of_medians_m.toFixed(1)} m`],
    ["Main channel", pct(s.median_coverage_main)],
  ]);
}
let statusText = "";
function setStatus(text) { if (text !== statusText) { $("status").textContent = text; statusText = text; } }

// ---------------------------------------------------------------- per-frame strike update
function strikeFrame(now, dt) {
  const intro = inIntro(now);
  let flash = 0, leaderL = Infinity;
  if (intro) {
    const s = (now - player.introStart) / 1000;
    if (s < LEADER_S) {
      const k = s / LEADER_S; leaderL = S.maxLeader * k * k * (3 - 2 * k);
      setStatus("Leader descending (slowed down)");
    } else {
      const f = s - LEADER_S;
      flash = Math.exp(-f / 0.15) + (f > 0.1 ? 0.35 * Math.exp(-(f - 0.1) / 0.1) : 0);
      setStatus("Return stroke");
    }
  } else if (player.introStart >= 0) {
    player.introStart = -1;
    syncAudio();
  }

  if (player.playing && !intro) {
    if (soundOn && !audio.paused && !audioBroken && !audio.seeking) player.t = audio.currentTime;
    else player.t += dt * player.speed;
    const end = player.mode === "recon" ? S.recT1 : S.strikeEnd;
    if (player.t >= end) {
      player.t = end; setPlaying(false);
      if (player.mode === "recon" && cur === 2) setTimeout(() => cur === 2 && !player.playing && show(3), 900);
    }
  }
  const t = player.t;
  let level = 1;
  if (player.mode === "strike") {
    if (!intro) heardGlow(S, t, 0.25);
    S.recon.visible = false;
    drawRays(S, -1);
    for (const m of S.shells) {
      const r = 0.343 * t;
      m.visible = !intro && t > 0.02 && r < 3.5;
      m.scale.setScalar(Math.max(r, 1e-3));
      m.material.uniforms.opacity.value = 0.1 * (1 - r / 3.5);
    }
    if (!intro) {
      if (!player.playing && t === 0) setStatus("Press play to hear the thunder");
      else if (t < S.firstArrival) setStatus(`Sound arrives in ${(S.firstArrival - t).toFixed(1)} s`);
      else if (t <= S.lastArrival + 0.3) setStatus("Thunder arriving · teal is what you hear");
      else setStatus("The rumble is over");
    }
  } else {
    level = 0.32;
    S.shells.forEach((m) => (m.visible = false));
    S.recon.visible = true;
    if (!finalHold) {
      heardGlow(S, t, S.D.window_s * 0.75);
      const k = showPoints(S, t, 0.25 * Math.max(player.speed, 1));
      drawRays(S, t);
      setStatus(`Rebuilding · ${k} of ${S.np} points`);
      const el = $("live-count"); if (el) el.textContent = String(k);
    } else {
      S.glow.fill(0); showPoints(S, Infinity, -1); drawRays(S, -1);
      setStatus(`${S.np} points · median error ${fmt(S.M.median_error_m)} m`);
      level = truthLevel;
    }
  }
  paintStrike(S, { level, leaderL, flash });

  // The sound level at the array drives the microphone marker and its ring.
  const W = S.D.waveform, i = Math.floor(t / W.dt_s);
  const lvl = !intro && i >= 0 && i < W.max.length ? Math.min(Math.max(W.max[i], -W.min[i]) * 3, 1) : 0;
  micSprite.material.color.set(0xe6e9ef).lerp(HEARD, lvl);
  micSprite.scale.setScalar(0.01 * (1 + 1.2 * lvl));
  micRing.scale.setScalar(0.06 + 0.45 * lvl);
  micRing.material.opacity = 0.8 * lvl;
  return flash;
}

// ---------------------------------------------------------------- all strikes at once
const storm = { built: false, group: new THREE.Group(), ranges: [], flashes: [], next: 0, dirty: false };
storm.group.visible = false;
scene.add(storm.group);
async function buildStorm(progress) {
  if (storm.built) return;
  let done = 0;
  const all = await Promise.all(LIB.strikes.map((s) => fetchStrike(s.id).then((d) => { progress(++done / NS); return d; })));
  const pos = [], col = [], pts = [], pcol = [];
  for (const D of all) {
    const nodes = D.channel.nodes.map(toV), start = pos.length / 6;
    D.channel.segments.forEach(([a, b], i) => {
      pos.push(...nodes[a].toArray(), ...nodes[b].toArray());
      const c = TRUTH[D.channel.kind[i]].clone().multiplyScalar(0.42).toArray();
      col.push(...c, ...c);
    });
    storm.ranges.push([start, pos.length / 6]);
    D.points.xyz.forEach((p, i) => { pts.push(toV(p)); pcol.push(...errColor(D.points.error_m[i]).multiplyScalar(0.65).toArray()); });
  }
  storm.base = Float32Array.from(col);
  storm.col = Float32Array.from(col);
  storm.geom = new LineSegmentsGeometry().setPositions(pos).setColors(storm.col);
  storm.colBuf = storm.geom.attributes.instanceColorStart.data;
  storm.lines = new LineSegments2(storm.geom, stormMat);
  const pg = new THREE.BufferGeometry().setFromPoints(pts);
  pg.setAttribute("color", new THREE.Float32BufferAttribute(pcol, 3));
  storm.points = new THREE.Points(pg, ptMat(4));
  storm.group.add(storm.lines, storm.points);
  storm.built = true;
}
function stormFrame(now) {
  // Now and then a strike flashes, so the field doesn't sit still.
  if (now > storm.next && !reduceMotion) {
    storm.flashes.push({ k: Math.floor(Math.random() * NS), t0: now });
    storm.next = now + 900 + Math.random() * 1800;
  }
  if (!storm.flashes.length && !storm.dirty) return { flash: 0, at: null };
  storm.col.set(storm.base);
  storm.dirty = storm.flashes.length > 0;
  let f = 0, at = null;
  storm.flashes = storm.flashes.filter((fl) => {
    const e = Math.exp(-(now - fl.t0) / 180);
    if (e < 0.02) return false;
    const [a, b] = storm.ranges[fl.k];
    for (let i = a * 6; i < b * 6; i++) storm.col[i] += (1 - storm.col[i]) * Math.min(e * 1.1, 1);
    if (e > f) { f = e; at = fl.k; }
    return true;
  });
  storm.colBuf.array.set(storm.col); storm.colBuf.needsUpdate = true;
  const s = at === null ? null : LIB.strikes[at];
  const a = s ? (s.azimuth_deg * Math.PI) / 180 : 0;
  return { flash: f * 0.3, at: s ? toV([s.distance_km * 1000 * Math.sin(a), s.distance_km * 1000 * Math.cos(a), 6000]) : null };
}

// ---------------------------------------------------------------- story
let cur = 0;
const panelEl = $("panel"), stepEl = $("step"), tabs = $("tabs");
const steps = [
  {
    tab: "Strike",
    title: () => `${TYPE[S.A.preset]}, ${S.A.distance_km.toFixed(1)} km ${COMPASS[dirIdx(S.A.azimuth_deg)]}`,
    html: () => `<p>A <b>${S.A.length_km.toFixed(1)} km</b> channel starting <b>${S.A.top_km.toFixed(1)} km</b> up, generated by the
      simulator. It grows in 10 m steps whose turns and branching follow measurements of real lightning${S.A.preset === "with_incloud" ? ", with a long section inside the cloud (violet)" : ""}.</p>
      <p class="small">Blue is the true channel. The reconstruction never sees it. There are ${NS} strikes; press New strike for another.</p>
      <div class="row"><button class="primary" id="go">Play the thunder</button></div>`,
    enter: (fresh) => {
      $("go").onclick = () => show(1);
      if (fresh) { playStrike(true); setPlaying(false); player.introStart = reduceMotion ? -1 : performance.now(); }
    },
  },
  {
    tab: "Thunder",
    title: () => "The thunder",
    html: () => `<p>Every half-metre of channel sends out a pressure pulse. The simulator traces each one through this
      strike's air: <b>${S.A.temperature_c.toFixed(0)} °C</b> at the ground, wind <b>${S.A.wind_mps.toFixed(1)} m/s</b> from the ${COMPASS[dirIdx(S.A.wind_from_deg)]}.</p>
      <p>Near parts are heard first and far parts last, so a flash lasting microseconds rumbles for
      <b>${Math.round(S.lastArrival - S.firstArrival)} s</b>. <span style="color:var(--heard)">Teal</span> marks the part you are hearing.</p>
      <details><summary>What the simulation includes</summary><ul>
        <li>Pulse shape set by the energy per metre of channel (Few's thunder model).</li>
        <li>Refraction by temperature and wind, from exact ray tracing.</li>
        <li>Air absorption (ISO 9613-1) and the echo off the ground.</li>
        <li>${pct(S.M.shadow_fraction)} of this channel is in the acoustic shadow and is never heard.</li>
        <li>Five microphones 50 m apart, with realistic noise, clock and position errors.</li>
      </ul></details>
      <div class="row"><button id="replay">Replay</button><button class="primary" id="go">Rebuild from sound</button></div>`,
    enter: (fresh) => { $("replay").onclick = () => playStrike(true); $("go").onclick = () => show(2); if (fresh) playStrike(true); },
  },
  {
    tab: "Rebuild",
    title: () => "Rebuilding it from sound",
    html: () => `<p>Only the five recordings and the time of the flash go in. In each 0.1 s slice, the tiny delays between
      microphones give a direction, and the time since the flash gives a distance. Together they place a point.</p>
      <p><b class="num live" id="live-count">0</b> of <span class="num">${S.np}</span> points placed. Amber lines are the directions measured at the array.</p>
      <details><summary>Method</summary><ul>
        <li>Steered response power (method B) separates up to three sounds arriving at once.</li>
        <li>Distances are traced back through the true atmosphere. With the wind unknown, errors grow to about 130 m.</li>
      </ul></details>
      <div class="row"><button id="replay">Replay</button><button class="primary" id="go">See the result</button></div>`,
    enter: (fresh) => { $("replay").onclick = playRecon; $("go").onclick = () => show(3); if (fresh) playRecon(); else { player.mode = "recon"; finalHold = true; } },
  },
  {
    tab: "Result",
    title: () => "How close it got",
    html: () => `<dl class="stats">
        <dt>${fmt(S.M.median_error_m)} m</dt><dd>median distance from the true channel</dd>
        <dt>${fmt(S.M.p90_error_m, 0)} m</dt><dd>90% of points are closer than this</dd>
        <dt>${pct(S.M.coverage_main_50m)}</dt><dd>of the main channel found (within 50 m)</dd>
        ${S.M.coverage_branch_50m === null ? "" : `<dt>${pct(S.M.coverage_branch_50m)}</dt><dd>of the branches found</dd>`}
      </dl>
      <div class="row">
        <button id="t-truth" aria-pressed="true">True channel</button>
        <button id="t-recon" aria-pressed="true">Rebuilt</button>
        <button id="t-sigma" aria-pressed="false">Uncertainty</button>
        <button id="t-side" aria-pressed="false">Side by side</button>
      </div>
      <p class="small">Missed parts are mostly in the acoustic shadow, or arrive at the same moment as louder sound.
      Across all ${NS} strikes the median error is ${LIB.summary.median_of_medians_m.toFixed(1)} m.</p>
      <div class="row"><button class="primary" id="next-strike">New strike</button><button id="all">All ${NS} strikes</button></div>`,
    enter: () => {
      player.mode = "recon"; setPlaying(false); player.t = S.recT1; finalHold = true; truthLevel = 0.5;
      const tog = (id, fn) => { const b = $(id); b.onclick = () => { const on = b.getAttribute("aria-pressed") !== "true"; b.setAttribute("aria-pressed", String(on)); fn(on); invalidate(); }; };
      tog("t-truth", (on) => (S.line.visible = on));
      tog("t-recon", (on) => (S.points.visible = on));
      tog("t-sigma", (on) => (S.sigma.visible = on));
      tog("t-side", (on) => { S.recon.position.copy(on ? S.sideOffset : new THREE.Vector3()); truthLevel = on ? 0.9 : 0.5; });
      $("next-strike").onclick = generate; $("all").onclick = enterStorm;
    },
  },
];
tabs.innerHTML = steps.map((s, i) => `<button data-step="${i}"><span>${i + 1}</span>${s.tab}</button>`).join("");
tabs.querySelectorAll("button").forEach((b) => (b.onclick = () => show(parseInt(b.dataset.step, 10))));

function show(k, fresh = true) {
  if (stormOn) leaveStorm(false);
  cur = Math.max(0, Math.min(steps.length - 1, k));
  const s = steps[cur];
  setPlaying(false); finalHold = false;
  S.line.visible = true; S.points.visible = true; S.sigma.visible = false; S.recon.position.set(0, 0, 0);
  view = cur >= 2 ? { t0: S.recT0, t1: S.recT1 } : { t0: 0, t1: S.dur };
  if (cur < 2) player.mode = "strike";
  stepEl.innerHTML = `<h2>${s.title()}</h2>${s.html()}`;
  stepEl.scrollTop = 0;
  tabs.hidden = false;
  tabs.querySelectorAll("button").forEach((b, i) => (i === cur ? b.setAttribute("aria-current", "step") : b.removeAttribute("aria-current")));
  s.enter(fresh);
  factsStrike();
  invalidate();
}
$("collapse").onclick = () => {
  const folded = panelEl.classList.toggle("folded");
  $("collapse").textContent = folded ? "Show" : "Hide";
  $("collapse").setAttribute("aria-expanded", String(!folded));
};

// ---------------------------------------------------------------- loading strikes
let bag = [], loadSeq = 0;
async function openStrike(id, step = 0) {
  const seq = ++loadSeq;
  let D;
  try { D = await fetchStrike(id); }
  catch { setStatus("Couldn't load that strike. Try again."); return; }
  if (seq !== loadSeq) return;  // a newer request won
  if (stormOn) leaveStorm(false);
  if (S) disposeStrike(S);
  S = buildStrike(D);
  scene.add(S.group);
  audio.src = `viewer/bolts/${id}.mp3`;
  audioBroken = false;
  groundGlow.position.set(S.strike.x, 0.002, S.strike.z);
  placeRingLabels(S.along);
  buildClouds(S.focus.x, S.focus.z, 9, S.box.max.y, narrow() ? 40 : 70);
  flyTo(S.home, S.focus);
  history.replaceState(null, "", `#strike=${id}`);
  show(step);
  if (!bag.length) refillBag();
  fetchStrike(bag[bag.length - 1]).catch(() => {});  // warm the cache for the next New strike
}
function refillBag() {
  bag = LIB.strikes.map((s) => s.id).filter((i) => !S || i !== S.A.id);
  for (let i = bag.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [bag[i], bag[j]] = [bag[j], bag[i]]; }
}
function generate() {
  if (!bag.length) refillBag();
  openStrike(bag.pop(), 1);
}
$("gen").onclick = generate;

// ---------------------------------------------------------------- all-strikes view
async function enterStorm() {
  if (stormOn) return;
  stormOn = true;
  setPlaying(false);
  $("storm").setAttribute("aria-pressed", "true");
  $("player").hidden = true; $("hint").hidden = true;
  tabs.hidden = true;
  stepEl.innerHTML = `<h2>All ${NS} strikes</h2>
    <p>Every strike in the library, each rebuilt from its own recording. None were left out.</p>
    <div class="progress"><i id="prog"></i></div><div id="storm-body" hidden></div>`;
  try { await buildStorm((p) => { const el = $("prog"); if (el) el.style.width = `${p * 100}%`; }); }
  catch { stepEl.insertAdjacentHTML("beforeend", "<p>Couldn't load every strike. Check your connection and try again.</p>"); return; }
  if (!stormOn) return;
  S.group.visible = false; S.labels.forEach((l) => (l.visible = false));
  storm.group.visible = true;
  micSprite.material.color.set(0xe6e9ef); micSprite.scale.setScalar(0.01); micRing.material.opacity = 0;
  groundGlow.material.opacity = 0;
  buildClouds(0, 0, 10, 6.6, narrow() ? 50 : 90);
  const s = LIB.summary;
  $("storm-body").innerHTML = `<dl class="stats">
      <dt>${s.median_of_medians_m.toFixed(1)} m</dt><dd>median error per strike</dd>
      <dt>${pct(s.median_coverage_main)}</dt><dd>of a typical main channel found</dd>
      <dt>${s.total_points.toLocaleString("en-US")}</dt><dd>points rebuilt in total</dd>
    </dl>
    <p class="small">Strikes by median error (m)</p>
    <canvas id="hist" aria-label="Histogram of median error per strike, in metres"></canvas>
    <div class="row"><button id="s-truth" aria-pressed="true">True channels</button><button id="s-recon" aria-pressed="true">Rebuilt</button></div>
    <div class="list" id="slist">${LIB.strikes.map((x) => `<button data-id="${x.id}"${x.id === S.A.id ? ' class="cur"' : ""}>
      <span>${x.id + 1}. ${x.distance_km.toFixed(1)} km ${SHORT[dirIdx(x.azimuth_deg)]}, ${x.preset === "with_incloud" ? "in-cloud" : x.preset}</span>
      <span class="num">${fmt(x.median_error_m)} m</span></button>`).join("")}</div>
    <p class="small">Select a strike to watch it. All runs know the atmosphere.</p>
    <div class="row"><button id="s-back">Back to strike ${S.A.id + 1}</button></div>`;
  $("storm-body").hidden = false;
  document.querySelector(".progress").hidden = true;
  drawHist();
  document.querySelectorAll("#slist button").forEach((b) => (b.onclick = () => openStrike(parseInt(b.dataset.id, 10), 1)));
  $("s-back").onclick = () => leaveStorm(true);
  const tog = (id, obj) => { const b = $(id); b.onclick = () => { const on = b.getAttribute("aria-pressed") !== "true"; b.setAttribute("aria-pressed", String(on)); obj.visible = on; invalidate(); }; };
  tog("s-truth", storm.lines); tog("s-recon", storm.points);
  factsStorm();
  controls.autoRotate = !reduceMotion;
  flyTo(new THREE.Vector3(narrow() ? 19 : 13, narrow() ? 14 : 9.5, narrow() ? 21 : 15), new THREE.Vector3(0, 2.4, 0), 2);
  history.replaceState(null, "", "#storm");
}
function leaveStorm(restore) {
  stormOn = false;
  controls.autoRotate = false;
  $("storm").setAttribute("aria-pressed", "false");
  $("player").hidden = false; $("hint").hidden = false;
  storm.group.visible = false;
  S.group.visible = true; S.labels.forEach((l) => (l.visible = true));
  buildClouds(S.focus.x, S.focus.z, 9, S.box.max.y, narrow() ? 40 : 70);
  if (restore) { flyTo(S.home, S.focus); show(cur, false); history.replaceState(null, "", `#strike=${S.A.id}`); }
  invalidate();
}
$("storm").onclick = () => (stormOn ? leaveStorm(true) : enterStorm());
function drawHist() {
  const c = $("hist"); if (!c) return;
  const dpr = Math.min(devicePixelRatio, 2), w = c.clientWidth, h = c.clientHeight;
  c.width = w * dpr; c.height = h * dpr;
  const g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const bins = new Array(10).fill(0);  // 1 m bins, the last one open-ended
  LIB.strikes.forEach((s) => { if (s.median_error_m !== null) bins[Math.min(9, Math.floor(s.median_error_m))]++; });
  const top = Math.max(...bins), bw = w / bins.length;
  g.font = "10.5px 'IBM Plex Mono', monospace"; g.textAlign = "center";
  bins.forEach((n, i) => {
    const bh = (n / top) * (h - 30);
    g.fillStyle = errColor(i + 0.5).getStyle();
    g.fillRect(i * bw + 3, h - 16 - bh, bw - 6, bh);
    g.fillStyle = "#6c7588";
    g.fillText(i === 9 ? "9+" : String(i), i * bw + bw / 2, h - 3);
    if (n) { g.fillStyle = "#a2abbc"; g.fillText(String(n), i * bw + bw / 2, h - 20 - bh); }
  });
}

// ---------------------------------------------------------------- controls
document.querySelectorAll("[data-speed]").forEach((b) => (b.onclick = () => setSpeed(parseFloat(b.dataset.speed))));
$("play").onclick = () => {
  if (!S || stormOn) return;
  if (player.playing) return setPlaying(false);
  const end = player.mode === "recon" ? S.recT1 : S.strikeEnd;
  if (player.t >= end - 0.05) return player.mode === "recon" ? playRecon() : playStrike(true);
  finalHold = false;
  player.introStart = -1;
  setPlaying(true);
};
$("sound").onclick = () => {
  soundOn = !soundOn; audioBroken = false;
  $("sound").setAttribute("aria-pressed", String(soundOn));
  syncAudio();
};
addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.code === "Space" && e.target.tagName !== "BUTTON" && e.target.tagName !== "SUMMARY") { e.preventDefault(); $("play").click(); }
  else if (e.key === "n" || e.key === "N") generate();
  else if (e.key === "Escape" && stormOn) leaveStorm(true);
  else if (e.key === "ArrowRight" && !stormOn) show(cur + 1);
  else if (e.key === "ArrowLeft" && !stormOn) show(cur - 1);
});
document.addEventListener("visibilitychange", () => { if (document.hidden && player.playing) setPlaying(false); });

// ---------------------------------------------------------------- main loop
let last = performance.now();
function tick() {
  requestAnimationFrame(tick);
  const now = performance.now();  // one clock everywhere (rAF timestamps can use another timebase)
  const dt = Math.min((now - last) / 1000, 0.1); last = now;
  if (tween) {
    const k = Math.min((now - tween.start) / tween.dur, 1), e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
    camera.position.lerpVectors(tween.p0, tween.p1, e); controls.target.lerpVectors(tween.q0, tween.q1, e);
    if (k >= 1) tween = null;
    dirty = true;
  }
  controls.update();  // damping and auto-rotate; fires "change" (and so marks dirty) while moving
  if (!(inIntro(now) || player.playing || stormOn || dirty)) return;
  dirty = false;

  let flash = 0, at = null;
  if (stormOn && storm.built) ({ flash, at } = stormFrame(now));
  else if (S && !stormOn) { flash = strikeFrame(now, dt); at = S.top; }
  skyMat.uniforms.flash.value = flash;
  if (at) skyMat.uniforms.flashDir.value.copy(at).sub(camera.position).normalize();
  lightClouds(flash, at);
  groundMat.color.setRGB(0.059 + 0.1 * flash, 0.078 + 0.11 * flash, 0.122 + 0.15 * flash, THREE.SRGBColorSpace);
  if (!stormOn) { groundGlow.material.opacity = Math.min(flash, 1); groundGlow.scale.setScalar(2.5 + 2 * flash); }
  $("flash").style.opacity = reduceMotion ? "0" : String(Math.min(flash * 0.25, 0.3));
  bloom.strength = (stormOn ? 0.4 : 0.7) + 0.9 * flash;
  sky.position.copy(camera.position);
  drawWave();
  composer.render();
  labels.render(scene, camera);
}

// ---------------------------------------------------------------- start
const q = new URLSearchParams(location.hash.slice(1));
const wantStorm = location.hash.includes("storm");
setPlaying(false);
setSpeed(2);
const first = q.has("strike") ? Math.min(NS - 1, Math.max(0, parseInt(q.get("strike"), 10) || 0)) : 0;
await openStrike(first, 0);
if (!S) fail("Couldn't load the first strike. Check your connection and reload the page.");
const startStep = Math.max(0, (parseInt(q.get("step") ?? "1", 10) || 1) - 1);
if (startStep) show(startStep);
camera.position.copy(S.home); controls.target.copy(S.focus); tween = null;
if (q.has("t")) { setPlaying(false); seek(parseFloat(q.get("t"))); if (cur === 3 && q.get("side") === "1") $("t-side").click(); }
if (wantStorm) enterStorm();
$("loading").style.opacity = "0";
setTimeout(() => $("loading").remove(), 450);
requestAnimationFrame(tick);
window.__viewer = { ready: true, strikes: NS, get t() { return player.t; }, get step() { return cur; } };
