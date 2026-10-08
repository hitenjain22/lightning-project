// Lightning from Thunder: interactive viewer for the simulated strike library.
// Data: viewer/bolts/index.json + one <id>.json / <id>.mp3 per strike, written by
// scripts/make_viewer_data.py from the project's real pipeline. Scene units are kilometres;
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
const KM = 1 / 1000;
const toV = (p) => new THREE.Vector3(p[0] * KM, p[2] * KM, -p[1] * KM);
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
const narrow = () => innerWidth <= 820;

const LIB = await (await fetch("viewer/bolts/index.json")).json();
const NS = LIB.strikes.length;
$("storm-n").textContent = String(NS);

// ---------------------------------------------------------------- renderer & camera
const host = $("scene");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
host.appendChild(renderer.domElement);
const labels = new CSS2DRenderer();
labels.setSize(innerWidth, innerHeight);
Object.assign(labels.domElement.style, { position: "absolute", top: "0", pointerEvents: "none" });
host.appendChild(labels.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x0c0f20, 0.03);
const camera = new THREE.PerspectiveCamera(45, innerWidth / innerHeight, 0.01, 300);
camera.position.set(8, 3, 10);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.maxPolarAngle = Math.PI * 0.495;  // stay above the ground
controls.minDistance = 0.3;
controls.maxDistance = 45;
controls.autoRotateSpeed = 0.35;

const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(innerWidth, innerHeight), 0.85, 0.55, 0.2);
composer.addPass(bloom);
composer.addPass(new OutputPass());

function frame() {
  const w = innerWidth, h = innerHeight;
  camera.aspect = w / h;
  // Wide screens: story panel on the left, facts on the right: nudge the scene right of centre.
  // Narrow screens: the story sheet covers the bottom: shift the scene up.
  if (!narrow()) camera.setViewOffset(w, h, -70, 0, w, h);
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
});

let tween = null;
function flyTo(pos, target, dur = 1.8) {
  tween = { p0: camera.position.clone(), q0: controls.target.clone(), p1: pos, q1: target, start: performance.now(), dur: dur * 1000 };
}
controls.addEventListener("start", () => { tween = null; controls.autoRotate = false; });

// ---------------------------------------------------------------- textures
function canvasTex(size, draw) {
  const c = document.createElement("canvas"); c.width = c.height = size;
  draw(c.getContext("2d"), size);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t;
}
const dotTex = canvasTex(64, (g) => {
  const r = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  r.addColorStop(0, "rgba(255,255,255,1)"); r.addColorStop(0.42, "rgba(255,255,255,1)");
  r.addColorStop(0.6, "rgba(255,255,255,0.35)"); r.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = r; g.fillRect(0, 0, 64, 64);
});
const glowTex = canvasTex(128, (g) => {
  const r = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  r.addColorStop(0, "rgba(255,255,255,0.9)"); r.addColorStop(0.25, "rgba(200,215,255,0.35)"); r.addColorStop(1, "rgba(160,180,255,0)");
  g.fillStyle = r; g.fillRect(0, 0, 128, 128);
});
// Puffy cloud blobs: many soft circles, denser near the centre. Seeded so the sky is stable.
let seed = 7;
const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
const cloudTex = [0, 1, 2, 3].map(() => canvasTex(128, (g) => {
  for (let i = 0; i < 26; i++) {
    const a = rand() * Math.PI * 2, d = Math.sqrt(rand()) * 34, x = 64 + Math.cos(a) * d, y = 64 + Math.sin(a) * d * 0.7;
    const rr = 16 + rand() * 22, r = g.createRadialGradient(x, y, 0, x, y, rr);
    r.addColorStop(0, "rgba(255,255,255,0.22)"); r.addColorStop(1, "rgba(255,255,255,0)");
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
      // Colours picked in sRGB; converted to linear at the end (the output pass re-encodes to sRGB).
      vec3 horizon = vec3(0.07, 0.085, 0.17), zenith = vec3(0.012, 0.016, 0.045), below = vec3(0.045, 0.055, 0.1);
      vec3 c = mix(horizon, zenith, smoothstep(0.0, 0.55, d.y));
      c = mix(c, below, smoothstep(0.0, -0.12, d.y));
      float f = flash * (0.25 + 0.75 * pow(max(dot(d, flashDir), 0.0), 4.0));
      c += vec3(0.5, 0.56, 0.95) * f * 0.7;
      gl_FragColor = vec4(pow(c, vec3(2.2)), 1.0); }`,
});
const sky = new THREE.Mesh(new THREE.SphereGeometry(120, 32, 16), skyMat);
scene.add(sky);

const groundMat = new THREE.MeshBasicMaterial({ color: 0x070a13 });
const ground = new THREE.Mesh(new THREE.PlaneGeometry(400, 400), groundMat);
ground.rotation.x = -Math.PI / 2; ground.position.y = -0.003;
scene.add(ground);
const grid = new THREE.GridHelper(40, 40, 0x24335e, 0x141d38);
grid.material.transparent = true; grid.material.opacity = 0.75;
scene.add(grid);
const groundGlow = new THREE.Mesh(new THREE.PlaneGeometry(1, 1),
  new THREE.MeshBasicMaterial({ map: glowTex, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, opacity: 0 }));
groundGlow.rotation.x = -Math.PI / 2; groundGlow.position.y = 0.002;
scene.add(groundGlow);

function label(text, pos, cls = "label3d") {
  const el = document.createElement("div"); el.className = cls; el.textContent = text;
  const o = new CSS2DObject(el); o.position.copy(pos); return o;
}
// Range rings around the array.
for (const r of [1, 2, 4, 6]) {
  const pts = [];
  for (let i = 0; i <= 128; i++) { const a = (i / 128) * Math.PI * 2; pts.push(new THREE.Vector3(Math.cos(a) * r, 0.001, Math.sin(a) * r)); }
  const ring = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({ color: 0x3b4f8f, transparent: true, opacity: 0.55 }));
  scene.add(ring);
  scene.add(label(`${r} km`, new THREE.Vector3(r * 0.707, 0, r * 0.707), "label3d dim"));
}
scene.add(label("N", new THREE.Vector3(0, 0, -6.4), "label3d dim"));

// The microphone array (50 m across: one marker at this scale) and a ring that pulses with the sound.
const micSprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: dotTex, color: 0xffffff, sizeAttenuation: false, depthTest: false, transparent: true }));
micSprite.scale.setScalar(0.011); micSprite.position.set(0, 0.002, 0); micSprite.renderOrder = 10;
scene.add(micSprite);
const micRing = new THREE.Mesh(new THREE.RingGeometry(0.92, 1, 64),
  new THREE.MeshBasicMaterial({ color: 0x4dffb5, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
micRing.rotation.x = -Math.PI / 2; micRing.position.y = 0.004;
scene.add(micRing);
const micLabel = label("5 microphones · 50 m array", new THREE.Vector3(0, 0.25, 0));
scene.add(micLabel);

// Cloud deck: soft sprites around the channel top (drawn there so the whole channel stays visible;
// real cloud bases are lower). Lit by the flash, more strongly near it.
let clouds = [];
function buildClouds(cx, cz, radius, height, count) {
  clouds.forEach((c) => { scene.remove(c); c.material.dispose(); });
  clouds = [];
  for (let i = 0; i < count; i++) {
    const a = rand() * Math.PI * 2, d = Math.sqrt(rand()) * radius;
    const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: cloudTex[i % 4], color: 0x222a4a, transparent: true,
      opacity: 0.45 + rand() * 0.35, depthWrite: false, fog: true }));
    s.position.set(cx + Math.cos(a) * d, height + 0.2 + rand() * 1.2, cz + Math.sin(a) * d);
    const k = 2.2 + rand() * 3.2; s.scale.set(k * 1.6, k, 1);
    s.material.rotation = rand() * Math.PI;
    scene.add(s); clouds.push(s);
  }
}
const cloudBase = new THREE.Color(0x222a4a), cloudLit = new THREE.Color(0xa9b6ff), tmpC = new THREE.Color();
function lightClouds(f, at) {
  for (const s of clouds) {
    const w = f * Math.exp(-s.position.distanceTo(at) / 3.5);
    s.material.color.copy(cloudBase).lerp(cloudLit, Math.min(w, 1));
  }
}

// Rain: streaks in a box that travels with the camera.
const RAIN = reduceMotion ? 0 : 1300, RBOX = 0.5;
const rainPos = new Float32Array(RAIN * 6);
const rainOff = new Float32Array(RAIN * 3).map(() => (Math.random() - 0.5) * RBOX);
const rainGeom = new THREE.BufferGeometry();
rainGeom.setAttribute("position", new THREE.BufferAttribute(rainPos, 3));
const rain = new THREE.LineSegments(rainGeom, new THREE.LineBasicMaterial({ color: 0x8796c2, transparent: true, opacity: 0.1, fog: false }));
rain.frustumCulled = false;
scene.add(rain);
function rainFrame(dt) {
  if (!RAIN) return;
  const c = camera.position, fall = 0.09 * dt;
  for (let i = 0; i < RAIN; i++) {
    let y = rainOff[i * 3 + 1] - fall;
    if (y < -RBOX / 2) y += RBOX;
    rainOff[i * 3 + 1] = y;
    const x = c.x + rainOff[i * 3], z = c.z + rainOff[i * 3 + 2], yy = c.y + y;
    rainPos.set([x, yy, z, x + 0.001, yy + 0.007, z], i * 6);
  }
  rainGeom.attributes.position.needsUpdate = true;
}

// ---------------------------------------------------------------- colours
const BASE = [new THREE.Color(0x6fb6ff), new THREE.Color(0x8e95ff).multiplyScalar(0.85), new THREE.Color(0xb48cff).multiplyScalar(0.8)];
const WHITE = new THREE.Color(0xffffff), MINT = new THREE.Color(0x4dffb5), LEADER = new THREE.Color(0x8a8cff);
const WARM = ["#fff2b3", "#ffb000", "#ff5a1f", "#e0105a"].map((h) => new THREE.Color(h));
function errColor(e) {  // low error = hot white-gold, 15 m and beyond = crimson
  const x = Math.min(Math.max((e ?? 15) / 15, 0), 1) * 3, i = Math.min(Math.floor(x), 2);
  return WARM[i].clone().lerp(WARM[i + 1], x - i);
}
const chanMat = new LineMaterial({ vertexColors: true, linewidth: 2.3, resolution: new THREE.Vector2(innerWidth, innerHeight) });
const stormMat = new LineMaterial({ vertexColors: true, linewidth: 1.5, resolution: new THREE.Vector2(innerWidth, innerHeight) });
const ptMat = (size) => new THREE.PointsMaterial({ size, map: dotTex, vertexColors: true, sizeAttenuation: false,
  transparent: true, depthWrite: false, alphaTest: 0.05 });
const shellGeom = new THREE.SphereGeometry(1, 48, 24);
const shellMat = new THREE.ShaderMaterial({
  uniforms: { opacity: { value: 0 } },
  vertexShader: `varying float rim; void main() { vec4 mv = modelViewMatrix * vec4(position, 1.0);
    rim = 1.0 - abs(dot(normalize(normalMatrix * normal), normalize(-mv.xyz))); gl_Position = projectionMatrix * mv; }`,
  fragmentShader: `uniform float opacity; varying float rim; void main() { gl_FragColor = vec4(0.45, 0.7, 1.0, opacity * pow(rim, 4.0)); }`,
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
});

// ---------------------------------------------------------------- one strike
const cache = new Map();
async function fetchStrike(id) {
  if (!cache.has(id)) cache.set(id, fetch(`viewer/bolts/${id}.json`).then((r) => r.json()));
  return cache.get(id);
}

function buildStrike(D) {
  const S = {};
  S.D = D; S.A = D.about; S.M = D.metrics;
  const group = new THREE.Group(); S.group = group;
  const nodes = D.channel.nodes.map(toV);
  // Segments sorted by leader distance, so "the leader has reached here" is a draw count.
  const raw = D.channel;
  const order = raw.segments.map((_, i) => i).sort((a, b) => raw.leader_m[a] - raw.leader_m[b]);
  S.n = order.length;
  S.kind = order.map((i) => raw.kind[i]);
  S.arrival = order.map((i) => raw.arrival_s[i]);
  S.leader = order.map((i) => raw.leader_m[i]);
  S.maxLeader = Math.max(1, ...S.leader);
  const pos = new Float32Array(S.n * 6);
  S.mid = [];
  order.forEach((i, k) => {
    const [a, b] = raw.segments[i];
    nodes[a].toArray(pos, k * 6); nodes[b].toArray(pos, k * 6 + 3);
    S.mid.push(nodes[a].clone().lerp(nodes[b], 0.5));
  });
  S.col = new Float32Array(S.n * 6);
  S.geom = new LineSegmentsGeometry().setPositions(pos).setColors(S.col);
  S.colBuf = S.geom.attributes.instanceColorStart.data;
  S.line = new LineSegments2(S.geom, chanMat);
  group.add(S.line);
  S.glow = new Float32Array(S.n);

  const box = new THREE.Box3().setFromPoints(nodes);
  S.box = box;
  S.center = box.getCenter(new THREE.Vector3());
  S.strike = toV(raw.strike);
  S.top = nodes.reduce((a, b) => (b.y > a.y ? b : a), nodes[0]);
  S.firstArrival = Math.min(...S.arrival.filter((a) => a !== null));
  S.lastArrival = Math.max(...S.arrival.filter((a) => a !== null));
  S.dur = D.waveform.duration_s;

  // Reconstructed points (time-ordered), the newest drawn larger, uncertainty bars, rays.
  const P = D.points; S.P = P; S.np = P.t.length;
  S.ptVec = P.xyz.map(toV);
  S.recon = new THREE.Group(); group.add(S.recon);
  S.ptGeom = new THREE.BufferGeometry().setFromPoints(S.ptVec);
  S.ptGeom.setAttribute("color", new THREE.Float32BufferAttribute(P.error_m.flatMap((e) => errColor(e).toArray()), 3));
  S.points = new THREE.Points(S.ptGeom, ptMat(9));
  S.newGeom = S.ptGeom.clone(); S.newGeom.setDrawRange(0, 0);
  S.points.add(new THREE.Points(S.newGeom, ptMat(22)));
  S.recon.add(S.points);
  const sg = [];
  P.sigma_m.forEach((s, i) => {
    const p = S.ptVec[i];
    [[2 * s[0] * KM, 0, 0], [0, 2 * s[2] * KM, 0], [0, 0, 2 * s[1] * KM]].forEach(([dx, dy, dz]) =>
      sg.push(p.x - dx, p.y - dy, p.z - dz, p.x + dx, p.y + dy, p.z + dz));
  });
  const sgGeom = new THREE.BufferGeometry(); sgGeom.setAttribute("position", new THREE.Float32BufferAttribute(sg, 3));
  S.sigma = new THREE.LineSegments(sgGeom, new THREE.LineBasicMaterial({ color: 0xffe2a8, transparent: true, opacity: 0.5 }));
  S.sigma.visible = false; S.recon.add(S.sigma);
  const RAYS = 10;
  S.rayGeom = new THREE.BufferGeometry();
  S.rayGeom.setAttribute("position", new THREE.BufferAttribute(new Float32Array(RAYS * 6), 3));
  S.rayGeom.setAttribute("color", new THREE.BufferAttribute(new Float32Array(RAYS * 6), 3));
  S.rays = new THREE.LineSegments(S.rayGeom, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, blending: THREE.AdditiveBlending }));
  S.rays.frustumCulled = false;
  group.add(S.rays);

  // Illustrative sound shells from points along the main channel.
  S.shells = [];
  const mains = S.mid.filter((_, k) => S.kind[k] === 0);
  for (let k = 0; k < mains.length; k += Math.max(1, Math.floor(mains.length / 9))) {
    const m = new THREE.Mesh(shellGeom, shellMat.clone());
    m.position.copy(mains[k]); m.visible = false; group.add(m); S.shells.push(m);
  }

  // CSS2D labels ignore their parent's visibility, so keep a handle to hide them in the storm view.
  S.labels = [label("strike point", S.strike.clone().add(new THREE.Vector3(0, -0.12, 0)))];
  S.labels.forEach((l) => group.add(l));

  // Framing: fit the bolt and the array in view, looking side-on to the line between them, a
  // little from above. The panels cover part of the screen, so fit to the free area.
  const flat = new THREE.Vector3(S.center.x, 0, S.center.z);
  S.along = flat.clone().normalize();
  S.side = new THREE.Vector3(-S.along.z, 0, S.along.x);
  const sphere = box.clone().expandByPoint(new THREE.Vector3(0, 0, 0)).getBoundingSphere(new THREE.Sphere());
  S.focus = sphere.center.clone();
  const vfov = THREE.MathUtils.degToRad(camera.fov), aspect = innerWidth / innerHeight;
  const hfov = 2 * Math.atan(Math.tan(vfov / 2) * aspect * (narrow() ? 0.95 : 0.6));
  const vfit = narrow() ? 2 * Math.atan(Math.tan(vfov / 2) * 0.55) : vfov * 0.82;
  const dist = sphere.radius / Math.sin(Math.min(hfov, vfit) / 2);
  const dir = S.side.clone().addScaledVector(S.along, -0.3).normalize();
  dir.y = 0.2; dir.normalize();
  S.home = S.focus.clone().addScaledVector(dir, dist);
  S.sideOffset = S.along.clone().multiplyScalar(-(0.8 * Math.hypot(box.max.x - box.min.x, box.max.z - box.min.z) + 0.8));
  S.recT0 = Math.max(0, Math.min(P.t[0] ?? S.firstArrival, S.firstArrival) - 0.6);
  S.recT1 = Math.min(S.dur, Math.max(P.t[S.np - 1] ?? S.lastArrival, S.lastArrival) + 0.6);
  return S;
}

function disposeStrike(S) {
  scene.remove(S.group);
  S.group.traverse((o) => {
    if (o.isCSS2DObject) o.element.remove();
    if (o.geometry && o.geometry !== shellGeom) o.geometry.dispose();
    if (o.material && o.material !== chanMat) o.material.dispose?.();
  });
}

// Paint the true channel: base colour × level, leader phase, flash (white), heard-now glow (mint).
function paintStrike(S, { level = 1, leaderL = Infinity, flash = 0 }) {
  const c = new THREE.Color();
  let shown = 0;
  for (let k = 0; k < S.n; k++) {
    if (S.leader[k] > leaderL) break;
    shown = k + 1;
    if (leaderL < Infinity) {
      const tip = Math.max(0, 1 - (leaderL - S.leader[k]) / 300);
      c.copy(LEADER).multiplyScalar(0.45 + 0.55 * tip).lerp(WHITE, tip * 0.6);
    } else {
      c.copy(BASE[S.kind[k]]).multiplyScalar(level);
      if (S.glow[k] > 0) c.lerp(MINT, Math.min(S.glow[k], 1));
      if (flash > 0) c.lerp(WHITE, Math.min(flash, 1));
    }
    c.toArray(S.col, k * 6); c.toArray(S.col, k * 6 + 3);
  }
  S.geom.instanceCount = leaderL < Infinity ? shown : S.n;
  S.colBuf.array.set(S.col); S.colBuf.needsUpdate = true;
}

const countUpTo = (S, t) => { let k = 0; while (k < S.np && S.P.t[k] <= t) k++; return k; };
function showPoints(S, t, popWindow) {
  const k = countUpTo(S, t), j = countUpTo(S, t - popWindow);
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
    if (age > 1.0) break;
    pos.set([0, 0.002, 0], r * 6); S.ptVec[i].toArray(pos, r * 6 + 3);
    const c = WARM[1].clone().multiplyScalar(0.55 * (1 - age));
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
let S = null, soundOn = true, audioBroken = false, finalHold = false, stormOn = false;
const LEADER_S = 1.6, FLASH_S = 1.3;  // intro: leader descent, then the return-stroke flash (real seconds)
const inIntro = (now) => player.introStart >= 0 && now - player.introStart < (LEADER_S + FLASH_S) * 1000;

function setSpeed(s) {
  player.speed = s;
  document.querySelectorAll("[data-speed]").forEach((b) => b.setAttribute("aria-pressed", String(parseFloat(b.dataset.speed) === s)));
  syncAudio();
}
function setPlaying(on) {
  player.playing = on;
  if (on) audioBroken = false;  // retry audio on every play (autoplay may have been blocked before a click)
  $("play").innerHTML = on
    ? '<svg width="14" height="14" viewBox="0 0 14 14"><rect x="2" y="1" width="3.5" height="12" rx="1" fill="currentColor"/><rect x="8.5" y="1" width="3.5" height="12" rx="1" fill="currentColor"/></svg>'
    : '<svg width="14" height="14" viewBox="0 0 14 14"><path d="M3 1.5v11l9.5-5.5z" fill="currentColor"/></svg>';
  $("play").setAttribute("aria-label", on ? "Pause" : "Play");
  syncAudio();
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
  S.glow.fill(0);
  syncAudio();
}
function playStrike(withIntro = true) {
  finalHold = false;
  player.mode = "strike"; player.t = 0; S.glow.fill(0);
  setSpeed(2);
  player.introStart = withIntro && !reduceMotion ? performance.now() : -1;
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
  const grad = wctx.createLinearGradient(0, 0, Math.max(1, x(t)), 0);
  grad.addColorStop(0, "rgba(255,201,74,0.02)"); grad.addColorStop(1, "rgba(255,201,74,0.16)");
  wctx.fillStyle = grad; wctx.fillRect(0, 0, Math.max(0, x(t)), h);
  wctx.lineWidth = 1;
  for (const played of [false, true]) {
    wctx.strokeStyle = played ? "#a8d2ff" : "#5b6f9c"; wctx.beginPath();
    for (let i = 0; i < W.min.length; i++) {
      const tt = i * W.dt_s; if (tt < view.t0 || tt > view.t1 || (tt <= t) !== played) continue;
      const xx = x(tt);
      wctx.moveTo(xx, h / 2 - W.max[i] * h * 0.47); wctx.lineTo(xx, h / 2 - W.min[i] * h * 0.47 + 0.5);
    }
    wctx.stroke();
  }
  if (player.mode === "recon") {
    for (let i = 0; i < S.np; i++) {
      if (S.P.t[i] > t) break;
      if (S.P.t[i] < view.t0) continue;
      wctx.fillStyle = errColor(S.P.error_m[i]).getStyle();
      wctx.fillRect(x(S.P.t[i]) - 0.6, h - 6, 1.4, 6);
    }
  }
  wctx.strokeStyle = "#ffc94a"; wctx.lineWidth = 2; wctx.beginPath(); wctx.moveTo(x(t), 0); wctx.lineTo(x(t), h); wctx.stroke();
  $("tlabel").textContent = t.toFixed(1);
}
wave.addEventListener("click", (e) => {
  if (!S || stormOn) return;
  const r = wave.getBoundingClientRect();
  seek(view.t0 + ((e.clientX - r.left) / r.width) * (view.t1 - view.t0));
});

// ---------------------------------------------------------------- HUD & phase chip
const COMPASS = ["north", "north-east", "east", "south-east", "south", "south-west", "west", "north-west"];
const SHORT = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];
const dirIdx = (deg) => Math.round((((deg % 360) + 360) % 360) / 45) % 8;
const TYPE = { branched: "Branched strike", with_incloud: "Strike with an in-cloud channel", tortuous: "Single-channel strike" };
const pct = (x) => (x === null || x === undefined ? "n/a" : `${Math.round(x * 100)}%`);
const fmt = (x, d = 1) => (x === null || x === undefined ? "n/a" : x.toFixed(d));

function hudStrike() {
  const A = S.A;
  $("hud-kicker").textContent = "Strike";
  $("hud-id").textContent = `#${A.id + 1} of ${NS}`;
  $("hud-title").textContent = TYPE[A.preset] ?? A.preset;
  const rows = [
    ["Distance", `${A.distance_km.toFixed(1)} km ${SHORT[dirIdx(A.azimuth_deg)]}`],
    ["Channel", `${A.length_km.toFixed(1)} km long`],
    ["Starts", `${A.top_km.toFixed(1)} km up`],
    ["Branches", String(A.n_branches)],
    ["Air", `${A.temperature_c.toFixed(0)} °C, wind ${A.wind_mps.toFixed(1)} m/s ${SHORT[dirIdx(A.wind_from_deg)]}`],
    ["Thunder", `heard ${fmt(A.first_arrival_s)}–${fmt(S.lastArrival)} s`],
  ];
  if (cur >= 4) rows.push(["Rebuilt", `${A.n_points} pts · ${fmt(A.median_error_m)} m`]);
  $("hud-facts").innerHTML = rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
}
function hudStorm() {
  const s = LIB.summary;
  $("hud-kicker").textContent = "Library";
  $("hud-id").textContent = `${NS} strikes`;
  $("hud-title").textContent = "The whole storm";
  const d = LIB.strikes.map((x) => x.distance_km);
  $("hud-facts").innerHTML = [
    ["Strikes", `${s.n_strikes} (${s.n_reconstructed} rebuilt)`],
    ["Distances", `${Math.min(...d).toFixed(1)}–${Math.max(...d).toFixed(1)} km`],
    ["Points", s.total_points.toLocaleString()],
    ["Median error", `${s.median_of_medians_m.toFixed(1)} m`],
    ["Main channel", `${pct(s.median_coverage_main)} recovered`],
  ].map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
}
let phaseText = "";
function setPhase(html) { if (html !== phaseText) { $("phase").innerHTML = html; phaseText = html; } }

// ---------------------------------------------------------------- per-frame strike logic
function strikeFrame(now, dt) {
  const intro = inIntro(now);
  let flash = 0, leaderL = Infinity;
  if (intro) {
    const s = (now - player.introStart) / 1000;
    if (s < LEADER_S) {
      const k = s / LEADER_S; leaderL = S.maxLeader * (k * k * (3 - 2 * k));
      setPhase("Stepped leader working down from the cloud <span style='color:var(--muted)'>(slowed down)</span>");
    } else {
      const f = s - LEADER_S;
      flash = Math.min(1.2, Math.exp(-f / 0.16) + 0.4 * Math.exp(-Math.max(0, f - 0.1) / 0.1) * (f > 0.1 ? 1 : 0));
      setPhase("⚡ Return stroke: the channel heats to about 30,000 K in microseconds");
    }
  } else if (player.introStart >= 0) {
    player.introStart = -1;
    syncAudio();
  }

  if (player.playing && !intro) {
    if (soundOn && !audio.paused && !audioBroken && !audio.seeking) player.t = audio.currentTime;
    else player.t += dt * player.speed;
    const end = player.mode === "recon" ? S.recT1 : Math.min(S.lastArrival + 1, S.dur);
    if (player.t >= end) {
      player.t = end; setPlaying(false);
      if (player.mode === "recon" && cur === 4) setTimeout(() => cur === 4 && show(5), 900);
    }
  }
  const t = player.t;
  let level = 1;
  if (player.mode === "strike") {
    if (!intro) heardGlow(S, t, 0.25);
    S.recon.visible = false;
    drawRays(S, -1);
    S.shells.forEach((m) => {
      const r = 0.343 * t;
      m.visible = !intro && t > 0.02 && r < 3.5;
      m.scale.setScalar(Math.max(r, 1e-3));
      m.material.uniforms.opacity.value = 0.12 * Math.max(0, 1 - r / 3.5);
    });
    if (!intro && !finalHold) {
      if (!player.playing && t === 0) setPhase("⚡ Press <b>▶</b> to hear the thunder, or <b class='c-warm'>Generate strike</b> for a new one");
      else if (t < S.firstArrival) setPhase(`Sound travelling at ~${Math.round(S.A.sound_speed_mps ?? 343)} m/s · first thunder reaches the microphones in <span class="live">${(S.firstArrival - t).toFixed(1)}</span> s`);
      else if (t <= S.lastArrival + 0.3) setPhase("<span class='c-mint'>●</span> Thunder arriving: green marks the parts of the channel being heard right now");
      else setPhase("The last of the rumble has arrived");
    }
  } else {
    level = 0.32;
    S.shells.forEach((m) => (m.visible = false));
    S.recon.visible = true;
    if (!finalHold) {
      heardGlow(S, t, S.D.window_s * 0.75);
      const k = showPoints(S, t, 0.25 * Math.max(player.speed, 1));
      drawRays(S, t);
      setPhase(`Rebuilding from sound: <span class="live c-warm">${k}</span> of ${S.np} points placed`);
      const el = $("live-count"); if (el) el.textContent = String(k);
    } else {
      S.glow.fill(0); showPoints(S, Infinity, -1); drawRays(S, -1);
      setPhase(`${S.np} points rebuilt from five microphones · median error ${fmt(S.M.median_error_m)} m`);
      level = truthLevel;
    }
  }
  paintStrike(S, { level, leaderL, flash });

  // Sound level at the array drives the mic marker and the ground ring.
  const W = S.D.waveform, i = Math.floor(t / W.dt_s);
  const lvl = !intro && i >= 0 && i < W.max.length ? Math.min(Math.max(W.max[i], -W.min[i]) * 3, 1) : 0;
  micSprite.material.color.copy(WHITE).lerp(MINT, lvl);
  micSprite.scale.setScalar(0.011 * (1 + 1.5 * lvl));
  micRing.scale.setScalar(0.06 + 0.5 * lvl);
  micRing.material.opacity = 0.9 * lvl;
  return flash;
}

// ---------------------------------------------------------------- storm (all strikes)
const storm = { built: false, group: new THREE.Group(), ranges: [], col: null, base: null, flashes: [], next: 0 };
scene.add(storm.group);
storm.group.visible = false;
async function buildStorm(progress) {
  if (storm.built) return;
  let done = 0;
  const all = await Promise.all(LIB.strikes.map((s) => fetchStrike(s.id).then((d) => { progress(++done / NS); return d; })));
  const pos = [], col = [], pts = [], pcol = [];
  all.forEach((D) => {
    const nodes = D.channel.nodes.map(toV), start = pos.length / 6;
    D.channel.segments.forEach(([a, b], i) => {
      pos.push(...nodes[a].toArray(), ...nodes[b].toArray());
      const c = BASE[D.channel.kind[i]].clone().multiplyScalar(0.4).toArray();
      col.push(...c, ...c);
    });
    storm.ranges.push([start, pos.length / 6]);
    D.points.xyz.forEach((p, i) => { pts.push(toV(p)); pcol.push(...errColor(D.points.error_m[i]).multiplyScalar(0.6).toArray()); });
  });
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
  // Random strikes flash now and then, so the storm feels alive.
  if (now > storm.next && !reduceMotion) {
    storm.flashes.push({ k: Math.floor(Math.random() * NS), t0: now });
    storm.next = now + 500 + Math.random() * 1400;
  }
  if (!storm.flashes.length && !storm.dirty) return { f: 0, at: null };
  storm.col.set(storm.base);
  storm.dirty = storm.flashes.length > 0;
  let f = 0, at = null;
  storm.flashes = storm.flashes.filter((fl) => {
    const s = (now - fl.t0) / 1000, e = Math.exp(-s / 0.18);
    if (e < 0.02) return false;
    const [a, b] = storm.ranges[fl.k];
    for (let i = a * 6; i < b * 6; i++) storm.col[i] = storm.base[i] + (1 - storm.base[i]) * Math.min(e * 1.2, 1);
    if (e > f) { f = e; at = fl.k; }
    return true;
  });
  storm.colBuf.array.set(storm.col); storm.colBuf.needsUpdate = true;
  return { f: f * 0.3, at: at === null ? null : LIB.strikes[at] };
}

// ---------------------------------------------------------------- story
let cur = 0, truthLevel = 0.45;
const panelEl = $("panel"), stepEl = $("step"), dots = $("dots");
const steps = [
  {
    kicker: "1 · The strike",
    title: () => `${TYPE[S.A.preset]}, ${S.A.distance_km.toFixed(1)} km away`,
    html: () => `<p>The simulator generated this strike: a <b>${S.A.length_km.toFixed(1)} km</b> channel that starts
      <b>${S.A.top_km.toFixed(1)} km</b> up and hits the ground <b>${S.A.distance_km.toFixed(1)} km ${COMPASS[dirIdx(S.A.azimuth_deg)]}</b> of five microphones.
      Press <b class="c-warm">Generate strike</b> for another one: there are ${NS}, all different.</p>
      <ul><li>It grows as a random walk of 10 m steps. The turn angle between steps follows the distribution
      measured on photographs of real lightning (mean about 16°).</li>
      <li>${S.A.n_branches ? `${S.A.n_branches} side branches fork off at random, get shorter and weaker each generation, and never reach the ground.` : "This one has no side branches: a single tortuous channel."}</li>
      ${S.A.preset === "with_incloud" ? "<li>It also has a long, nearly horizontal section inside the cloud (violet) feeding the channel.</li>" : ""}</ul>
      <p class="note">Blue is the simulation's ground truth. The reconstruction (step 5) never sees it.
      The cloud is drawn at the channel's top so you can see all of it; real cloud bases are lower and hide
      most of the channel from eyes, but not from microphones.</p>`,
    enter: (fresh) => {  // a fresh strike shows the leader and the flash, then waits for Play
      player.mode = "strike";
      if (fresh) { playStrike(true); setPlaying(false); player.introStart = reduceMotion ? -1 : performance.now(); }
    },
  },
  {
    kicker: "2 · The thunder",
    title: () => "Every metre of channel makes a bang",
    html: () => `<p>The return stroke heats the channel to about 30,000 K in microseconds. Each piece expands as a
      shock wave and relaxes into a pressure pulse (an "N-wave").</p>
      <ul><li>Pulse length follows from the energy per metre (Few's thunder model), which puts thunder's power
      near 100 Hz: the deep rumble.</li>
      <li>The simulator emits a pulse from every 0.5 m of channel, plus small kinks inside each 10 m step:
      about ${Math.round(S.A.length_km * 2000).toLocaleString()} sources for this strike.</li></ul>
      <div class="row"><button class="gold" id="again">▶ Play the strike</button></div>
      <p class="note">Blue shells: sound spreading out (illustrative). <span class="c-mint">Green</span>: the parts of
      the channel whose sound is reaching the microphones at that instant, from exact ray tracing.</p>`,
    enter: (fresh) => { $("again").onclick = () => playStrike(true); if (fresh) playStrike(true); },
  },
  {
    kicker: "3 · The atmosphere",
    title: () => "Sound bends on its way",
    html: () => `<p>This strike's air: <b>${S.A.temperature_c.toFixed(0)} °C</b> at the ground, cooling 6.5 °C per km
      of height, and a <b>${S.A.wind_mps.toFixed(1)} m/s</b> wind from the ${COMPASS[dirIdx(S.A.wind_from_deg)]}.</p>
      <ul><li><b>Refraction:</b> sound is slower in cold air, so it bends upward, and the wind pushes it
      sideways. Every path is found by exact ray tracing.</li>
      <li><b>Absorption:</b> high frequencies fade with distance (ISO 9613-1).</li>
      <li><b>Ground echo:</b> each pulse arrives twice, directly and off the ground.</li>
      <li><b>Shadow:</b> ${pct(S.M.shadow_fraction)} of this channel is in the acoustic shadow: none of its sound reaches the microphones.</li></ul>
      <p>Sound from near parts arrives first and from far parts last, so a strike lasting microseconds is heard
      for <b>${(S.lastArrival - S.firstArrival).toFixed(0)} s</b>: the rumble.</p>`,
    enter: () => {},
  },
  {
    kicker: "4 · The recording",
    title: () => "What five microphones hear",
    html: () => `<p>Five microphones, a square 50 m across with one in the middle, record the thunder (bottom strip).
      The first clap comes from the nearest part of the channel; the rumble that follows is sound from farther away.</p>
      <div class="row"><button id="listen">🔊 Listen at real speed</button></div>
      <ul><li>The recordings include realistic microphone responses, background and wind noise, GPS clock
      errors of about a microsecond and centimetre-level position errors.</li>
      <li>The same sound reaches each microphone up to ~0.15 ms apart. Those tiny delays say where it came from.</li></ul>`,
    enter: () => { $("listen").onclick = () => { finalHold = false; player.mode = "strike"; soundOn = true; $("sound").setAttribute("aria-pressed", "true");
      setSpeed(1); seek(Math.max(0, S.firstArrival - 0.4)); setPlaying(true); }; },
  },
  {
    kicker: "5 · Rebuilding",
    title: () => "From sound back to lightning",
    html: () => `<p>Now forget the true channel. Using only the five recordings and the time of the flash:</p>
      <ul><li><b>Direction:</b> in each 0.1 s slice, the delays between every pair of microphones
      (cross-correlation) give the direction the sound came from.</li>
      <li><b>Distance:</b> time since the flash × the speed of sound, traced back through the atmosphere.</li>
      <li><b>Point:</b> direction + distance = a point in 3D. Method B can separate up to three sounds arriving at once.</li></ul>
      <p><b class="c-warm"><span id="live-count">0</span></b> of ${S.np} points placed. <span class="c-warm">Gold rays</span>: the
      directions measured at the array. <span class="c-mint">Green</span>: where the sound in the current slice really came from.</p>
      <p class="note">This uses the true atmosphere (the "oracle" setting); see step 6 for what happens when the wind is unknown.</p>`,
    enter: () => playRecon(),
  },
  {
    kicker: "6 · Result",
    title: () => "How close did sound get?",
    html: () => `<div class="stats">
        <b>${S.M.n_points}</b><span>points rebuilt from sound alone</span>
        <b>${fmt(S.M.median_error_m)} m</b><span>median distance to the true channel (90% within ${fmt(S.M.p90_error_m, 0)} m)</span>
        <b>${fmt(S.M.angular_error_deg, 2)}°</b><span>median direction error seen from the array</span>
        <b>${pct(S.M.coverage_main_50m)}</b><span>of the main channel recovered (within 50 m)</span>
        ${S.M.coverage_branch_50m === null ? "" : `<b>${pct(S.M.coverage_branch_50m)}</b><span>of the side branches recovered</span>`}
      </div>
      <div class="row">
        <button id="t-truth" aria-pressed="true">True channel</button>
        <button id="t-recon" aria-pressed="true">Rebuilt</button>
        <button id="t-sigma" aria-pressed="false">Uncertainty</button>
        <button id="t-side" aria-pressed="false">Side by side</button>
      </div>
      <div class="row"><button class="gold" id="gen2">⚡ Generate another</button><button id="storm2">All ${NS} strikes</button></div>
      <p class="note">What it misses: channel in the acoustic shadow, sound from several places arriving at once, and
      weak branches. Across all ${NS} strikes the median error is ${LIB.summary.median_of_medians_m.toFixed(1)} m. These runs know
      the atmosphere; with the wind unknown, errors grow to about 130 m, the project's main finding (see the report).</p>`,
    enter: () => {
      player.mode = "recon"; setPlaying(false); player.t = S.recT1; finalHold = true; truthLevel = 0.45;
      const tog = (id, fn) => { const b = $(id); b.onclick = () => { const on = b.getAttribute("aria-pressed") !== "true"; b.setAttribute("aria-pressed", String(on)); fn(on); }; };
      tog("t-truth", (on) => (S.line.visible = on));
      tog("t-recon", (on) => (S.points.visible = on));
      tog("t-sigma", (on) => (S.sigma.visible = on));
      tog("t-side", (on) => { S.recon.position.copy(on ? S.sideOffset : new THREE.Vector3()); truthLevel = on ? 0.9 : 0.45; });
      $("gen2").onclick = generate; $("storm2").onclick = () => enterStorm();
    },
  },
];
steps.forEach((_, i) => { const d = document.createElement("i"); d.title = `Step ${i + 1}`; d.onclick = () => !stormOn && show(i); dots.appendChild(d); });

function show(k, fresh = true) {
  if (stormOn) leaveStorm(false);
  cur = Math.max(0, Math.min(steps.length - 1, k));
  const s = steps[cur];
  setPlaying(false); finalHold = false;
  S.line.visible = true; S.points.visible = true; S.sigma.visible = false; S.recon.position.set(0, 0, 0);
  view = cur >= 4 ? { t0: S.recT0, t1: S.recT1 } : { t0: 0, t1: S.dur };
  if (cur < 4) { player.mode = "strike"; }
  stepEl.innerHTML = `<div class="act"><span>${s.kicker}</span><button id="collapse">${panelEl.classList.contains("collapsed") ? "Show" : "Hide"}</button></div><h2>${s.title()}</h2>${s.html()}`;
  $("collapse").onclick = (e) => { const c = panelEl.classList.toggle("collapsed"); e.target.textContent = c ? "Show" : "Hide"; };
  [...dots.children].forEach((d, i) => d.classList.toggle("on", i === cur));
  $("nav").classList.remove("hidden");
  $("prev").disabled = cur === 0;
  const next = $("next");
  next.textContent = cur === 3 ? "Rebuild from sound →" : cur === steps.length - 1 ? "Generate another" : "Next";
  next.className = cur === 3 || cur === steps.length - 1 ? "gold" : "";
  s.enter(fresh);
  hudStrike();
}
$("prev").onclick = () => show(cur - 1);
$("next").onclick = () => (cur === steps.length - 1 ? generate() : show(cur + 1));

// ---------------------------------------------------------------- loading strikes
let bag = [];
async function openStrike(id, step = 1, autoplay = true) {
  const D = await fetchStrike(id);
  if (S) disposeStrike(S);
  S = buildStrike(D);
  scene.add(S.group);
  audio.src = `viewer/bolts/${id}.mp3`;
  audioBroken = false;
  groundGlow.position.set(S.strike.x, 0.002, S.strike.z);
  skyMat.uniforms.flashDir.value.copy(S.top).sub(camera.position).normalize();
  buildClouds(S.focus.x, S.focus.z, 9, S.box.max.y + 0.15, narrow() ? 90 : 170);
  flyTo(S.home, S.focus, 1.8);
  history.replaceState(null, "", `#strike=${id}`);
  show(step, autoplay);
  // Warm the cache with the next strike so Generate feels instant.
  if (!bag.length) refillBag();
  fetchStrike(bag[bag.length - 1]);
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

// ---------------------------------------------------------------- storm mode UI
async function enterStorm() {
  stormOn = true;
  setPlaying(false);
  controls.autoRotate = !reduceMotion;
  $("nav").classList.add("hidden");
  stepEl.innerHTML = `<div class="act"><span>The whole library</span><button id="collapse">Hide</button></div>
    <h2>${NS} strikes, all rebuilt from sound</h2>
    <p>Each strike below was generated, propagated through its own atmosphere, recorded by the same five
    microphones and rebuilt from the recording. Every strike that was generated is shown, good or bad.</p>
    <div class="progress"><i id="prog"></i></div>
    <div id="storm-body" class="hidden"></div>`;
  $("collapse").onclick = (e) => { const c = panelEl.classList.toggle("collapsed"); e.target.textContent = c ? "Show" : "Hide"; };
  await buildStorm((p) => { const el = $("prog"); if (el) el.style.width = `${p * 100}%`; });
  if (!stormOn) return;
  S.group.visible = false; S.labels.forEach((l) => (l.visible = false));
  storm.group.visible = true;
  micSprite.material.color.copy(WHITE); micSprite.scale.setScalar(0.011); micRing.material.opacity = 0;
  setPhase(`${NS} simulated strikes · every channel rebuilt from its own thunder`);
  buildClouds(0, 0, 10, 6.9, narrow() ? 110 : 220);
  groundGlow.material.opacity = 0;
  const s = LIB.summary;
  $("storm-body").innerHTML = `<div class="stats">
      <b>${s.total_points.toLocaleString()}</b><span>points rebuilt from sound across ${s.n_strikes} strikes</span>
      <b>${s.median_of_medians_m.toFixed(1)} m</b><span>typical (median) error per strike</span>
      <b>${pct(s.median_coverage_main)}</b><span>of a typical main channel recovered</span>
    </div>
    <div style="font-size:12.5px;color:var(--muted)">Median error per strike (m)</div>
    <canvas id="hist"></canvas>
    <div class="row"><button id="s-truth" aria-pressed="true">True channels</button><button id="s-recon" aria-pressed="true">Rebuilt</button></div>
    <div class="list" id="slist">${LIB.strikes.map((x) => `<button data-id="${x.id}" class="${S && x.id === S.A.id ? "cur" : ""}">
      <span>#${x.id + 1} · ${x.distance_km.toFixed(1)} km ${SHORT[dirIdx(x.azimuth_deg)]} · ${x.preset === "with_incloud" ? "in-cloud" : x.preset}</span>
      <span>${fmt(x.median_error_m)} m</span></button>`).join("")}</div>
    <div class="row"><button class="gold" id="s-back">← Back to strike #${S.A.id + 1}</button></div>
    <p class="note">Click a strike to watch it and its reconstruction. All runs know the atmosphere (Method B, oracle).</p>`;
  $("storm-body").classList.remove("hidden");
  document.querySelector(".progress").classList.add("hidden");
  drawHist();
  document.querySelectorAll("#slist button").forEach((b) => (b.onclick = () => { leaveStorm(false); openStrike(parseInt(b.dataset.id, 10), 1); }));
  $("s-back").onclick = () => leaveStorm(true);
  const tog = (id, obj) => { const b = $(id); b.onclick = () => { const on = b.getAttribute("aria-pressed") !== "true"; b.setAttribute("aria-pressed", String(on)); obj.visible = on; }; };
  tog("s-truth", storm.lines); tog("s-recon", storm.points);
  hudStorm();
  flyTo(new THREE.Vector3(narrow() ? 19 : 13, narrow() ? 14 : 9.5, narrow() ? 21 : 15), new THREE.Vector3(0, 2.4, 0), 2.2);
  history.replaceState(null, "", "#storm");
}
function leaveStorm(restore) {
  stormOn = false;
  controls.autoRotate = false;
  storm.group.visible = false;
  if (S) {
    S.group.visible = true; S.labels.forEach((l) => (l.visible = true));
    buildClouds(S.focus.x, S.focus.z, 9, S.box.max.y + 0.15, narrow() ? 90 : 170);
    if (restore) { flyTo(S.home, S.focus); show(cur, false); history.replaceState(null, "", `#strike=${S.A.id}`); }
  }
}
$("storm").onclick = () => (stormOn ? leaveStorm(true) : enterStorm());
function drawHist() {
  const c = $("hist"); if (!c) return;
  const dpr = Math.min(devicePixelRatio, 2), w = c.clientWidth, h = c.clientHeight;
  c.width = w * dpr; c.height = h * dpr;
  const g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const bins = new Array(10).fill(0), step = 1;
  LIB.strikes.forEach((s) => { if (s.median_error_m !== null) bins[Math.min(9, Math.floor(s.median_error_m / step))]++; });
  const top = Math.max(...bins), bw = w / bins.length;
  bins.forEach((n, i) => {
    const bh = (n / top) * (h - 16);
    g.fillStyle = errColor(i * step + step / 2).getStyle();
    g.fillRect(i * bw + 2, h - 14 - bh, bw - 4, bh);
    g.fillStyle = "#8a96b2"; g.font = "10px Inter, sans-serif"; g.textAlign = "center";
    g.fillText(i === 9 ? "9+" : `${i * step}–${(i + 1) * step}`, i * bw + bw / 2, h - 2);
    if (n) { g.fillStyle = "#f1f4fb"; g.fillText(String(n), i * bw + bw / 2, h - 17 - bh); }
  });
}

// ---------------------------------------------------------------- controls wiring
document.querySelectorAll("[data-speed]").forEach((b) => (b.onclick = () => setSpeed(parseFloat(b.dataset.speed))));
$("play").onclick = () => {
  if (!S || stormOn) return;
  if (player.playing) return setPlaying(false);
  const end = player.mode === "recon" ? S.recT1 : Math.min(S.lastArrival + 1, S.dur);
  if (player.mode === "recon" && finalHold) { finalHold = false; }
  if (player.t >= end - 0.05) return player.mode === "recon" ? playRecon() : playStrike(true);
  setPlaying(true);
};
$("sound").onclick = () => {
  soundOn = !soundOn; audioBroken = false;
  $("sound").setAttribute("aria-pressed", String(soundOn));
  syncAudio();
};
addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  if (e.code === "Space") { e.preventDefault(); $("play").click(); }
  else if (e.key === "g" || e.key === "G") generate();
  else if (e.key === "ArrowRight" && !stormOn) show(cur + 1);
  else if (e.key === "ArrowLeft" && !stormOn) show(cur - 1);
});

// ---------------------------------------------------------------- main loop
let last = performance.now();
function tick() {
  const now = performance.now();  // one clock for everything (rAF timestamps can use another timebase)
  const dt = Math.min((now - last) / 1000, 0.1); last = now;
  if (tween) {
    const k = Math.min((now - tween.start) / tween.dur, 1), e = k < 0.5 ? 4 * k * k * k : 1 - Math.pow(-2 * k + 2, 3) / 2;
    camera.position.lerpVectors(tween.p0, tween.p1, e); controls.target.lerpVectors(tween.q0, tween.q1, e);
    if (k >= 1) tween = null;
  }
  let flash = 0, flashAt = null;
  if (stormOn && storm.built) {
    const r = stormFrame(now); flash = r.f;
    if (r.at) flashAt = toV([r.at.distance_km * 1000 * Math.sin((r.at.azimuth_deg * Math.PI) / 180), r.at.distance_km * 1000 * Math.cos((r.at.azimuth_deg * Math.PI) / 180), 6000]);
  } else if (S && !stormOn) {
    flash = strikeFrame(now, dt); flashAt = S.top;
  }
  // The flash lights the sky, the clouds, the ground and the screen.
  skyMat.uniforms.flash.value = flash;
  if (flashAt) { skyMat.uniforms.flashDir.value.copy(flashAt).sub(camera.position).normalize(); lightClouds(flash, flashAt); }
  else lightClouds(0, camera.position);
  groundMat.color.setRGB(0.03 + 0.12 * flash, 0.04 + 0.13 * flash, 0.07 + 0.2 * flash, THREE.SRGBColorSpace);
  if (S && !stormOn) { groundGlow.material.opacity = Math.min(flash, 1); groundGlow.scale.setScalar(2.5 + 2 * flash); }
  $("flash").style.opacity = reduceMotion ? 0 : String(Math.min(flash * 0.28, 0.35));
  bloom.strength = (stormOn ? 0.45 : 0.85) + 1.1 * flash;
  sky.position.copy(camera.position);
  rainFrame(dt);
  drawWave();
  controls.update();
  composer.render();
  labels.render(scene, camera);
  requestAnimationFrame(tick);
}

// ---------------------------------------------------------------- start
const q = new URLSearchParams(location.hash.slice(1));
const wantStorm = location.hash.includes("storm");
setPlaying(false);
setSpeed(2);
const first = q.has("strike") ? Math.min(NS - 1, Math.max(0, parseInt(q.get("strike"), 10) || 0)) : 0;
await openStrike(first, Math.max(0, (parseInt(q.get("step") ?? "1", 10) || 1) - 1), !q.has("t"));
camera.position.copy(S.home); controls.target.copy(S.focus); tween = null;
if (q.has("t")) { setPlaying(false); seek(parseFloat(q.get("t"))); if (cur === 5 && q.get("side") === "1") $("t-side").click(); }
if (wantStorm) enterStorm();
$("loading").style.opacity = "0";
setTimeout(() => $("loading").remove(), 600);
requestAnimationFrame(tick);
window.__viewer = { ready: true, strikes: NS, get t() { return player.t; }, get step() { return cur; } };
