import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import {
  buildForegroundMask,
  computeGarmentPose,
  drawDebug,
  drawWarpedGarment,
} from './warpUtils';

const Parallel3DMode = ({ onBack }) => {
  const mountRef = useRef(null);
  const compositorCanvasRef = useRef(null);
  const latestPoseRef = useRef(null);
  const renderModeRef = useRef('compositor');
  const garmentPoseRef = useRef(null);
  const prevAnchorRef = useRef(null);
  const scaleWindowRef = useRef([]);
  const showKeypointsRef = useRef(false);
  const showWarpGridRef = useRef(false);
  const showMaskRef = useRef(false);
  const failSinceRef = useRef(null);
  const passSinceRef = useRef(null);
  const lastAutoFallbackMsRef = useRef(0);

  const [status, setStatus] = useState('Initializing WebGL scene...');
  const [renderMode, setRenderMode] = useState('compositor');
  const [renderFps, setRenderFps] = useState(0);
  const [poseLatencyMs, setPoseLatencyMs] = useState(null);
  const [maskLatencyMs, setMaskLatencyMs] = useState(null);
  const [anchorJitter, setAnchorJitter] = useState(0);
  const [scaleVariance, setScaleVariance] = useState(0);
  const [poseConfidence, setPoseConfidence] = useState(0);
  const [segmentationMode, setSegmentationMode] = useState('placeholder');
  const [wsConnected, setWsConnected] = useState(false);
  const [showKeypoints, setShowKeypoints] = useState(false);
  const [showWarpGrid, setShowWarpGrid] = useState(false);
  const [showMask, setShowMask] = useState(false);
  const [demoReady, setDemoReady] = useState(false);

  useEffect(() => {
    renderModeRef.current = renderMode;
  }, [renderMode]);

  useEffect(() => {
    showKeypointsRef.current = showKeypoints;
  }, [showKeypoints]);

  useEffect(() => {
    showWarpGridRef.current = showWarpGrid;
  }, [showWarpGrid]);

  useEffect(() => {
    showMaskRef.current = showMask;
  }, [showMask]);

  useEffect(() => {
    let renderer;
    let scene;
    let camera;
    let cube;
    let webcamPlane;
    let videoTexture;
    let rafId;
    let stream;
    let ws;
    let overlayCtx;
    let fgCanvas;
    let fgCtx;

    let garmentImage = null;
    let garmentReady = false;

    const defaultGarmentCanvas = document.createElement('canvas');
    defaultGarmentCanvas.width = 420;
    defaultGarmentCanvas.height = 560;
    const defaultCtx = defaultGarmentCanvas.getContext('2d');
    if (defaultCtx) {
      const grd = defaultCtx.createLinearGradient(0, 0, 0, defaultGarmentCanvas.height);
      grd.addColorStop(0, '#84b6ff');
      grd.addColorStop(1, '#2a5fcb');
      defaultCtx.fillStyle = grd;
      defaultCtx.beginPath();
      defaultCtx.moveTo(80, 40);
      defaultCtx.lineTo(340, 40);
      defaultCtx.lineTo(390, 160);
      defaultCtx.lineTo(320, 540);
      defaultCtx.lineTo(100, 540);
      defaultCtx.lineTo(30, 160);
      defaultCtx.closePath();
      defaultCtx.fill();
      defaultCtx.fillStyle = 'rgba(255,255,255,0.2)';
      defaultCtx.fillRect(145, 72, 130, 26);
    }
    garmentImage = defaultGarmentCanvas;
    garmentReady = true;

    const tryLoadGarment = (garmentName) => {
      if (!garmentName) return;
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        garmentImage = img;
        garmentReady = true;
      };
      img.onerror = () => {
        garmentImage = defaultGarmentCanvas;
        garmentReady = true;
      };
      img.src = `/assets/garments/${encodeURIComponent(garmentName)}`;
    };

    const video = document.createElement('video');
    video.autoplay = true;
    video.muted = true;
    video.playsInline = true;

    let lastFrameTime = performance.now();
    let fpsFrames = 0;
    let fpsWindowStart = performance.now();

    const mount = mountRef.current;
    const overlayCanvas = compositorCanvasRef.current;
    if (!mount) {
      return undefined;
    }

    const init = async () => {
      try {
        scene = new THREE.Scene();
        scene.background = new THREE.Color(0x05070f);

        camera = new THREE.PerspectiveCamera(60, mount.clientWidth / mount.clientHeight, 0.1, 1000);
        camera.position.set(0, 0.5, 3.2);

        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: 'high-performance' });
        renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        renderer.setSize(mount.clientWidth, mount.clientHeight);
        mount.appendChild(renderer.domElement);

        const hemi = new THREE.HemisphereLight(0xffffff, 0x202040, 1.0);
        hemi.position.set(0, 2, 0);
        scene.add(hemi);

        const dir = new THREE.DirectionalLight(0xffffff, 0.9);
        dir.position.set(2, 3, 2);
        scene.add(dir);

        const cubeGeo = new THREE.BoxGeometry(0.45, 0.45, 0.45);
        const cubeMat = new THREE.MeshStandardMaterial({ color: 0x4aa3ff, roughness: 0.3, metalness: 0.1 });
        cube = new THREE.Mesh(cubeGeo, cubeMat);
        cube.position.set(-0.95, 0.15, 0.0);
        scene.add(cube);

        if (overlayCanvas) {
          overlayCtx = overlayCanvas.getContext('2d', { alpha: false, desynchronized: true });
          fgCanvas = document.createElement('canvas');
          fgCtx = fgCanvas.getContext('2d');
        }

        stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: { ideal: 1280 },
            height: { ideal: 720 },
            frameRate: { ideal: 30, max: 60 },
            facingMode: 'user',
          },
          audio: false,
        });

        video.srcObject = stream;
        await video.play();

        videoTexture = new THREE.VideoTexture(video);
        videoTexture.minFilter = THREE.LinearFilter;
        videoTexture.magFilter = THREE.LinearFilter;
        videoTexture.format = THREE.RGBFormat;

        const planeGeo = new THREE.PlaneGeometry(2.0, 1.2);
        const planeMat = new THREE.MeshBasicMaterial({ map: videoTexture });
        webcamPlane = new THREE.Mesh(planeGeo, planeMat);
        webcamPlane.position.set(0.5, 0.0, 0.0);
        scene.add(webcamPlane);

        setStatus('WebGL live: cube + webcam texture active');

        const wsProto = window.location.protocol === 'https:' ? 'wss' : 'ws';
        const wsUrl = `${wsProto}://${window.location.hostname}:8765`;
        ws = new WebSocket(wsUrl);
        ws.onopen = () => {
          setWsConnected(true);
          setStatus('Realtime pose stream connected.');
        };
        ws.onclose = () => {
          setWsConnected(false);
          setStatus('Camera live, waiting for pose stream...');
        };
        ws.onerror = () => {
          setWsConnected(false);
          setStatus('Camera live, pose stream connection error');
        };
        ws.onmessage = (evt) => {
          try {
            const packet = JSON.parse(evt.data);
            latestPoseRef.current = packet;
            if (typeof packet?.segmentation?.latencyMs === 'number') {
              setMaskLatencyMs(packet.segmentation.latencyMs);
            }
            if (typeof packet?.segmentation?.mode === 'string') {
              setSegmentationMode(packet.segmentation.mode);
            }
            if (typeof packet?.poseConfidence === 'number') {
              setPoseConfidence(packet.poseConfidence);
            }
            if (packet && typeof packet.timestamp === 'number') {
              setPoseLatencyMs(Math.max(0, Date.now() - packet.timestamp));
            }
            if (packet?.garment) {
              tryLoadGarment(packet.garment);
            }
          } catch (_err) {
            // Ignore malformed packets.
          }
        };

        const computeScaleVariance = () => {
          const values = scaleWindowRef.current;
          if (values.length < 5) return 0;
          const mean = values.reduce((a, b) => a + b, 0) / values.length;
          const variance = values.reduce((acc, n) => acc + (n - mean) ** 2, 0) / values.length;
          return Math.sqrt(variance);
        };

        const resizeOverlay = () => {
          if (!overlayCanvas || !mount) return;
          const w = Math.max(1, mount.clientWidth);
          const h = Math.max(1, mount.clientHeight);
          const dpr = Math.min(window.devicePixelRatio || 1, 2);
          overlayCanvas.width = Math.floor(w * dpr);
          overlayCanvas.height = Math.floor(h * dpr);
          overlayCanvas.style.width = `${w}px`;
          overlayCanvas.style.height = `${h}px`;
          if (overlayCtx) {
            overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
          }
          if (fgCanvas) {
            fgCanvas.width = overlayCanvas.width;
            fgCanvas.height = overlayCanvas.height;
          }
        };

        resizeOverlay();

        const animate = (now) => {
          const dt = now - lastFrameTime;
          lastFrameTime = now;

          fpsFrames += 1;
          if (now - fpsWindowStart >= 500) {
            const fps = (fpsFrames * 1000) / (now - fpsWindowStart);
            setRenderFps(Math.round(fps));
            fpsFrames = 0;
            fpsWindowStart = now;
          }

          const packet = latestPoseRef.current;

          if (packet && packet.anchor && typeof packet.anchor.x === 'number' && typeof packet.anchor.y === 'number') {
            const prev = prevAnchorRef.current;
            if (prev) {
              const jitter = Math.hypot(packet.anchor.x - prev.x, packet.anchor.y - prev.y) * 1000;
              setAnchorJitter(jitter);
            }
            prevAnchorRef.current = { x: packet.anchor.x, y: packet.anchor.y };
          }

          if (packet && typeof packet.scale === 'number') {
            const values = scaleWindowRef.current;
            values.push(packet.scale);
            if (values.length > 45) values.shift();
            setScaleVariance(computeScaleVariance());
          }

          if (renderModeRef.current === 'sanity' && cube) {
            cube.rotation.y += dt * 0.0012;
            cube.rotation.x += dt * 0.0007;

            if (packet && Array.isArray(packet.pose2D)) {
              const nose = packet.pose2D.find((pt) => Array.isArray(pt) && pt[0] === 0);
              if (nose && typeof nose[1] === 'number' && typeof nose[2] === 'number') {
                cube.position.x = -0.95 + (nose[1] - 0.5) * 0.8;
                cube.position.y = 0.15 + (0.5 - nose[2]) * 0.8;
              }
            }
            if (renderer?.domElement) {
              renderer.domElement.style.display = 'block';
            }
            if (overlayCanvas) {
              overlayCanvas.style.display = 'none';
            }
            renderer.render(scene, camera);
          } else if (overlayCtx && overlayCanvas) {
            if (renderer?.domElement) {
              renderer.domElement.style.display = 'none';
            }
            overlayCanvas.style.display = 'block';

            const drawW = overlayCanvas.clientWidth;
            const drawH = overlayCanvas.clientHeight;

            if (video.readyState >= 2) {
              overlayCtx.drawImage(video, 0, 0, drawW, drawH);
            } else {
              overlayCtx.fillStyle = '#05070f';
              overlayCtx.fillRect(0, 0, drawW, drawH);
            }

            if (packet && garmentReady) {
              garmentPoseRef.current = computeGarmentPose(packet, garmentPoseRef.current);
              drawWarpedGarment(overlayCtx, garmentImage, garmentPoseRef.current, drawW, drawH);

              if (fgCtx && fgCanvas && video.readyState >= 2) {
                const dpr = overlayCanvas.width / Math.max(1, drawW);
                fgCtx.setTransform(1, 0, 0, 1, 0, 0);
                fgCtx.clearRect(0, 0, fgCanvas.width, fgCanvas.height);
                fgCtx.drawImage(video, 0, 0, fgCanvas.width, fgCanvas.height);
                fgCtx.fillStyle = '#fff';
                fgCtx.globalCompositeOperation = 'destination-in';
                fgCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
                buildForegroundMask(fgCtx, packet, drawW, drawH);
                fgCtx.setTransform(1, 0, 0, 1, 0, 0);
                fgCtx.globalCompositeOperation = 'source-over';
                overlayCtx.drawImage(fgCanvas, 0, 0, drawW, drawH);
              }

              drawDebug(overlayCtx, packet, drawW, drawH, {
                showKeypoints: showKeypointsRef.current,
                showWarpGrid: showWarpGridRef.current,
                showMask: showMaskRef.current,
              });
            }
          }
          rafId = requestAnimationFrame(animate);
        };

        rafId = requestAnimationFrame(animate);
      } catch (err) {
        setStatus(`Initialization failed: ${err?.message || 'unknown error'}`);
      }
    };

    const onResize = () => {
      if (!renderer || !camera || !mount) return;
      camera.aspect = mount.clientWidth / mount.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(mount.clientWidth, mount.clientHeight);
      if (overlayCanvas && overlayCtx) {
        const w = Math.max(1, mount.clientWidth);
        const h = Math.max(1, mount.clientHeight);
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        overlayCanvas.width = Math.floor(w * dpr);
        overlayCanvas.height = Math.floor(h * dpr);
        overlayCanvas.style.width = `${w}px`;
        overlayCanvas.style.height = `${h}px`;
        overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      }
    };

    window.addEventListener('resize', onResize);
    init();

    return () => {
      window.removeEventListener('resize', onResize);

      if (rafId) cancelAnimationFrame(rafId);

      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }

      if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        ws.close();
      }

      if (renderer) {
        renderer.dispose();
        if (renderer.domElement && renderer.domElement.parentNode === mount) {
          mount.removeChild(renderer.domElement);
        }
      }

      if (videoTexture) videoTexture.dispose();
      if (cube) {
        cube.geometry.dispose();
        cube.material.dispose();
      }
      if (webcamPlane) {
        webcamPlane.geometry.dispose();
        webcamPlane.material.dispose();
      }
    };
  }, []);

  const gateLevel = (value, warnMax, failMax, lowerIsBetter = true) => {
    if (value == null || Number.isNaN(value)) return 'unknown';
    if (lowerIsBetter) {
      if (value <= warnMax) return 'pass';
      if (value <= failMax) return 'warn';
      return 'fail';
    }
    if (value >= warnMax) return 'pass';
    if (value >= failMax) return 'warn';
    return 'fail';
  };

  const latencyGate = gateLevel(poseLatencyMs, 90, 160, true);
  const jitterGate = gateLevel(anchorJitter, 14, 26, true);
  const scaleGate = gateLevel(scaleVariance, 0.03, 0.065, true);
  const confidenceGate = gateLevel(poseConfidence, 0.7, 0.5, false);
  const gateStates = [latencyGate, jitterGate, scaleGate, confidenceGate];
  const overallGate = gateStates.includes('fail') ? 'fail' : gateStates.includes('warn') ? 'warn' : gateStates.includes('pass') ? 'pass' : 'unknown';

  useEffect(() => {
    const interval = setInterval(() => {
      if (!wsConnected || renderMode !== 'compositor') {
        failSinceRef.current = null;
        passSinceRef.current = null;
        setDemoReady(false);
        return;
      }

      const now = Date.now();

      if (overallGate === 'pass') {
        failSinceRef.current = null;
        if (passSinceRef.current == null) {
          passSinceRef.current = now;
        }
        const stableForMs = now - passSinceRef.current;
        setDemoReady(stableForMs >= 2200);
        return;
      }

      passSinceRef.current = null;
      setDemoReady(false);

      if (overallGate === 'fail') {
        if (failSinceRef.current == null) {
          failSinceRef.current = now;
          return;
        }

        const failForMs = now - failSinceRef.current;
        const cooldownOk = now - lastAutoFallbackMsRef.current >= 8000;
        if (failForMs >= 2600 && cooldownOk) {
          lastAutoFallbackMsRef.current = now;
          setRenderMode('sanity');
          setStatus('Auto-guard triggered: sustained FAIL, switched to sanity mode.');
          failSinceRef.current = null;
        }
      } else {
        failSinceRef.current = null;
      }
    }, 200);

    return () => clearInterval(interval);
  }, [overallGate, renderMode, wsConnected]);

  return (
    <div className="parallel3d-shell">
      <div className="parallel3d-toolbar">
        <button type="button" onClick={onBack}>Back</button>
        <button type="button" onClick={() => setRenderMode((m) => (m === 'compositor' ? 'sanity' : 'compositor'))}>
          Mode: {renderMode === 'compositor' ? '2.5D Compositor' : 'Sanity 3D'}
        </button>
        <button type="button" onClick={() => setShowKeypoints((v) => !v)}>Keypoints</button>
        <button type="button" onClick={() => setShowWarpGrid((v) => !v)}>Warp Grid</button>
        <button type="button" onClick={() => setShowMask((v) => !v)}>Mask</button>
        <div className="parallel3d-pill">Render FPS: {renderFps}</div>
        <div className="parallel3d-pill">Pose WS: {wsConnected ? 'Connected' : 'Disconnected'}</div>
        <div className="parallel3d-pill">Pose Latency: {poseLatencyMs == null ? '--' : `${poseLatencyMs} ms`}</div>
        <div className="parallel3d-pill">Mask Latency: {maskLatencyMs == null ? '--' : `${maskLatencyMs.toFixed(1)} ms`}</div>
        <div className="parallel3d-pill">Mask Mode: {segmentationMode}</div>
        <div className="parallel3d-pill">Anchor Jitter: {anchorJitter.toFixed(2)}</div>
        <div className="parallel3d-pill">Scale Var: {scaleVariance.toFixed(4)}</div>
        <div className={`parallel3d-gate ${overallGate}`}>Overall: {overallGate.toUpperCase()}</div>
        <div className={`parallel3d-gate ${latencyGate}`}>Latency</div>
        <div className={`parallel3d-gate ${jitterGate}`}>Jitter</div>
        <div className={`parallel3d-gate ${scaleGate}`}>Scale</div>
        <div className={`parallel3d-gate ${confidenceGate}`}>Confidence</div>
        <div className={`parallel3d-gate ${demoReady ? 'pass' : 'warn'}`}>Demo Ready: {demoReady ? 'YES' : 'NO'}</div>
        <div className="parallel3d-status">{status}</div>
      </div>
      <div className="parallel3d-canvas parallel3d-stage" ref={mountRef}>
        <canvas ref={compositorCanvasRef} className="parallel3d-overlay-canvas" />
      </div>
    </div>
  );
};

export default Parallel3DMode;
