import { useMemo, useRef } from 'react';
import * as THREE from 'three';
import { useThreeScene } from '../../hooks/useThreeScene';
import { createGarmentMesh } from './GarmentMesh';
import { createParticleField } from './ParticleField';
import { createSkeletonRenderer } from './SkeletonRenderer';
import styles from './ARScene.module.css';

function makeFallbackCameraTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 1280;
  canvas.height = 720;
  const ctx = canvas.getContext('2d');

  return {
    canvas,
    texture: new THREE.CanvasTexture(canvas),
    draw: (time) => {
      const g = ctx.createLinearGradient(0, 0, canvas.width, canvas.height);
      g.addColorStop(0, '#0b1021');
      g.addColorStop(1, '#111f38');
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = 'rgba(6,182,212,0.12)';
      for (let i = 0; i < 8; i += 1) {
        const x = (i * 180 + Math.sin(time * 0.001 + i) * 40) % canvas.width;
        ctx.fillRect(x, 0, 18, canvas.height);
      }
      ctx.fillStyle = 'rgba(124,58,237,0.3)';
      ctx.font = '700 42px Space Grotesk';
      ctx.fillText('MOCK CAMERA STREAM', 70, 82);
    },
  };
}

export default function ARScene({ posePoints, garment }) {
  const mountRef = useRef(null);
  const poseRef = useRef(posePoints);
  poseRef.current = posePoints;

  const sceneDeps = useMemo(() => [garment.texture], [garment.texture]);

  useThreeScene(() => {
    const mount = mountRef.current;
    if (!mount) {
      return undefined;
    }

    const w = mount.clientWidth;
    const h = mount.clientHeight;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, w / h, 0.1, 1000);
    camera.position.z = 4.3;

    const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(w, h);
    renderer.setClearColor(0x000000, 0);
    mount.appendChild(renderer.domElement);

    const fallback = makeFallbackCameraTexture();
    const bgGeo = new THREE.PlaneGeometry(16, 9);
    const bgMat = new THREE.MeshBasicMaterial({ map: fallback.texture });
    const bgMesh = new THREE.Mesh(bgGeo, bgMat);
    bgMesh.position.z = -1;
    scene.add(bgMesh);

    const garmentRig = createGarmentMesh(scene, garment.texture);
    const skeleton = createSkeletonRenderer(scene);
    const particles = createParticleField(scene, 200);

    const ambient = new THREE.AmbientLight(0xffffff, 0.62);
    scene.add(ambient);

    let raf;
    const animate = (t) => {
      raf = requestAnimationFrame(animate);

      fallback.draw(t);
      fallback.texture.needsUpdate = true;

      garmentRig.updateFromPose(poseRef.current);
      garmentRig.tick();
      skeleton.update(poseRef.current, t * 0.001);
      particles.update();

      renderer.render(scene, camera);
    };

    animate(0);

    const onResize = () => {
      const nw = mount.clientWidth;
      const nh = mount.clientHeight;
      camera.aspect = nw / nh;
      camera.updateProjectionMatrix();
      renderer.setSize(nw, nh);
    };

    window.addEventListener('resize', onResize);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', onResize);
      garmentRig.dispose();
      skeleton.dispose();
      particles.dispose();
      scene.remove(bgMesh);
      bgGeo.dispose();
      bgMat.dispose();
      fallback.texture.dispose();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
  }, sceneDeps);

  return <div ref={mountRef} className={styles.scene} />;
}
