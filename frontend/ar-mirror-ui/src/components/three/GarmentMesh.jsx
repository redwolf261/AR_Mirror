import * as THREE from 'three';

function createFallbackTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d');

  const g = ctx.createLinearGradient(0, 0, 512, 512);
  g.addColorStop(0, 'rgba(124,58,237,0.9)');
  g.addColorStop(1, 'rgba(6,182,212,0.85)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 512, 512);
  ctx.fillStyle = 'rgba(255,255,255,0.14)';
  for (let i = 0; i < 8; i += 1) {
    ctx.fillRect(24 + i * 56, 0, 20, 512);
  }
  return new THREE.CanvasTexture(canvas);
}

export function createGarmentMesh(scene, textureUrl) {
  const loader = new THREE.TextureLoader();
  const garmentGeo = new THREE.PlaneGeometry(3.2, 4.0);
  const fallbackTexture = createFallbackTexture();

  const garmentMat = new THREE.MeshBasicMaterial({
    map: fallbackTexture,
    transparent: true,
    alphaTest: 0.1,
    side: THREE.DoubleSide,
  });

  const garmentMesh = new THREE.Mesh(garmentGeo, garmentMat);
  garmentMesh.position.set(0, -0.1, 0.2);
  scene.add(garmentMesh);

  const targetPosition = new THREE.Vector3(0, -0.1, 0.2);
  const targetScale = new THREE.Vector3(1, 1, 1);

  const loadTexture = (url) => {
    if (!url) {
      return;
    }
    loader.load(
      url,
      (tex) => {
        tex.encoding = THREE.sRGBEncoding;
        garmentMat.map = tex;
        garmentMat.needsUpdate = true;
      },
      undefined,
      () => {
        garmentMat.map = fallbackTexture;
        garmentMat.needsUpdate = true;
      }
    );
  };

  loadTexture(textureUrl);

  const updateFromPose = (pose) => {
    const lShoulder = pose.find((k) => k.id === 'left_shoulder');
    const rShoulder = pose.find((k) => k.id === 'right_shoulder');
    const lHip = pose.find((k) => k.id === 'left_hip');
    const rHip = pose.find((k) => k.id === 'right_hip');

    if (!lShoulder || !rShoulder || !lHip || !rHip) {
      return;
    }

    const centerX = (lShoulder.x + rShoulder.x + lHip.x + rHip.x) / 4;
    const centerY = (lShoulder.y + rShoulder.y + lHip.y + rHip.y) / 4;
    const shoulderWidth = Math.abs(rShoulder.x - lShoulder.x);
    const torsoHeight = Math.abs(((lHip.y + rHip.y) / 2) - ((lShoulder.y + rShoulder.y) / 2));

    targetPosition.set((centerX / 640 - 0.5) * 6.4, -(centerY / 480 - 0.5) * 3.8, 0.2);
    targetScale.set(Math.max(0.75, shoulderWidth / 170), Math.max(0.75, torsoHeight / 185), 1);
  };

  const tick = () => {
    garmentMesh.position.lerp(targetPosition, 0.15);
    garmentMesh.scale.lerp(targetScale, 0.12);
  };

  const dispose = () => {
    scene.remove(garmentMesh);
    garmentGeo.dispose();
    garmentMat.dispose();
    fallbackTexture.dispose();
  };

  return {
    garmentMesh,
    updateFromPose,
    tick,
    loadTexture,
    dispose,
  };
}
