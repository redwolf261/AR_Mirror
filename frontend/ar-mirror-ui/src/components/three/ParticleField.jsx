import * as THREE from 'three';

export function createParticleField(scene, count = 200) {
  const positions = new Float32Array(count * 3);

  for (let i = 0; i < count; i += 1) {
    const idx = i * 3;
    positions[idx] = (Math.random() - 0.5) * 12;
    positions[idx + 1] = (Math.random() - 0.5) * 7;
    positions[idx + 2] = -2 + Math.random() * 4;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

  const material = new THREE.PointsMaterial({
    color: 0x7c3aed,
    size: 0.015,
    transparent: true,
    opacity: 0.4,
    depthWrite: false,
  });

  const points = new THREE.Points(geometry, material);
  scene.add(points);

  const update = (speed = 0.0018) => {
    const attr = geometry.attributes.position;
    for (let i = 0; i < count; i += 1) {
      const idx = i * 3;
      attr.array[idx + 1] += speed + ((i % 5) * 0.00012);
      if (attr.array[idx + 1] > 4.0) {
        attr.array[idx + 1] = -4.0;
      }
    }
    attr.needsUpdate = true;
  };

  const dispose = () => {
    scene.remove(points);
    geometry.dispose();
    material.dispose();
  };

  return { points, update, dispose };
}
