import * as THREE from 'three';

const JOINT_IDS = [
  'nose',
  'left_shoulder',
  'right_shoulder',
  'left_elbow',
  'right_elbow',
  'left_wrist',
  'right_wrist',
  'left_hip',
  'right_hip',
];

export function createSkeletonRenderer(scene) {
  const joints = new Map();
  const shoulderLights = [];
  const geo = new THREE.SphereGeometry(0.04, 16, 16);

  JOINT_IDS.forEach((id) => {
    const mat = new THREE.MeshBasicMaterial({ color: 0x06b6d4 });
    const mesh = new THREE.Mesh(geo, mat);
    scene.add(mesh);
    joints.set(id, mesh);

    if (id === 'left_shoulder' || id === 'right_shoulder') {
      const light = new THREE.PointLight(0x06b6d4, 0.8, 2);
      mesh.add(light);
      shoulderLights.push(light);
    }
  });

  const update = (pose, t) => {
    pose.forEach((kp) => {
      const mesh = joints.get(kp.id);
      if (!mesh) {
        return;
      }
      mesh.position.x = (kp.x / 640 - 0.5) * 6.4;
      mesh.position.y = -(kp.y / 480 - 0.5) * 3.8;
      mesh.position.z = 0.35;
      const pulse = 1 + Math.sin(t * 4 + kp.x * 0.003) * 0.2;
      mesh.scale.setScalar(pulse);
    });
  };

  const dispose = () => {
    joints.forEach((mesh) => {
      scene.remove(mesh);
      mesh.geometry.dispose();
      mesh.material.dispose();
    });
    shoulderLights.length = 0;
  };

  return { update, dispose };
}
