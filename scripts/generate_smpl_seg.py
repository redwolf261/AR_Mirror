"""
Generate smpl_vert_segmentation.json from the SMPL neutral pkl.
SMPL has a weights matrix (J_regressor or weights) of shape (6890, 24)
where column j gives the skinning weight of vertex i for joint j.
We assign each vertex to the joint it has highest weight for,
then map joint indices to standard part names.
"""
import pickle, numpy as np, json, pathlib

JOINT_NAMES = [
    'hips', 'leftUpLeg', 'rightUpLeg', 'spine',
    'leftLeg', 'rightLeg', 'spine1', 'leftFoot',
    'rightFoot', 'spine2', 'leftToeBase', 'rightToeBase',
    'neck', 'leftShoulder', 'rightShoulder', 'head',
    'leftArm', 'rightArm', 'leftForeArm', 'rightForeArm',
    'leftHand', 'rightHand', 'leftHandIndex1', 'rightHandIndex1',
]

with open('models/smpl_neutral.pkl', 'rb') as f:
    smpl = pickle.load(f, encoding='latin1')

print("SMPL keys:", [k for k in smpl.keys()])

# LBS weights: (6890, 24) — skinning weight per vertex per joint
W = smpl['weights']
if hasattr(W, 'toarray'):
    W = W.toarray()
W = np.array(W)
print(f"Weights shape: {W.shape}")

# Assign each vertex to the joint with highest skinning weight
assignments = np.argmax(W, axis=1)  # (6890,)

seg = {name: [] for name in JOINT_NAMES}
for vert_idx, joint_idx in enumerate(assignments):
    if joint_idx < len(JOINT_NAMES):
        seg[JOINT_NAMES[joint_idx]].append(int(vert_idx))

# Report
for name, verts in seg.items():
    print(f"  {name}: {len(verts)} vertices")

out = pathlib.Path('models/smpl_vert_segmentation.json')
with open(out, 'w') as f:
    json.dump(seg, f)

print(f"\nSaved to {out}  ({out.stat().st_size} bytes)")
