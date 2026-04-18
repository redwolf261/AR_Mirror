export const MOCK_MEASUREMENTS = {
  size: 'L',
  confidence: 33,
  confidenceLabel: 'Low',
  metrics: {
    chest: { value: 74.8, unit: 'cm' },
    waist: { value: 60.0, unit: 'cm' },
    shoulder: { value: 65.1, unit: 'cm' },
    collar: { value: 12, unit: 'in' },
    trouserWaist: { value: 24, unit: 'in' },
    torso: { value: 90.0, unit: 'cm' },
  },
  scores: { base: 30, stability: 20, coverage: 70 },
  tips: [
    'Stability across captures: 20/100',
    'Measurement coverage: 70/100 (4/4 metrics)',
    'Side-view distinctness: 100/100',
    'Recapture in better framing to improve precision',
  ],
};

export const MOCK_GARMENTS = [
  { id: 1, name: 'Blue Check Shirt', size: 'L', thumb: '/garments/shirt-blue.svg', texture: '/garments/shirt-blue.svg' },
  { id: 2, name: 'White Polo', size: 'L', thumb: '/garments/polo-white.svg', texture: '/garments/polo-white.svg' },
  { id: 3, name: 'Striped Tee', size: 'M', thumb: '/garments/tee-stripe.svg', texture: '/garments/tee-stripe.svg' },
  { id: 4, name: 'Navy Blazer', size: 'L', thumb: '/garments/blazer-navy.svg', texture: '/garments/blazer-navy.svg' },
];

export const INITIAL_POSE = [
  { id: 'nose', x: 320, y: 120 },
  { id: 'left_shoulder', x: 220, y: 240 },
  { id: 'right_shoulder', x: 420, y: 240 },
  { id: 'left_elbow', x: 170, y: 360 },
  { id: 'right_elbow', x: 470, y: 360 },
  { id: 'left_wrist', x: 140, y: 460 },
  { id: 'right_wrist', x: 500, y: 460 },
  { id: 'left_hip', x: 240, y: 440 },
  { id: 'right_hip', x: 400, y: 440 },
];

export const POSE_BONES = [
  ['left_shoulder', 'right_shoulder'],
  ['left_shoulder', 'left_elbow'],
  ['left_elbow', 'left_wrist'],
  ['right_shoulder', 'right_elbow'],
  ['right_elbow', 'right_wrist'],
  ['left_shoulder', 'left_hip'],
  ['right_shoulder', 'right_hip'],
  ['left_hip', 'right_hip'],
];

