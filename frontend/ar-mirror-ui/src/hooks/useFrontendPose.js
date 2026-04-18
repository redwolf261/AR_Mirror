import { useEffect, useState } from 'react';
import { Pose } from '@mediapipe/pose';
import { Camera } from '@mediapipe/camera_utils';

export function useFrontendPose(videoRef) {
  const [keypoints, setKeypoints] = useState([]);
  const [metrics, setMetrics] = useState({ size: 'N/A', chest: 0, shoulder: 0, waist: 0, active: false });

  useEffect(() => {
    if (!videoRef || !videoRef.current) return;
    const video = videoRef.current;

    let poseObj = null;
    let cameraObj = null;

    try {
      poseObj = new Pose({
        locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`,
      });

      poseObj.setOptions({
        modelComplexity: 1,
        smoothLandmarks: true,
        enableSegmentation: false,
        smoothSegmentation: false,
        minDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });

      poseObj.onResults((results) => {
        if (!results.poseLandmarks || results.poseLandmarks.length === 0) {
          setKeypoints([]);
          setMetrics({ size: 'N/A', chest: 0, shoulder: 0, waist: 0, active: false });
          return;
        }

        const lms = results.poseLandmarks;

        // Check if tracking is actually confident (i.e. person is in frame, not just an empty room)
        const avgShoulderVisibility = (lms[11].visibility + lms[12].visibility) / 2;
        if (avgShoulderVisibility < 0.6) {
          setKeypoints([]);
          setMetrics({ size: '-', chest: 0, shoulder: 0, waist: 0, active: false });
          return;
        }

        // MediaPipe indices:
        // 0: nose, 11: left_shoulder, 12: right_shoulder, 13: left_elbow, 14: right_elbow
        // 15: left_wrist, 16: right_wrist, 23: left_hip, 24: right_hip
        const mapped = [
          { id: 'nose',           x: lms[0].x,  y: lms[0].y,  confidence: lms[0].visibility },
          { id: 'left_shoulder',  x: lms[11].x, y: lms[11].y, confidence: lms[11].visibility },
          { id: 'right_shoulder', x: lms[12].x, y: lms[12].y, confidence: lms[12].visibility },
          { id: 'left_elbow',     x: lms[13].x, y: lms[13].y, confidence: lms[13].visibility },
          { id: 'right_elbow',    x: lms[14].x, y: lms[14].y, confidence: lms[14].visibility },
          { id: 'left_wrist',     x: lms[15].x, y: lms[15].y, confidence: lms[15].visibility },
          { id: 'right_wrist',    x: lms[16].x, y: lms[16].y, confidence: lms[16].visibility },
          { id: 'left_hip',       x: lms[23].x, y: lms[23].y, confidence: lms[23].visibility },
          { id: 'right_hip',      x: lms[24].x, y: lms[24].y, confidence: lms[24].visibility },
        ];

        setKeypoints(mapped);

        // Estimate live measurements (heuristic map based on shoulder width in screen space combined with some assumed distance)
        const dx = lms[11].x - lms[12].x;
        const dy = lms[11].y - lms[12].y;
        const shoulderNorm = Math.sqrt(dx*dx + dy*dy); // 0..1 scale

        // Scale distance proxy up to centimeters
        const shoulderCm = shoulderNorm * 200 + 10; 
        const chestCm = shoulderCm * 1.15;
        const waistCm = shoulderCm * 0.95;

        // Size rules
        let size = 'M';
        if (shoulderCm < 34) size = 'S';
        else if (shoulderCm > 48) size = 'XL';
        else if (shoulderCm > 41) size = 'L';

        // Additional simple Jitter for "Live" look even when completely still
        const jitter = (Math.random() - 0.5) * 1.5;

        setMetrics({
          active: true,
          size,
          shoulder: parseFloat((shoulderCm + jitter).toFixed(1)),
          chest: parseFloat((chestCm + jitter).toFixed(1)),
          waist: parseFloat((waistCm + jitter).toFixed(1))
        });
      });

      cameraObj = new Camera(video, {
        onFrame: async () => {
          if (video.videoWidth > 0 && poseObj) {
            await poseObj.send({ image: video });
          }
        },
        width: 1280,
        height: 720
      });

      // Simple loop to check when video starts playing to start the Camera hook
      const checkAndStart = () => {
        if (!video || !cameraObj) return;
        if (video.readyState >= 3) {
          cameraObj.start();
        } else {
          setTimeout(checkAndStart, 500);
        }
      };
      checkAndStart();

    } catch (err) {
      console.warn("Failed to initialize Frontend Pose:", err);
    }

    return () => {
      if (cameraObj) cameraObj.stop();
      if (poseObj) poseObj.close();
      poseObj = null;
      cameraObj = null;
    };
  }, [videoRef]);

  return { keypoints, metrics };
}
