/**
 * AR Try-On Screen
 * Uses the device camera and calls the python-ml service REST endpoint
 * to overlay the selected garment.  Initial release is REST-first (no
 * on-device inference); ONNX / CoreML / TFLite is a future milestone.
 */
import React, { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Pressable,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { router } from 'expo-router';
import { useARStore } from '../src/store/arStore';
import { apiClient } from '../src/services/api';

export default function TryOnScreen() {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef  = useRef<CameraView>(null);
  const [busy, setBusy] = useState(false);
  const [fitLabel, setFitLabel] = useState<string | null>(null);

  const selectedProductId = useARStore(s => s.selectedProductId);
  const sessionId         = useARStore(s => s.sessionId);
  const setSessionId      = useARStore(s => s.setSessionId);
  const setMeasurements   = useARStore(s => s.setMeasurements);
  const addFitPrediction  = useARStore(s => s.addFitPrediction);

  useEffect(() => {
    if (!permission?.granted) requestPermission();
  }, []);

  // Capture + call backend every 2 s while screen is active
  useEffect(() => {
    if (!permission?.granted || !selectedProductId) return;
    const interval = setInterval(captureAndPredict, 2000);
    return () => clearInterval(interval);
  }, [permission, selectedProductId, sessionId]);

  const captureAndPredict = async () => {
    if (busy || !cameraRef.current) return;
    setBusy(true);
    try {
      // Capture a low-res JPEG
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.4,
        base64: true,
        skipProcessing: true,
      });

      // Call the python-ml /try-on endpoint
      const { data } = await apiClient.post('/fit-prediction/predict', {
        productId:   selectedProductId,
        frameBase64: photo.base64,
      });

      if (data?.decision) {
        setFitLabel(data.decision);
        addFitPrediction(data);
      }
      if (data?.measurements) {
        setMeasurements(data.measurements);
      }
      if (data?.sessionId && !sessionId) {
        setSessionId(data.sessionId);
      }
    } catch {
      // Silently ignore on-device errors — network hiccups are expected
    } finally {
      setBusy(false);
    }
  };

  if (!permission) {
    return <View style={styles.center}><ActivityIndicator color="#00d4ff" /></View>;
  }
  if (!permission.granted) {
    return (
      <View style={styles.center}>
        <Text style={styles.permText}>Camera permission required for AR try-on.</Text>
        <Pressable style={styles.btn} onPress={requestPermission}>
          <Text style={styles.btnText}>Grant Permission</Text>
        </Pressable>
      </View>
    );
  }

  const fitColor = fitLabel === 'GOOD' ? '#00ff88'
                 : fitLabel === 'TIGHT' ? '#ff6060'
                 : fitLabel === 'LOOSE' ? '#ffd060'
                 : '#aaaaaa';

  return (
    <View style={styles.container}>
      <CameraView ref={cameraRef} style={styles.camera} facing="front" />

      {/* HUD overlay */}
      <View style={styles.hud} pointerEvents="none">
        {fitLabel && (
          <View style={[styles.fitBadge, { borderColor: fitColor }]}>
            <Text style={[styles.fitText, { color: fitColor }]}>{fitLabel}</Text>
          </View>
        )}
        {busy && <ActivityIndicator color="#00d4ff" style={styles.spinner} />}
      </View>

      {/* Controls */}
      <View style={styles.controls}>
        <Pressable style={styles.btn} onPress={() => router.back()}>
          <Text style={styles.btnText}>← Back</Text>
        </Pressable>
        <Pressable style={styles.btn} onPress={() => router.push('/profile')}>
          <Text style={styles.btnText}>My Fit</Text>
        </Pressable>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  camera:    { flex: 1 },
  center:    { flex: 1, justifyContent: 'center', alignItems: 'center', gap: 12, backgroundColor: '#0a0a0a' },
  permText:  { color: '#aaa', textAlign: 'center', paddingHorizontal: 24 },
  hud:       { ...StyleSheet.absoluteFillObject, justifyContent: 'flex-start', alignItems: 'flex-end', padding: 16 },
  fitBadge:  { borderWidth: 2, borderRadius: 8, paddingHorizontal: 14, paddingVertical: 6, backgroundColor: 'rgba(0,0,0,0.6)' },
  fitText:   { fontSize: 18, fontWeight: '800', letterSpacing: 1 },
  spinner:   { marginTop: 8 },
  controls:  { position: 'absolute', bottom: 32, left: 0, right: 0, flexDirection: 'row', justifyContent: 'space-around' },
  btn:       { paddingHorizontal: 24, paddingVertical: 12, backgroundColor: 'rgba(0,0,0,0.7)', borderRadius: 24, borderWidth: 1, borderColor: '#00d4ff' },
  btnText:   { color: '#00d4ff', fontWeight: '700' },
});
