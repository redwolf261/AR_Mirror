/**
 * My Measurements Profile Screen
 * Shows the latest body measurements captured during AR try-on sessions.
 */
import React from 'react';
import { ScrollView, StyleSheet, Text, View } from 'react-native';
import { useARStore } from '../src/store/arStore';

interface MeasRow { label: string; value: string | null }

export default function ProfileScreen() {
  const measurements   = useARStore(s => s.currentMeasurements);
  const fitPredictions = useARStore(s => s.fitPredictions);

  const rows: MeasRow[] = measurements ? [
    { label: 'Shoulder Width', value: `${measurements.shoulderWidthCm?.toFixed(1)} cm` },
    { label: 'Chest Width',    value: `${measurements.chestWidthCm?.toFixed(1)} cm` },
    { label: 'Torso Length',   value: `${measurements.torsoLengthCm?.toFixed(1)} cm` },
    { label: 'Waist',          value: measurements.waistCm     ? `${measurements.waistCm.toFixed(1)} cm`     : '—' },
    { label: 'Hip',            value: measurements.hipCm       ? `${measurements.hipCm.toFixed(1)} cm`       : '—' },
    { label: 'Inseam',         value: measurements.inseamCm    ? `${measurements.inseamCm.toFixed(1)} cm`    : '—' },
  ] : [];

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      <Text style={styles.sectionTitle}>Body Measurements</Text>

      {rows.length === 0 ? (
        <Text style={styles.emptyText}>
          No measurements yet.{'\n'}Complete an AR try-on session to see your data here.
        </Text>
      ) : (
        <View style={styles.card}>
          {rows.map(row => (
            <View key={row.label} style={styles.row}>
              <Text style={styles.rowLabel}>{row.label}</Text>
              <Text style={styles.rowValue}>{row.value}</Text>
            </View>
          ))}
        </View>
      )}

      {fitPredictions.length > 0 && (
        <>
          <Text style={[styles.sectionTitle, { marginTop: 24 }]}>Recent Fit Predictions</Text>
          {fitPredictions.slice(0, 5).map((fp, idx) => {
            const col = fp.decision === 'GOOD' ? '#00ff88'
                      : fp.decision === 'TIGHT' ? '#ff6060' : '#ffd060';
            return (
              <View key={idx} style={styles.card}>
                <View style={styles.row}>
                  <Text style={styles.rowLabel}>Fit</Text>
                  <Text style={[styles.rowValue, { color: col }]}>{fp.decision}</Text>
                </View>
                <View style={styles.row}>
                  <Text style={styles.rowLabel}>Confidence</Text>
                  <Text style={styles.rowValue}>{(fp.confidence * 100).toFixed(0)}%</Text>
                </View>
              </View>
            );
          })}
        </>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container:    { flex: 1, backgroundColor: '#0a0a0a' },
  content:      { padding: 16, paddingBottom: 40 },
  sectionTitle: { color: '#ffffff', fontSize: 18, fontWeight: '700', marginBottom: 12 },
  emptyText:    { color: '#666', textAlign: 'center', lineHeight: 22, marginTop: 40 },
  card:         { backgroundColor: '#1a1a1a', borderRadius: 12, overflow: 'hidden', marginBottom: 12 },
  row:          { flexDirection: 'row', justifyContent: 'space-between', padding: 14, borderBottomWidth: StyleSheet.hairlineWidth, borderBottomColor: '#2a2a2a' },
  rowLabel:     { color: '#888', fontSize: 14 },
  rowValue:     { color: '#fff', fontSize: 14, fontWeight: '600' },
});
