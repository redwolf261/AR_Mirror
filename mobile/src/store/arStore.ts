/**
 * Chic India Mobile - State Management with Zustand
 * Global app state for AR measurements and fit predictions
 */
import { create } from 'zustand';
import { Measurement, FitPredictionResponse } from '../services/api';

interface ARState {
  // Session
  sessionId: string | null;
  setSessionId: (id: string) => void;

  // Measurements
  currentMeasurements: Measurement | null;
  setMeasurements: (measurements: Measurement) => void;

  // Fit predictions
  fitPredictions: FitPredictionResponse[];
  addFitPrediction: (prediction: FitPredictionResponse) => void;

  // Selected product
  selectedProductId: string | null;
  setSelectedProduct: (id: string | null) => void;

  // UI state
  isCameraActive: boolean;
  setCameraActive: (active: boolean) => void;

  isProcessing: boolean;
  setProcessing: (processing: boolean) => void;

  // Reset
  reset: () => void;
}

export const useARStore = create<ARState>((set) => ({
  // Initial state
  sessionId: null,
  currentMeasurements: null,
  fitPredictions: [],
  selectedProductId: null,
  isCameraActive: false,
  isProcessing: false,

  // Actions
  setSessionId: (id) => set({ sessionId: id }),

  setMeasurements: (measurements) => set({ currentMeasurements: measurements }),

  addFitPrediction: (prediction) => 
    set((state) => ({ 
      fitPredictions: [prediction, ...state.fitPredictions] 
    })),

  setSelectedProduct: (id) => set({ selectedProductId: id }),

  setCameraActive: (active) => set({ isCameraActive: active }),

  setProcessing: (processing) => set({ isProcessing: processing }),

  reset: () => set({
    sessionId: null,
    currentMeasurements: null,
    fitPredictions: [],
    selectedProductId: null,
    isCameraActive: false,
    isProcessing: false
  })
}));
