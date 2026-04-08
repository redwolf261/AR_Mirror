import { createContext, useContext, useMemo, useState } from 'react';
import React from 'react';
import { MOCK_GARMENTS, MOCK_MEASUREMENTS } from '../data/mockData';

const AppStoreContext = createContext(null);

export function AppStoreProvider({ children }) {
  const [mode, setMode] = useState('fit-engine');
  const [selectedGarmentId, setSelectedGarmentId] = useState(MOCK_GARMENTS[0].id);
  const [measurements] = useState(MOCK_MEASUREMENTS);
  const [sessionActive, setSessionActive] = useState(false);
  const [captureState, setCaptureState] = useState({ front: false, side: false });

  const selectedGarment = useMemo(
    () => MOCK_GARMENTS.find((g) => g.id === selectedGarmentId) || MOCK_GARMENTS[0],
    [selectedGarmentId]
  );

  const value = {
    mode,
    setMode,
    garments: MOCK_GARMENTS,
    selectedGarment,
    selectedGarmentId,
    setSelectedGarmentId,
    measurements,
    sessionActive,
    setSessionActive,
    captureState,
    setCaptureState,
  };

  return React.createElement(AppStoreContext.Provider, { value }, children);
}

export function useAppStore() {
  const ctx = useContext(AppStoreContext);
  if (!ctx) {
    throw new Error('useAppStore must be used inside AppStoreProvider');
  }
  return ctx;
}
