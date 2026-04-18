import { createContext, useContext, useState } from 'react';
import React from 'react';

const AppStoreContext = createContext(null);

export function AppStoreProvider({ children }) {
  const [mode, setMode] = useState('fit-engine');
  const [selectedGarmentId, setSelectedGarmentId] = useState(null);

  const value = {
    mode,
    setMode,
    selectedGarmentId,
    setSelectedGarmentId,
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
