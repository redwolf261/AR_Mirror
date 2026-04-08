import { AnimatePresence, motion } from 'framer-motion';
import FitEngineView from './components/fit-engine/FitEngineView';
import GarmentView from './components/garment-overlay/GarmentView';
import Navbar from './components/layout/Navbar';
import Sidebar from './components/layout/Sidebar';
import { useFrameTime } from './hooks/useFrameTime';
import { useWebSocket } from './hooks/useWebSocket';
import { AppStoreProvider, useAppStore } from './store/appStore';
import styles from './App.module.css';

function AppContent() {
  const socket = useWebSocket();
  const { fps } = useFrameTime();
  const { mode, garments, selectedGarment, setSelectedGarmentId } = useAppStore();

  return (
    <div className={styles.app}>
      <Navbar connected={socket.connected} fps={fps} />
      <Sidebar />

      <AnimatePresence mode="wait">
        <motion.div
          key={mode}
          initial={{ opacity: 0, scale: 0.97 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 1.02 }}
          transition={{ duration: 0.35 }}
        >
          {mode === 'fit-engine' ? (
            <FitEngineView />
          ) : (
            <GarmentView
              garments={garments}
              selectedGarment={selectedGarment}
              onSelect={setSelectedGarmentId}
              fps={fps}
            />
          )}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}

export default function App() {
  return (
    <AppStoreProvider>
      <AppContent />
    </AppStoreProvider>
  );
}
