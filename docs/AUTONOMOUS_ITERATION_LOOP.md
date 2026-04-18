# Autonomous Iteration Loop

This workspace now includes a live iteration system so the app can be observed and adjusted continuously.

## What It Does

`npm run autoloop` will:

1. Start backend (`app.py`) if not already running.
2. Start frontend (`frontend/ar-mirror-ui`) if not already running.
3. Open the app in browser (`http://localhost:3001`).
4. Poll backend snapshot API (`http://localhost:5051/api/snapshot`) and write artifacts to `logs/live-monitor/`.

Artifacts generated:

- `latest_frame.jpg`
- `latest_state.json`
- timestamped `frame_*.jpg` and `state_*.json` history

## Run

```powershell
npm run autoloop
```

Optional direct run:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\autonomous-dev-loop.ps1 -BackendPhase 0 -MonitorIntervalSec 1.2
```

## Why This Helps Autonomous Iteration

- The visual output is persisted every cycle, so behavior changes can be inspected without manual screenshot capture.
- State snapshots (`latest_state.json`) expose readiness/session/measurement values in sync with each frame.
- This gives a repeatable loop for diagnose -> edit -> verify.

## Monitor Only

If backend/frontend are already running, monitor directly:

```powershell
.\.venv\Scripts\python.exe .\scripts\live_snapshot_monitor.py --api-url http://localhost:5051/api/snapshot --out-dir logs/live-monitor --interval 1.5
```
