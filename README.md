# AR Mirror

AR Mirror is a virtual try-on platform with a Python real-time inference/rendering app, a NestJS backend API, and a React frontend UI.

This repository contains the full implementation used for local development and deployment.

## Solution Summary

The system is organized into three layers:

1. Runtime engine (Python): captures camera frames, estimates pose/body signals, applies garment fitting/warping, and serves debug/live state.
2. Backend API (NestJS + Prisma): manages products, measurements, sessions, and FitEngine endpoints used by the UI.
3. Frontend app (React): provides operator and user-facing flows (fit capture, size summary, and overlay controls).

## Architecture At A Glance

- Entry point: app.py
- Web stream/state bridge: web_server.py
- Core fitting and rendering modules: src/core and src/app
- Backend service: backend/src
- Frontend app: frontend/ar-mirror-ui/src

Request/processing flow:

1. Camera frame enters app.py.
2. Pose/segmentation workers update shared buffers.
3. Garment fitting and compositing produce output frames.
4. web_server.py publishes MJPEG stream and JSON state.
5. Frontend consumes stream/state and drives user actions.
6. Backend persists measurement/session/product data.

## Repository Structure

Top-level directories and purpose:

- src: Python runtime modules (rendering, fitting, pipelines)
- backend: NestJS backend API and Prisma schema
- frontend/ar-mirror-ui: React frontend application
- scripts: automation, validation, and maintenance scripts
- docs: design, architecture, deployment, and planning documents
- models: model checkpoints and ONNX assets (local/runtime dependent)
- tests: Python tests
- config: environment and configuration files

Structure standards:

- Repository structure contract: [docs/SOLUTION_OVERVIEW.md](docs/SOLUTION_OVERVIEW.md)
- Formatting baseline: [.editorconfig](.editorconfig)

## Getting Started

### Prerequisites

- Python 3.12
- Node.js 18+
- npm
- Webcam
- (Optional) NVIDIA CUDA-capable GPU

### Python Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Backend Setup

```powershell
cd backend
npm install
npm run prisma:generate
npm run build
cd ..
```

### Frontend Setup

```powershell
cd frontend/ar-mirror-ui
npm install
npm run build
cd ../..
```

## Running The System

### Main app (Python runtime)

```powershell
python app.py
```

### Backend API

```powershell
cd backend
npm run start:dev
```

### Frontend UI

```powershell
cd frontend/ar-mirror-ui
npm start
```

### Workspace stabilization check

```powershell
npm run stabilize
```

## Validation Commands

Use these before sharing or deploying:

```powershell
# Python type and import sanity
python -m py_compile app.py
python -m py_compile web_server.py

# Backend build
cd backend
npm run build
cd ..

# Frontend build
cd frontend/ar-mirror-ui
npm run build
cd ../..
```

## Included Files Checklist

The following files are required and included for a complete solution handoff:

- app.py: primary runtime entry point
- web_server.py: stream and API bridge for live UI state
- package.json: workspace scripts (stabilize/build/start helpers)
- backend/package.json: backend build/run scripts
- backend/tsconfig.json: backend TypeScript compiler configuration
- backend/prisma/schema.prisma: persistence schema
- frontend/ar-mirror-ui/package.json: frontend scripts and dependencies
- pyrightconfig.json: Python static analysis scope for maintained code
- requirements.txt: Python dependencies
- Dockerfile.backend and docker-compose.yml: container/deployment support
- docs/ARCHITECTURE.md and docs/DEPLOYMENT.md: system and deployment docs

## Documentation Index

Primary references:

- docs/ARCHITECTURE.md
- docs/DEPLOYMENT.md
- docs/SOLUTION_OVERVIEW.md
- docs/EXECUTION_PLAN.md
- docs/AUTONOMOUS_ITERATION_LOOP.md
- QUICKSTART.md

## Notes On Scope

This repository contains both active modules and historical/experimental assets. Static analysis is intentionally scoped to maintained runtime paths so day-to-day diagnostics stay actionable.

## License

MIT. See LICENSE.
