# Solution Overview

This document defines the intended repository structure and ownership boundaries for AR Mirror.

## System Layers

1. Python runtime layer
Purpose: camera ingestion, body/pose inference, fitting, compositing, stream publishing.
Primary paths: app.py, web_server.py, src/

2. Backend API layer
Purpose: measurement/session/product persistence and API endpoints for frontend workflows.
Primary paths: backend/src, backend/prisma

3. Frontend UI layer
Purpose: fit flow UX, live overlay controls, backend integration.
Primary paths: frontend/ar-mirror-ui/src

## Ownership Boundaries

- Runtime code belongs under src/ and top-level entrypoints.
- Backend code belongs under backend/src with DTOs/services/controllers grouped by module.
- Frontend code belongs under frontend/ar-mirror-ui/src with components, hooks, and store separated.
- Automation and diagnostics belong under scripts/.
- Architecture and deployment docs belong under docs/.

## Required Files For Complete Handoff

- app.py
- web_server.py
- package.json
- backend/package.json
- backend/tsconfig.json
- backend/prisma/schema.prisma
- frontend/ar-mirror-ui/package.json
- pyrightconfig.json
- requirements.txt
- Dockerfile.backend
- docker-compose.yml
- docs/ARCHITECTURE.md
- docs/DEPLOYMENT.md

## Validation Baseline

Run from repository root:

```powershell
npm run backend:build
npm run ui:build
python -m py_compile app.py
python -m py_compile web_server.py
```

If these pass and required files are present, the repository is considered structurally complete for handoff.
