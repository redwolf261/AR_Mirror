# System Stabilization Guide

This project is now in stabilization mode: no feature iteration, only reliability, performance consistency, and release hardening.

## Stabilization Gate

Run from workspace root:

```powershell
.\stabilize.ps1
```

Checks executed:
- Workspace sanity (required files)
- Python dependency import smoke
- Backend API health smoke using app_cloud.py on port 5050
- Frontend production build in web-ui
- NestJS backend production build in backend

Exit code behavior:
- 0 = all checks passed
- 1 = at least one check failed

## Optional Flags

```powershell
# Skip selected gates
.\stabilize.ps1 -SkipPythonSmoke
.\stabilize.ps1 -SkipFrontendBuild
.\stabilize.ps1 -SkipBackendBuild
```

## Stabilization Policy

- Freeze feature development until the gate passes consistently.
- Treat every failing gate as a release blocker.
- Only accept changes that reduce operational risk.
- Run gate before merge and before release tagging.

## Minimum Release Criteria

- Stabilization gate passes locally.
- App boots and serves frontend and backend endpoints.
- No new critical errors in logs.
- Build artifacts are reproducible from a clean checkout.
