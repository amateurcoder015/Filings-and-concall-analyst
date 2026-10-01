# Filings & Concall Analyst frontend

React + TypeScript + Tailwind + Vite three-pane workspace for cited Q&A.

- Install: `npm install`
- Develop: `npm run dev` (http://localhost:5173)
- Test: `npx vitest --run` (or `npm test` for watch mode)
- Build: `npm run build`

The dev server proxies `/api/*` to the backend at `http://127.0.0.1:8000` by default. Start the backend
bound to loopback on that port:

```
uvicorn backend.api.server:app --host 127.0.0.1 --port 8000
```

To point the proxy elsewhere, set `VITE_API_TARGET`, for example
`VITE_API_TARGET=http://127.0.0.1:8001 npm run dev`. The default uses `127.0.0.1` rather than
`localhost` because `localhost` can resolve to a different process listening on IPv6 port 8000.
