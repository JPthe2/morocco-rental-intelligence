# Local frontend

Plain HTML/CSS/JS, no build step. Needs a local static server (not `file://`) because it fetches `data.json`.

## Run it

```
cd frontend
python -m http.server 5500
```

Then open http://localhost:5500

## What it talks to

- **Dashboard / Listings tabs** — reads the static `data.json` snapshot in this folder. Regenerate it by re-running the export script against the n8n Data Table whenever you want fresher numbers.
- **Predict tab** — calls the local FastAPI service at `http://127.0.0.1:8000/predict`. Make sure it's running:
  ```
  cd ../ml/price_prediction
  ./.venv/Scripts/python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
  ```
- **Chat tab** — calls the n8n chatbot's public webhook directly from the browser. Uses OpenRouter's free daily quota, which is limited — if chat errors out, that's almost always the quota, not a bug.

Edit the `CONFIG` object at the top of the `<script>` in `index.html` if any of these URLs change.
