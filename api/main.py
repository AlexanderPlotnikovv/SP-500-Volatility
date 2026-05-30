import json
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import config

app = FastAPI(title="S&P500 Volatility Forecasting")

RESULTS_PATH = config.OUTPUTS_DIR / "results.json"
FRONTEND_PATH = Path(__file__).parent.parent / "frontend"


@app.get("/api/health")
def health():
    """Check if server is running."""
    return {"status": "ok"}


@app.get("/api/results")
def get_results():
    """Return model predictions and metrics."""
    if not RESULTS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="results.json not found — run run_pipeline.py first"
        )
    with open(RESULTS_PATH, "r") as f:
        return JSONResponse(content=json.load(f), status_code=200)


@app.get("/")
def index():
    """Serve frontend."""
    html_path = FRONTEND_PATH / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(html_path, status_code=200)


app.mount("/css", StaticFiles(directory=str(FRONTEND_PATH / "css")), name="css")
app.mount("/js", StaticFiles(directory=str(FRONTEND_PATH / "js")), name="js")

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading


    def open_browser(url="http://localhost:8000"):
        import time
        time.sleep(1)
        webbrowser.open(url)


    threading.Thread(target=open_browser).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
