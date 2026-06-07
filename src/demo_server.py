from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.demo_runner import resume_hiregraph_scenario, run_hiregraph_scenarios


ROOT = Path(__file__).resolve().parents[1]
UI_DIST = ROOT / "ui" / "dist"
GRAPH_OUT = ROOT / "graph_out"

app = FastAPI(
    title="HireGraph Demo API",
    version="0.1.0",
    description="Runs HireGraph demo scenarios and serves the React dashboard.",
)

if GRAPH_OUT.exists():
    app.mount("/graph_out", StaticFiles(directory=GRAPH_OUT), name="graph_out")

if (UI_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=UI_DIST / "assets"), name="assets")


class ResumeRequest(BaseModel):
    scenario: str
    decision: str
    notes: str = ""


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/demo/run")
def run_demo() -> dict:
    try:
        return run_hiregraph_scenarios()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": exc.__class__.__name__,
                "message": str(exc),
            },
        ) from exc


@app.post("/api/demo/resume")
def resume_demo(request: ResumeRequest) -> dict:
    try:
        return resume_hiregraph_scenario(
            request.scenario,
            request.decision,
            request.notes,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": exc.__class__.__name__,
                "message": str(exc),
            },
        ) from exc


@app.get("/{path:path}")
def serve_dashboard(path: str):
    requested = UI_DIST / path
    if requested.is_file():
        return FileResponse(requested)

    index = UI_DIST / "index.html"
    if index.exists():
        return FileResponse(index)

    fallback = ROOT / "ui" / "index.html"
    if fallback.exists():
        return FileResponse(fallback)

    raise HTTPException(status_code=404, detail="UI build not found.")


def main() -> None:
    uvicorn.run("src.demo_server:app", host="127.0.0.1", port=8765, reload=False)


if __name__ == "__main__":
    main()
