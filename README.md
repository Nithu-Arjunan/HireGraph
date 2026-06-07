# HireGraph

HireGraph is a LangGraph hiring workflow demo that evaluates resumes against a job description, produces a scorecard, drafts candidate email, and records an audit trail for the run.

The active demo entrypoint is the FastAPI server in `src/demo_server.py`. It calls `src/demo_runner.py`, which builds the graph from `src/graph.py` and runs the sample candidate scenarios.

## 
Demo link - https://youtu.be/i6PslKtGbdw?si=69LMRqObGu_LIZPL

## What The Graph Demonstrates

- TypedDict state for raw resume/JD text, classification, scorecard, email drafts, audit trail, messages, and recovery fields.
- Prompt chaining: `parse_resume -> normalize_resume_skills -> extract_years_experience`.
- Structured output with Pydantic models and `with_structured_output`.
- Seniority routing: JD seniority routes through `seniority_router` into seniority-specific scoring profiles.
- Seniority-aware weighted scoring and a final `scorecard`.
- Parallel scoring with reducers for skill and dimension evaluations.
- Orchestrator/worker fan-out with `Send` for one worker per required JD skill.
- Tool-calling research agent using Tavily through a LangGraph `ToolNode`.
- LLM-recoverable loopback for research tool errors.
- Critic loop that evaluates and retries email drafts.
- Compensation path for downstream email/ATS failures.

## Setup

Create a `.env` file from `.env.example` and set the required OpenAI key:

```bash
OPENAI_API_KEY=your_openai_key
```

Optional keys for web research and sandbox email:

```bash
TAVILY_API_KEY=your_tavily_key
SMTP_HOST=sandbox.smtp.mailtrap.io
SMTP_PORT=2525
SMTP_USERNAME=your_smtp_user
SMTP_PASSWORD=your_smtp_password
SMTP_FROM_EMAIL=hiring@example.com
```

Install Python dependencies with `uv`:

```bash
uv sync
```

Build the React dashboard:

```bash
cd ui
npm install
npm run build
cd ..
```

## Run The Demo

Start the FastAPI server from the repo root:

```bash
uv run uvicorn src.demo_server:app --host 127.0.0.1 --port 8765
```

Open the dashboard:

```text
http://127.0.0.1:8765
```

Useful API endpoints:

```text
GET  /api/health
GET  /api/demo/run
POST /api/demo/resume
```

`/api/demo/run` runs the strong, borderline, and weak sample candidates. If a run pauses for human review, `/api/demo/resume` resumes the same stored graph thread for that scenario.

## Generated Artifacts

The demo writes:

```text
graph_out/hiregraph.mmd
graph_out/hiregraph.png
graph_out/scoreboard_summary.json
```

Use `graph_out/hiregraph.mmd` or `graph_out/hiregraph.png` during the demo to point at graph nodes and edges.

## Sample Data

Sample resumes live in:

```text
sample_data/resume/
```

The sample job description lives in:

```text
sample_data/jd/jd_junior_data.md
```

## Tests

Run the test suite:

```bash
uv run python -m unittest discover -s test -v
```


