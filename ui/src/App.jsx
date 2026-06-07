import { useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Clock3,
  FileText,
  GitBranch,
  Mail,
  Play,
  RefreshCcw,
  XCircle
} from "lucide-react";

const recommendationStyles = {
  advance: "bg-emerald-100 text-emerald-800 border-emerald-200",
  borderline: "bg-amber-100 text-amber-800 border-amber-200",
  reject: "bg-rose-100 text-rose-800 border-rose-200",
  unknown: "bg-stone-100 text-stone-700 border-stone-200"
};

function badgeClass(recommendation) {
  return recommendationStyles[recommendation] || recommendationStyles.unknown;
}

function App() {
  const [payload, setPayload] = useState(null);
  const [selectedScenario, setSelectedScenario] = useState("strong");
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [reviewDecision, setReviewDecision] = useState("approved");
  const [reviewNotes, setReviewNotes] = useState("");

  const decisions = payload?.decisions || [];
  const selected = useMemo(() => {
    return decisions.find((item) => item.scenario === selectedScenario) || decisions[0];
  }, [decisions, selectedScenario]);

  async function runDemo() {
    setStatus("running");
    setError("");
    try {
      const response = await fetch("/api/demo/run");
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.message || `Request failed with ${response.status}`);
      }
      const data = await response.json();
      setPayload(data);
      setSelectedScenario(data.decisions?.[0]?.scenario || "strong");
      setStatus("completed");
    } catch (err) {
      setError(err.message);
      setStatus("failed");
    }
  }

  async function submitHumanReview() {
    if (!selected) {
      return;
    }

    setStatus("running");
    setError("");
    try {
      const response = await fetch("/api/demo/resume", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          scenario: selected.scenario,
          decision: reviewDecision,
          notes: reviewNotes
        })
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        const message = body.detail?.message || body.message || `Request failed with ${response.status}`;
        throw new Error(message);
      }
      const data = await response.json();
      setPayload(data);
      setSelectedScenario(selected.scenario);
      setReviewNotes("");
      setStatus("completed");
    } catch (err) {
      setError(err.message);
      setStatus("failed");
    }
  }

  return (
    <main className="min-h-screen bg-paper text-ink">
      <section className="border-b border-stone-200 bg-white">
        <div className="mx-auto flex max-w-7xl flex-col gap-4 px-5 py-5 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <h1 className="text-2xl font-semibold tracking-normal">HireGraph Demo Console</h1>
            <p className="mt-1 text-sm text-stone-600">
              Compile graph, render PNG, run three candidate scenarios, and inspect Decision outputs.
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <CandidateSelect
              decisions={decisions}
              selectedScenario={selectedScenario}
              onSelect={setSelectedScenario}
            />
            <StatusPill status={status} />
            <button
              className="inline-flex h-10 items-center gap-2 rounded-md bg-ink px-4 text-sm font-medium text-white shadow-soft transition hover:bg-moss disabled:cursor-not-allowed disabled:opacity-60"
              disabled={status === "running"}
              onClick={runDemo}
              title="Run the full HireGraph demo"
            >
              {status === "running" ? <RefreshCcw className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              Run Demo
            </button>
          </div>
        </div>
      </section>

      <section className="mx-auto grid max-w-7xl gap-5 px-5 py-5">
        {error ? <ErrorBanner message={error} /> : null}

        <div className="grid gap-5 lg:grid-cols-[1.35fr_0.65fr]">
          <Scoreboard payload={payload} selectedScenario={selectedScenario} onSelect={setSelectedScenario} />
          <GraphPanel payload={payload} />
        </div>

        <HumanReviewPanel
          decision={selected}
          reviewDecision={reviewDecision}
          reviewNotes={reviewNotes}
          onDecisionChange={setReviewDecision}
          onNotesChange={setReviewNotes}
          onSubmit={submitHumanReview}
          disabled={status === "running"}
        />

        <div className="grid gap-5 xl:grid-cols-[1fr_0.9fr]">
          <DecisionPanel decision={selected} />
          <EmailPanel decision={selected} />
        </div>

        <AuditTrail decision={selected} />
      </section>
    </main>
  );
}

function CandidateSelect({ decisions, selectedScenario, onSelect }) {
  return (
    <label className="grid gap-1 text-xs font-medium text-stone-500">
      Candidate
      <select
        className="h-10 min-w-64 rounded-md border border-stone-200 bg-stone-50 px-3 text-sm font-medium text-ink outline-none transition focus:border-moss disabled:cursor-not-allowed disabled:opacity-60"
        disabled={!decisions.length}
        value={decisions.length ? selectedScenario : ""}
        onChange={(event) => onSelect(event.target.value)}
      >
        {decisions.length ? (
          decisions.map((item) => (
            <option key={item.scenario} value={item.scenario}>
              {item.candidate_name} - {item.label}
            </option>
          ))
        ) : (
          <option value="">Run demo to select candidate</option>
        )}
      </select>
    </label>
  );
}

function StatusPill({ status }) {
  const icon = {
    idle: <Clock3 className="h-4 w-4" />,
    running: <RefreshCcw className="h-4 w-4 animate-spin" />,
    completed: <CheckCircle2 className="h-4 w-4" />,
    failed: <XCircle className="h-4 w-4" />
  }[status];

  return (
    <div className="inline-flex h-10 items-center gap-2 rounded-md border border-stone-200 bg-stone-50 px-3 text-sm capitalize text-stone-700">
      {icon}
      {status}
    </div>
  );
}

function ErrorBanner({ message }) {
  return (
    <div className="flex items-center gap-3 rounded-md border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">
      <AlertTriangle className="h-5 w-5 shrink-0" />
      {message}
    </div>
  );
}

function Scoreboard({ payload, selectedScenario, onSelect }) {
  const summary = payload?.scoreboard;
  const decisions = payload?.decisions || [];

  return (
    <section className="rounded-md border border-stone-200 bg-white p-4 shadow-soft">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Activity className="h-5 w-5 text-jade" />
          <h2 className="text-base font-semibold">Scoreboard</h2>
        </div>
        <div className="text-sm text-stone-600">
          Avg {summary?.average_score ?? "--"} across {summary?.total ?? 0}
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-3">
        {decisions.length ? (
          decisions.map((item) => (
            <button
              key={item.scenario}
              className={`rounded-md border p-4 text-left transition ${
                selectedScenario === item.scenario
                  ? "border-ink bg-stone-50"
                  : "border-stone-200 bg-white hover:border-moss"
              }`}
              onClick={() => onSelect(item.scenario)}
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <div className="text-sm font-semibold">{item.label}</div>
                  <div className="mt-1 text-xs text-stone-500">{item.candidate_name}</div>
                </div>
                <span className={`rounded-md border px-2 py-1 text-xs font-medium ${badgeClass(item.decision.recommendation)}`}>
                  {item.run_status === "interrupted" ? "review" : item.decision.recommendation}
                </span>
              </div>
              <div className="mt-4 flex items-end justify-between">
                <div className="text-3xl font-semibold">{item.decision.scorecard.final_score}</div>
                <div className="text-xs text-stone-500">{item.elapsed_ms} ms</div>
              </div>
            </button>
          ))
        ) : (
          <EmptyState icon={<Play className="h-5 w-5" />} text="Run the demo to populate all three scenarios." />
        )}
      </div>
    </section>
  );
}

function HumanReviewPanel({
  decision,
  reviewDecision,
  reviewNotes,
  onDecisionChange,
  onNotesChange,
  onSubmit,
  disabled
}) {
  if (!decision || decision.run_status !== "interrupted") {
    return null;
  }

  const message = decision.interrupt?.value?.message || "Human review required for this candidate.";

  return (
    <section className="rounded-md border border-amber-200 bg-amber-50 p-4 shadow-soft">
      <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <div className="flex items-center gap-2 text-amber-900">
            <AlertTriangle className="h-5 w-5" />
            <h2 className="text-base font-semibold">Human Review Required</h2>
          </div>
          <p className="mt-1 text-sm text-amber-800">{message}</p>
        </div>
        <span className="w-fit rounded-md border border-amber-300 bg-white px-2 py-1 text-xs font-medium text-amber-800">
          {decision.candidate_name}
        </span>
      </div>

      <div className="grid gap-3 lg:grid-cols-[220px_1fr_auto]">
        <label className="grid gap-1 text-xs font-medium text-amber-900">
          Decision
          <select
            className="h-10 rounded-md border border-amber-200 bg-white px-3 text-sm text-ink outline-none focus:border-amber"
            value={reviewDecision}
            onChange={(event) => onDecisionChange(event.target.value)}
            disabled={disabled}
          >
            <option value="approved">Approve and draft invite</option>
            <option value="rejected">Reject and draft rejection</option>
          </select>
        </label>

        <label className="grid gap-1 text-xs font-medium text-amber-900">
          Reviewer Notes
          <input
            className="h-10 rounded-md border border-amber-200 bg-white px-3 text-sm text-ink outline-none focus:border-amber"
            value={reviewNotes}
            onChange={(event) => onNotesChange(event.target.value)}
            placeholder="Add notes for the audit trail"
            disabled={disabled}
          />
        </label>

        <button
          className="mt-auto inline-flex h-10 items-center justify-center gap-2 rounded-md bg-amber px-4 text-sm font-semibold text-white transition hover:bg-amber-700 disabled:cursor-not-allowed disabled:opacity-60"
          onClick={onSubmit}
          disabled={disabled}
        >
          <CheckCircle2 className="h-4 w-4" />
          Submit Review
        </button>
      </div>
    </section>
  );
}

function GraphPanel({ payload }) {
  const graphSrc = payload?.graph_png ? `/${payload.graph_png}?v=${Date.now()}` : null;

  return (
    <section className="rounded-md border border-stone-200 bg-white p-4 shadow-soft">
      <div className="mb-4 flex items-center gap-2">
        <GitBranch className="h-5 w-5 text-moss" />
        <h2 className="text-base font-semibold">Compiled Graph</h2>
      </div>
      {graphSrc ? (
        <div className="flex h-56 items-center justify-center overflow-hidden rounded-md border border-stone-200 bg-stone-50">
          <img className="max-h-full max-w-full object-contain" src={graphSrc} alt="HireGraph compiled graph" />
        </div>
      ) : (
        <EmptyState icon={<GitBranch className="h-5 w-5" />} text="Graph PNG appears here after a run." />
      )}
    </section>
  );
}

function DecisionPanel({ decision }) {
  const scorecard = decision?.decision?.scorecard;
  const dimensions = scorecard?.dimension_scores || [];
  const skills = scorecard?.skill_scores || [];

  return (
    <section className="rounded-md border border-stone-200 bg-white p-4 shadow-soft">
      <div className="mb-4 flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <FileText className="h-5 w-5 text-jade" />
          <h2 className="text-base font-semibold">Decision Object</h2>
        </div>
        {decision ? (
          <span className={`rounded-md border px-2 py-1 text-xs font-medium ${badgeClass(decision.decision.recommendation)}`}>
            {decision.decision.recommendation}
          </span>
        ) : null}
      </div>

      {decision ? (
        <div className="grid gap-4">
          <div className="grid gap-3 sm:grid-cols-3">
            <Metric label="Candidate" value={decision.candidate_name} />
            <Metric label="Final Score" value={scorecard.final_score} />
            <Metric label="Scenario" value={decision.label} />
          </div>
          <ScoreTable title="Dimension Scores" rows={dimensions} />
          <ScoreTable title="Required Skill Scores" rows={skills} />
          <p className="rounded-md bg-stone-50 p-3 text-sm leading-6 text-stone-700">
            {decision.decision.recommendation_reasoning || "No recommendation reasoning returned."}
          </p>
        </div>
      ) : (
        <EmptyState icon={<FileText className="h-5 w-5" />} text="Decision details appear after the demo run." />
      )}
    </section>
  );
}

function Metric({ label, value }) {
  return (
    <div className="rounded-md border border-stone-200 bg-stone-50 p-3">
      <div className="text-xs uppercase text-stone-500">{label}</div>
      <div className="mt-1 truncate text-lg font-semibold">{value ?? "--"}</div>
    </div>
  );
}

function ScoreTable({ title, rows }) {
  return (
    <div>
      <h3 className="mb-2 text-sm font-semibold">{title}</h3>
      <div className="overflow-hidden rounded-md border border-stone-200">
        <table className="w-full table-fixed text-sm">
          <thead className="bg-stone-50 text-left text-xs uppercase text-stone-500">
            <tr>
              <th className="w-36 px-3 py-2">Name</th>
              <th className="w-20 px-3 py-2">Score</th>
              <th className="px-3 py-2">Evidence</th>
            </tr>
          </thead>
          <tbody>
            {rows.length ? (
              rows.map((row, index) => (
                <tr className="border-t border-stone-200" key={`${row.name}-${index}`}>
                  <td className="px-3 py-2 font-medium">{row.name}</td>
                  <td className="px-3 py-2">{row.score}</td>
                  <td className="px-3 py-2 text-stone-600">{row.evidence || row.reasoning || "--"}</td>
                </tr>
              ))
            ) : (
              <tr>
                <td className="px-3 py-4 text-stone-500" colSpan="3">No score rows returned.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function EmailPanel({ decision }) {
  const actionStatus = decision?.decision?.action_status || {};

  return (
    <section className="rounded-md border border-stone-200 bg-white p-4 shadow-soft">
      <div className="mb-4 flex items-center gap-2">
        <Mail className="h-5 w-5 text-coral" />
        <h2 className="text-base font-semibold">Candidate Email</h2>
      </div>
      {decision ? (
        <div className="grid gap-4">
          <div className="rounded-md border border-stone-200 bg-stone-50 p-3 text-sm">
            <div className="grid gap-1 text-stone-600">
              <div>To: {decision.candidate_email || "--"}</div>
              <div>Scenario: {decision.label}</div>
            </div>
          </div>
          <pre className="max-h-96 overflow-auto whitespace-pre-wrap rounded-md border border-stone-200 bg-white p-4 text-sm leading-6 text-stone-700">
            {decision.decision.email_draft || "No email draft returned."}
          </pre>
          <div className="grid grid-cols-2 gap-2 text-sm">
            {Object.entries(actionStatus).map(([key, value]) => (
              <div className="flex items-center justify-between rounded-md bg-stone-50 px-3 py-2" key={key}>
                <span className="text-stone-600">{key.replaceAll("_", " ")}</span>
                <span className={value ? "font-semibold text-jade" : "font-semibold text-stone-500"}>
                  {String(value)}
                </span>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <EmptyState icon={<Mail className="h-5 w-5" />} text="Email draft appears after the demo run." />
      )}
    </section>
  );
}

function AuditTrail({ decision }) {
  const rows = decision?.decision?.audit_trail || [];

  return (
    <section className="rounded-md border border-stone-200 bg-white p-4 shadow-soft">
      <div className="mb-4 flex items-center gap-2">
        <Clock3 className="h-5 w-5 text-amber" />
        <h2 className="text-base font-semibold">Audit Trail</h2>
      </div>
      <div className="overflow-hidden rounded-md border border-stone-200">
        <table className="w-full table-fixed text-sm">
          <thead className="bg-stone-50 text-left text-xs uppercase text-stone-500">
            <tr>
              <th className="w-12 px-3 py-2">#</th>
              <th className="w-48 px-3 py-2">Node</th>
              <th className="w-32 px-3 py-2">Verdict</th>
              <th className="px-3 py-2">Message</th>
              <th className="w-28 px-3 py-2">Timing</th>
            </tr>
          </thead>
          <tbody>
            {rows.length ? (
              rows.map((row, index) => (
                <tr className="border-t border-stone-200" key={`${row.node}-${index}`}>
                  <td className="px-3 py-2 text-stone-500">{row.order || index + 1}</td>
                  <td className="px-3 py-2 font-medium">{row.node}</td>
                  <td className="px-3 py-2">{row.verdict}</td>
                  <td className="px-3 py-2 text-stone-600">{row.message}</td>
                  <td className="px-3 py-2 text-stone-600">{row.timing_ms} ms</td>
                </tr>
              ))
            ) : (
              <tr>
                <td className="px-3 py-4 text-stone-500" colSpan="5">No audit rows yet.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function EmptyState({ icon, text }) {
  return (
    <div className="flex min-h-32 items-center justify-center rounded-md border border-dashed border-stone-300 bg-stone-50 p-6 text-center text-sm text-stone-500">
      <div>
        <div className="mx-auto mb-2 flex h-9 w-9 items-center justify-center rounded-md bg-white text-stone-500">
          {icon}
        </div>
        {text}
      </div>
    </div>
  );
}

export default App;
