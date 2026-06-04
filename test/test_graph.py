import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch


class BuildHireGraphTests(unittest.TestCase):
    def test_build_hiregraph_runs_full_flow_with_mocked_services(self):
        from src.graph import build_hiregraph

        class StructuredLLM:
            def __init__(self, payload):
                self.payload = payload

            def invoke(self, prompt):
                return type(
                    "Result",
                    (),
                    {"model_dump": lambda _self: self.payload},
                )()

        class FakeLLM:
            def with_structured_output(self, schema, **kwargs):
                if schema.__name__ == "JDRequirements":
                    return StructuredLLM(
                        {
                            "job_title": "Junior Data Analyst",
                            "seniority": "junior",
                            "required_skills": ["SQL"],
                            "preferred_skills": [],
                            "experience_requirements": [],
                            "education_requirements": [],
                            "responsibilities": ["Analyze business data"],
                            "domain_keywords": ["data"],
                        }
                    )

                if schema.__name__ == "SkillScore":
                    return StructuredLLM(
                        {
                            "skill": "SQL",
                            "score": 8,
                            "evidence": "Resume mentions SQL.",
                            "reasoning": "Direct SQL evidence.",
                        }
                    )

                return StructuredLLM(
                    {
                        "dimension": "experience",
                        "score": 8,
                        "evidence": "Relevant project evidence.",
                        "reasoning": "Good fit for the role.",
                    }
                )

            def invoke(self, prompt):
                return type("Response", (), {"content": "Please share your availability."})()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            resume_path = tmp_path / "resume.md"
            jd_path = tmp_path / "jd.md"
            resume_path.write_text("Resume body with SQL", encoding="utf-8")
            jd_path.write_text("Junior Data Analyst role requiring SQL", encoding="utf-8")

            graph = build_hiregraph()
            with patch("src.node.llm", FakeLLM()), patch("src.node.tavily_search", return_value=[]):
                result = graph.invoke(
                    {
                        "resume_path": str(resume_path),
                        "jd_path": str(jd_path),
                    },
                    config={"configurable": {"thread_id": "test-full-flow"}},
                )

        self.assertEqual(result["raw_resume"], "Resume body with SQL")
        self.assertEqual(result["raw_jd"], "Junior Data Analyst role requiring SQL")
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["recommendation"], "advance")
        self.assertEqual(result["final_score"], 8)
        self.assertEqual(result["draft_email"], "Please share your availability.")


class MainTests(unittest.TestCase):
    def test_main_runs_with_sample_markdown_files(self):
        from main import main

        class FakeGraph:
            def invoke(self, input_value, config=None):
                return {
                    "recommendation": "advance",
                    "human_review_decision": None,
                    "final_score": 8.0,
                    "draft_email": "Please share your availability.",
                    "rejection_email": None,
                }

        output = StringIO()
        with (
            patch("main.build_hiregraph", return_value=FakeGraph()),
            patch("src.graph_print.save_graph_png"),
            patch("src.graph_print.show_graph"),
            redirect_stdout(output),
        ):
            main()

        text = output.getvalue()
        self.assertIn("========== FINAL RESULT ==========", text)
        self.assertIn("'recommendation': 'advance'", text)
        self.assertIn("'final_score': 8.0", text)


if __name__ == "__main__":
    unittest.main()
