import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch
from langchain_core.messages import AIMessage


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
                    {
                        "model_dump": lambda _self: self.payload,
                        "json": lambda _self: str(self.payload),
                    },
                )()

        class CritiqueLLM:
            def invoke(self, prompt):
                return type(
                    "Result",
                    (),
                    {"approved": True, "feedback": "Ready to send."},
                )()

        class ToolCallingLLM:
            def invoke(self, prompt):
                query = prompt.split("Use this query:", 1)[-1].strip()
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "tavily_search_tool",
                            "args": {"query": query},
                            "id": "call_tavily_search",
                        }
                    ],
                )

        class FakeLLM:
            def bind_tools(self, tools, **kwargs):
                return ToolCallingLLM()

            def with_structured_output(self, schema, **kwargs):
                if schema.__name__ == "ParsedResume":
                    return StructuredLLM(
                        {
                            "candidate_name": "Eitan Bergmann",
                            "headline": "Junior data analyst",
                            "skills": ["SQL"],
                            "roles": ["Analyst project"],
                            "education": [],
                            "projects": ["SQL dashboard"],
                        }
                    )

                if schema.__name__ == "NormalizedResumeSkills":
                    return StructuredLLM({"normalized_skills": ["SQL"]})

                if schema.__name__ == "YearsOfExperience":
                    return StructuredLLM(
                        {
                            "total_years": 1.0,
                            "years_by_skill": {"SQL": 1.0},
                            "reasoning": "Resume includes SQL project evidence.",
                        }
                    )

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

                if schema.__name__ == "EmailCritique":
                    return CritiqueLLM()

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
            resume_path.write_text(
                "# Eitan Bergmann\n\nEmail: eitan@example.com\n\nResume body with SQL",
                encoding="utf-8",
            )
            jd_path.write_text("Junior Data Analyst role requiring SQL", encoding="utf-8")

            graph = build_hiregraph()
            fake_llm = FakeLLM()
            with (
                patch("src.node.extract_llm", fake_llm),
                patch("src.node.score_llm", fake_llm),
                patch("src.node.email_llm", fake_llm),
                patch("src.node.critic_llm", fake_llm),
                patch("src.tools.tavily_search", return_value=[]),
                patch("src.node.perform_downstream_actions"),
            ):
                result = graph.invoke(
                    {
                        "resume_path": str(resume_path),
                        "jd_path": str(jd_path),
                    },
                    config={"configurable": {"thread_id": "test-full-flow"}},
                )

        self.assertIn("Resume body with SQL", result["raw_resume"])
        self.assertEqual(result["raw_jd"], "Junior Data Analyst role requiring SQL")
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["candidate_name"], "Eitan Bergmann")
        self.assertEqual(result["candidate_email"], "eitan@example.com")
        self.assertEqual(result["parsed_resume"]["candidate_name"], "Eitan Bergmann")
        self.assertEqual(result["normalized_skills"], ["SQL"])
        self.assertEqual(result["years_of_experience"]["total_years"], 1.0)
        self.assertEqual(result["classification"]["seniority"], "junior")
        self.assertEqual(
            result["classification"]["scoring_profile"],
            "junior_scoring_profile",
        )
        self.assertEqual(result["recommendation"], "advance")
        self.assertEqual(result["final_score"], 8)
        self.assertEqual(result["scorecard"]["final_score"], 8)
        self.assertEqual(result["scorecard"]["classification"]["seniority"], "junior")
        self.assertEqual(result["sender_email"], "Hiring_Team")
        self.assertEqual(result["draft_email"], "Please share your availability.")
        self.assertTrue(result["email_approved_by_critic"])
        self.assertTrue(result["email_sent"])
        self.assertTrue(result["ats_updated"])
        self.assertEqual(result["audit_trail"][-1]["node"], "finalize")

    def test_research_tool_error_loops_back_to_llm_query_repair(self):
        from src.graph import build_hiregraph

        class StructuredLLM:
            def __init__(self, payload):
                self.payload = payload

            def invoke(self, prompt):
                return type(
                    "Result",
                    (),
                    {
                        "model_dump": lambda _self: self.payload,
                        "json": lambda _self: str(self.payload),
                    },
                )()

        class CritiqueLLM:
            def invoke(self, prompt):
                return type(
                    "Result",
                    (),
                    {"approved": True, "feedback": "Ready to send."},
                )()

        class ToolCallingLLM:
            def invoke(self, prompt):
                query = prompt.split("Use this query:", 1)[-1].strip()
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "tavily_search_tool",
                            "args": {"query": query},
                            "id": "call_tavily_search",
                        }
                    ],
                )

        class FakeLLM:
            def bind_tools(self, tools, **kwargs):
                return ToolCallingLLM()

            def with_structured_output(self, schema, **kwargs):
                if schema.__name__ == "ParsedResume":
                    return StructuredLLM(
                        {
                            "candidate_name": "Eitan Bergmann",
                            "headline": "Junior data analyst",
                            "skills": ["SQL"],
                            "roles": ["Analyst project"],
                            "education": [],
                            "projects": ["SQL dashboard"],
                        }
                    )

                if schema.__name__ == "NormalizedResumeSkills":
                    return StructuredLLM({"normalized_skills": ["SQL"]})

                if schema.__name__ == "YearsOfExperience":
                    return StructuredLLM(
                        {
                            "total_years": 1.0,
                            "years_by_skill": {"SQL": 1.0},
                            "reasoning": "Resume includes SQL project evidence.",
                        }
                    )

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

                if schema.__name__ == "EmailCritique":
                    return CritiqueLLM()

                return StructuredLLM(
                    {
                        "dimension": "research",
                        "score": 8,
                        "evidence": "Recovered search evidence.",
                        "reasoning": "Recovered after query repair.",
                    }
                )

            def invoke(self, prompt):
                return type("Response", (), {"content": "repaired github projects query"})()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            resume_path = tmp_path / "resume.md"
            jd_path = tmp_path / "jd.md"
            resume_path.write_text(
                "# Eitan Bergmann\n\nEmail: eitan@example.com\n\nResume body with SQL",
                encoding="utf-8",
            )
            jd_path.write_text("Junior Data Analyst role requiring SQL", encoding="utf-8")

            graph = build_hiregraph()
            fake_llm = FakeLLM()
            with (
                patch("src.node.extract_llm", fake_llm),
                patch("src.node.score_llm", fake_llm),
                patch("src.node.email_llm", fake_llm),
                patch("src.node.critic_llm", fake_llm),
                patch(
                    "src.tools.tavily_search",
                    side_effect=[
                        ValueError("Malformed search query"),
                        [{"title": "Recovered", "url": "https://example.com", "content": "Evidence"}],
                    ],
                ),
            ):
                result = graph.invoke(
                    {
                        "resume_path": str(resume_path),
                        "jd_path": str(jd_path),
                    },
                    config={"configurable": {"thread_id": "test-research-repair"}},
                )

        self.assertEqual(result["tool_error"], "Malformed search query")
        self.assertEqual(result["tool_retry_count"], 1)
        self.assertEqual(result["research_query"], "repaired github projects query")
        self.assertEqual(result["research_results"][0]["query"], "repaired github projects query")

    def test_rejection_path_runs_terminal_nodes_once_after_parallel_scoring(self):
        from src.graph import build_hiregraph

        class StructuredLLM:
            def __init__(self, payload):
                self.payload = payload

            def invoke(self, prompt):
                return type(
                    "Result",
                    (),
                    {
                        "model_dump": lambda _self: self.payload,
                        "json": lambda _self: str(self.payload),
                    },
                )()

        class ToolCallingLLM:
            def invoke(self, prompt):
                query = prompt.split("Use this query:", 1)[-1].strip()
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "tavily_search_tool",
                            "args": {"query": query},
                            "id": "call_tavily_search",
                        }
                    ],
                )

        class FakeLLM:
            def bind_tools(self, tools, **kwargs):
                return ToolCallingLLM()

            def with_structured_output(self, schema, **kwargs):
                if schema.__name__ == "ParsedResume":
                    return StructuredLLM(
                        {
                            "candidate_name": "Mira Volkov",
                            "headline": "Frontend developer",
                            "skills": ["JavaScript"],
                            "roles": ["Frontend project"],
                            "education": [],
                            "projects": ["Frontend app"],
                        }
                    )

                if schema.__name__ == "NormalizedResumeSkills":
                    return StructuredLLM({"normalized_skills": ["JavaScript"]})

                if schema.__name__ == "YearsOfExperience":
                    return StructuredLLM(
                        {
                            "total_years": 1.0,
                            "years_by_skill": {"JavaScript": 1.0},
                            "reasoning": "Resume includes frontend project evidence.",
                        }
                    )

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
                            "score": 1,
                            "evidence": "No SQL evidence.",
                            "reasoning": "Weak match.",
                        }
                    )

                return StructuredLLM(
                    {
                        "dimension": "experience",
                        "score": 1,
                        "evidence": "No relevant evidence.",
                        "reasoning": "Weak match.",
                    }
                )

            def invoke(self, prompt):
                return type("Response", (), {"content": "Thank you for applying."})()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            resume_path = tmp_path / "resume.md"
            jd_path = tmp_path / "jd.md"
            resume_path.write_text(
                "# Mira Volkov\n\nEmail: mira@example.com\n\nFrontend resume",
                encoding="utf-8",
            )
            jd_path.write_text("Junior Data Analyst role requiring SQL", encoding="utf-8")

            graph = build_hiregraph()
            fake_llm = FakeLLM()
            with (
                patch("src.node.extract_llm", fake_llm),
                patch("src.node.score_llm", fake_llm),
                patch("src.node.email_llm", fake_llm),
                patch("src.node.critic_llm", fake_llm),
                patch("src.tools.tavily_search", return_value=[]),
            ):
                result = graph.invoke(
                    {
                        "resume_path": str(resume_path),
                        "jd_path": str(jd_path),
                    },
                    config={"configurable": {"thread_id": "test-reject-once"}},
                )

        audit_nodes = [item["node"] for item in result["audit_trail"]]
        self.assertEqual(result["recommendation"], "reject")
        self.assertEqual(audit_nodes.count("draft_rejection"), 1)
        self.assertEqual(audit_nodes.count("log_rejection"), 1)
        self.assertEqual(audit_nodes.count("finalize"), 1)


class MainTests(unittest.TestCase):
    def test_main_runs_with_sample_markdown_files(self):
        from src.main import main

        class FakeGraph:
            def invoke(self, input_value, config=None):
                return {
                    "recommendation": "advance",
                    "human_review_decision": None,
                    "final_score": 8.0,
                    "candidate_email": "eitan@example.com",
                    "sender_email": "Hiring_Team",
                    "draft_email": "Please share your availability.",
                    "rejection_email": None,
                    "email_sent": True,
                    "ats_updated": True,
                    "rejection_logged": None,
                    "compensation_done": None,
                    "audit_trail": [],
                }

        output = StringIO()
        with (
            patch("src.main.build_hiregraph", return_value=FakeGraph()),
            patch("src.graph_print.save_graph_png"),
            patch("src.graph_print.show_graph"),
            redirect_stdout(output),
        ):
            main()

        text = output.getvalue()
        self.assertIn("========== FINAL RESULT ==========", text)
        self.assertIn("'recommendation': 'advance'", text)
        self.assertIn("'final_score': 8.0", text)

    def test_main_uses_uuid_thread_id(self):
        from uuid import UUID

        from src.main import main

        class FakeGraph:
            def __init__(self):
                self.config = None

            def invoke(self, input_value, config=None):
                self.config = config
                return {
                    "recommendation": "advance",
                    "human_review_decision": None,
                    "final_score": 8.0,
                    "candidate_email": "eitan@example.com",
                    "sender_email": "Hiring_Team",
                    "draft_email": "Please share your availability.",
                    "rejection_email": None,
                    "email_sent": True,
                    "ats_updated": True,
                    "rejection_logged": None,
                    "compensation_done": None,
                    "audit_trail": [],
                }

        fake_graph = FakeGraph()
        fixed_uuid = UUID("12345678-1234-5678-1234-567812345678")

        output = StringIO()
        with (
            patch("src.main.uuid4", return_value=fixed_uuid),
            patch("src.main.build_hiregraph", return_value=fake_graph),
            patch("src.graph_print.save_graph_png"),
            patch("src.graph_print.show_graph"),
            redirect_stdout(output),
        ):
            main()

        self.assertEqual(
            fake_graph.config["configurable"]["thread_id"],
            "12345678-1234-5678-1234-567812345678",
        )


if __name__ == "__main__":
    unittest.main()
