import unittest
from unittest.mock import patch


class ExtractJDRequirementsTests(unittest.TestCase):
    def test_uses_function_calling_for_gpt4_structured_output(self):
        from src.node import extract_jd_requirements

        class StructuredLLM:
            def invoke(self, prompt):
                return type(
                    "Result",
                    (),
                    {"model_dump": lambda self: {"job_title": "Data Analyst"}},
                )()

        class RecordingLLM:
            def __init__(self):
                self.method = None

            def with_structured_output(self, schema, *, method):
                self.method = method
                return StructuredLLM()

        llm = RecordingLLM()

        with patch("src.node.extract_llm", llm):
            extract_jd_requirements({"raw_jd": "Analyze business data."})

        self.assertEqual(llm.method, "function_calling")


class LLMConfigurationTests(unittest.TestCase):
    def test_retries_transient_rate_limits(self):
        from src.node import critic_llm, email_llm, extract_llm, score_llm

        for llm in (extract_llm, score_llm, email_llm, critic_llm):
            self.assertGreaterEqual(llm.max_retries, 5)


class IngestResumeJDTests(unittest.TestCase):
    def test_missing_resume_path_records_error_without_candidate_extraction_crash(self):
        from src.node import ingest_resume_jd

        result = ingest_resume_jd({"resume_path": "missing.md"})

        self.assertIn("File not found: missing.md", result["errors"])
        self.assertEqual(result["candidate_name"], "candidate")
        self.assertIsNone(result["candidate_email"])


class ExperienceScorerTests(unittest.TestCase):
    def test_uses_function_calling_for_gpt4_structured_output(self):
        from src.node import experience_scorer

        class StructuredLLM:
            def invoke(self, prompt):
                return type(
                    "Result",
                    (),
                    {
                        "model_dump": lambda self: {
                            "dimension": "experience",
                            "score": 8,
                            "evidence": "Relevant experience.",
                            "reasoning": "Direct evidence.",
                        }
                    },
                )()

        class RecordingLLM:
            def __init__(self):
                self.method = None

            def with_structured_output(self, schema, *, method):
                self.method = method
                return StructuredLLM()

        llm = RecordingLLM()

        with patch("src.node.score_llm", llm):
            experience_scorer(
                {
                    "resume_text": "SQL analyst",
                    "jd_requirements": {
                        "experience_requirements": ["SQL"],
                        "responsibilities": ["Analyze data"],
                    },
                }
            )

        self.assertEqual(llm.method, "function_calling")


class StartParallelScoringTests(unittest.TestCase):
    def test_passes_raw_resume_text_to_each_skill_worker(self):
        from src.node import start_parallel_scoring

        packets = start_parallel_scoring(
            {
                "raw_resume": "Resume body",
                "jd_requirements": {"required_skills": ["SQL"]},
            }
        )

        self.assertEqual(
            packets[0].arg,
            {"resume_text": "Resume body", "skill": "SQL"},
        )


class SkillWorkerTests(unittest.TestCase):
    def test_uses_function_calling_for_gpt4_structured_output(self):
        from src.node import skill_worker

        class StructuredLLM:
            def invoke(self, prompt):
                return type(
                    "Result",
                    (),
                    {"model_dump": lambda self: {"skill": "SQL", "score": 10}},
                )()

        class RecordingLLM:
            def __init__(self):
                self.method = None

            def with_structured_output(self, schema, *, method):
                self.method = method
                return StructuredLLM()

        llm = RecordingLLM()

        with patch("src.node.score_llm", llm):
            skill_worker({"resume_text": "SQL", "skill": "SQL"})

        self.assertEqual(llm.method, "function_calling")


if __name__ == "__main__":
    unittest.main()
