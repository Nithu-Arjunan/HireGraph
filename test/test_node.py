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

    def test_initializes_action_status_flags(self):
        from src.node import ingest_resume_jd

        result = ingest_resume_jd({})

        self.assertEqual(result["email_sent"], False)
        self.assertEqual(result["ats_updated"], False)
        self.assertEqual(result["rejection_logged"], False)
        self.assertEqual(result["compensation_done"], False)


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


class ResumePromptChainTests(unittest.TestCase):
    def test_parse_resume_stores_structured_resume_data(self):
        from src.node import parse_resume

        class StructuredLLM:
            def invoke(self, prompt):
                return type(
                    "Result",
                    (),
                    {
                        "model_dump": lambda self: {
                            "candidate_name": "Eitan Bergmann",
                            "headline": "Junior data engineer",
                            "skills": ["SQL", "Python"],
                            "roles": ["Data intern"],
                            "education": ["BS Data Science"],
                            "projects": ["Pipeline project"],
                        }
                    },
                )()

        class RecordingLLM:
            def __init__(self):
                self.schema = None
                self.method = None

            def with_structured_output(self, schema, *, method):
                self.schema = schema
                self.method = method
                return StructuredLLM()

        llm = RecordingLLM()

        with patch("src.node.extract_llm", llm):
            result = parse_resume({"raw_resume": "SQL and Python resume"})

        self.assertEqual(llm.schema.__name__, "ParsedResume")
        self.assertEqual(llm.method, "function_calling")
        self.assertEqual(result["parsed_resume"]["candidate_name"], "Eitan Bergmann")

    def test_normalize_resume_skills_stores_normalized_skill_list(self):
        from src.node import normalize_resume_skills

        class StructuredLLM:
            def invoke(self, prompt):
                return type(
                    "Result",
                    (),
                    {
                        "model_dump": lambda self: {
                            "normalized_skills": ["Python", "SQL", "ETL"]
                        }
                    },
                )()

        class RecordingLLM:
            def with_structured_output(self, schema, *, method):
                self.schema = schema
                self.method = method
                return StructuredLLM()

        llm = RecordingLLM()

        with patch("src.node.extract_llm", llm):
            result = normalize_resume_skills(
                {"parsed_resume": {"skills": ["python", "sql", "etl"]}}
            )

        self.assertEqual(llm.schema.__name__, "NormalizedResumeSkills")
        self.assertEqual(result["normalized_skills"], ["Python", "SQL", "ETL"])

    def test_extract_years_experience_stores_years_by_area(self):
        from src.node import extract_years_experience

        class StructuredLLM:
            def invoke(self, prompt):
                return type(
                    "Result",
                    (),
                    {
                        "model_dump": lambda self: {
                            "total_years": 1.5,
                            "years_by_skill": {"SQL": 1.5},
                            "reasoning": "One internship and one project.",
                        }
                    },
                )()

        class RecordingLLM:
            def with_structured_output(self, schema, *, method):
                self.schema = schema
                self.method = method
                return StructuredLLM()

        llm = RecordingLLM()

        with patch("src.node.extract_llm", llm):
            result = extract_years_experience(
                {
                    "parsed_resume": {"roles": ["Data intern"]},
                    "normalized_skills": ["SQL"],
                }
            )

        self.assertEqual(llm.schema.__name__, "YearsOfExperience")
        self.assertEqual(result["years_of_experience"]["total_years"], 1.5)


class SeniorityRoutingTests(unittest.TestCase):
    def test_seniority_router_routes_to_matching_profile_node(self):
        from src.node import seniority_router

        result = seniority_router(
            {"jd_requirements": {"seniority": "senior"}}
        )

        self.assertEqual(result.goto, "senior_scoring_profile")
        self.assertEqual(result.update["classification"]["seniority"], "senior")
        self.assertEqual(
            result.update["classification"]["scoring_profile"],
            "senior_scoring_profile",
        )
        self.assertEqual(
            result.update["classification"]["scoring_weights"]["experience"],
            1.3,
        )

    def test_unknown_seniority_routes_to_mid_profile_node(self):
        from src.node import seniority_router

        result = seniority_router(
            {"jd_requirements": {"seniority": "unknown"}}
        )

        self.assertEqual(result.goto, "mid_scoring_profile")
        self.assertEqual(result.update["classification"]["seniority"], "unknown")
        self.assertEqual(
            result.update["classification"]["scoring_profile"],
            "mid_scoring_profile",
        )

    def test_seniority_profile_routes_to_parallel_scoring(self):
        from src.node import senior_scoring_profile

        result = senior_scoring_profile(
            {"jd_requirements": {"seniority": "senior"}}
        )

        self.assertEqual(result.goto, "start_parallel_scoring")
        self.assertEqual(
            result.update["classification"]["scoring_profile"],
            "senior_scoring_profile",
        )


class AggregateScoresTests(unittest.TestCase):
    def test_uses_seniority_weights_and_builds_scorecard(self):
        from src.node import aggregate_scores

        result = aggregate_scores(
            {
                "classification": {
                    "seniority": "junior",
                    "scoring_weights": {
                        "skill": 1.0,
                        "experience": 0.7,
                        "education": 1.0,
                        "signal": 1.2,
                        "research": 1.1,
                    },
                },
                "skill_evaluations": [
                    {"skill": "SQL", "score": 8},
                ],
                "dimension_evaluations": [
                    {"dimension": "experience", "score": 10},
                    {"dimension": "signal", "score": 4},
                ],
            }
        )

        self.assertEqual(result["final_score"], 6.83)
        self.assertEqual(result["scorecard"]["classification"]["seniority"], "junior")
        self.assertEqual(result["scorecard"]["weighted_scores"][1]["weight"], 0.7)
        self.assertEqual(result["score_summary"]["weighted_average"], 6.83)


class SendEmailUpdateATSTests(unittest.TestCase):
    def test_perform_downstream_actions_sends_sandbox_email(self):
        from src.node import perform_downstream_actions

        sent_messages = []

        class FakeSMTP:
            def __init__(self, host, port, timeout):
                self.host = host
                self.port = port
                self.timeout = timeout

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def starttls(self):
                self.started_tls = True

            def login(self, username, password):
                self.username = username
                self.password = password

            def send_message(self, message):
                sent_messages.append(message)

        with (
            patch("src.node.SMTP_HOST", "sandbox.smtp.test"),
            patch("src.node.SMTP_PORT", 2525),
            patch("src.node.SMTP_USERNAME", "sandbox-user"),
            patch("src.node.SMTP_PASSWORD", "sandbox-password"),
            patch("src.node.SMTP_FROM_EMAIL", "hiring@example.test"),
            patch("src.node.smtplib.SMTP", FakeSMTP),
        ):
            perform_downstream_actions(
                {
                    "candidate_email": "eitan@example.com",
                    "candidate_name": "Eitan Bergmann",
                    "draft_email": "Please share your availability.",
                    "jd_requirements": {"job_title": "Data Analyst"},
                }
            )

        self.assertEqual(len(sent_messages), 1)
        self.assertEqual(sent_messages[0]["To"], "eitan@example.com")
        self.assertEqual(sent_messages[0]["From"], "hiring@example.test")
        self.assertEqual(sent_messages[0]["Subject"], "Interview invitation for Data Analyst")
        self.assertIn("Please share your availability.", sent_messages[0].get_content())

    def test_routes_to_compensation_after_downstream_retries_exhausted(self):
        from src.node import send_email_update_ats

        with patch(
            "src.node.perform_downstream_actions",
            side_effect=ConnectionError("SMTP unavailable"),
        ):
            result = send_email_update_ats(
                {
                    "candidate_email": "eitan@example.com",
                    "candidate_name": "Eitan Bergmann",
                }
            )

        self.assertEqual(result.goto, "compensate")
        self.assertEqual(result.update["email_sent"], False)
        self.assertEqual(result.update["ats_updated"], False)
        self.assertEqual(result.update["downstream_attempts"], 3)
        self.assertIn("SMTP unavailable", result.update["downstream_error"])


if __name__ == "__main__":
    unittest.main()
