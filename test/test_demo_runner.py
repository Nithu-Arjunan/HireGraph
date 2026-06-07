import unittest


class DecisionObjectTests(unittest.TestCase):
    def test_build_decision_object_formats_scorecard_email_and_audit(self):
        from src.demo_runner import build_decision_object

        result = build_decision_object(
            "strong",
            {
                "candidate_name": "Priya Ramanathan",
                "candidate_email": "priya@example.com",
                "final_score": 8.7,
                "score_summary": {"skill_average": 9, "dimension_average": 8.4},
                "skill_evaluations": [
                    {"skill": "SQL", "score": 9, "evidence": "Strong SQL."}
                ],
                "dimension_evaluations": [
                    {
                        "dimension": "experience",
                        "score": 8,
                        "evidence": "Relevant analytics work.",
                        "reasoning": "Good fit.",
                    }
                ],
                "recommendation": "advance",
                "draft_email": "Please share your availability.",
                "audit_trail": [
                    {
                        "node": "ingest_resume_jd",
                        "status": "completed",
                        "message": "Loaded resume and JD.",
                    }
                ],
            },
            elapsed_ms=1234,
        )

        self.assertEqual(result["scenario"], "strong")
        self.assertEqual(result["candidate_name"], "Priya Ramanathan")
        self.assertEqual(result["decision"]["recommendation"], "advance")
        self.assertEqual(result["run_status"], "completed")
        self.assertEqual(result["decision"]["scorecard"]["final_score"], 8.7)
        self.assertEqual(result["decision"]["scorecard"]["skill_scores"][0]["name"], "SQL")
        self.assertEqual(result["decision"]["scorecard"]["dimension_scores"][0]["name"], "experience")
        self.assertEqual(result["decision"]["email_draft"], "Please share your availability.")
        self.assertEqual(result["decision"]["audit_trail"][0]["verdict"], "completed")
        self.assertEqual(result["decision"]["audit_trail"][0]["timing_ms"], 1234)

    def test_build_scoreboard_summary_counts_recommendations(self):
        from src.demo_runner import build_scoreboard_summary

        summary = build_scoreboard_summary(
            [
                {"decision": {"recommendation": "advance", "scorecard": {"final_score": 8}}},
                {"decision": {"recommendation": "reject", "scorecard": {"final_score": 3}}},
                {"decision": {"recommendation": "borderline", "scorecard": {"final_score": 5}}},
            ]
        )

        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["advance"], 1)
        self.assertEqual(summary["reject"], 1)
        self.assertEqual(summary["borderline"], 1)
        self.assertEqual(summary["average_score"], 5.33)

    def test_build_decision_object_marks_interrupted_run(self):
        from src.demo_runner import build_decision_object

        result = build_decision_object(
            "borderline",
            {
                "candidate_name": "Eitan Bergmann",
                "candidate_email": "eitan@example.com",
                "recommendation": "borderline",
                "__interrupt__": [{"value": {"message": "Review required"}}],
            },
            elapsed_ms=500,
            label="Borderline Candidate",
            run_status="interrupted",
        )

        self.assertEqual(result["run_status"], "interrupted")
        self.assertEqual(result["interrupt"]["value"]["message"], "Review required")
        self.assertEqual(result["decision"]["recommendation"], "borderline")


if __name__ == "__main__":
    unittest.main()
