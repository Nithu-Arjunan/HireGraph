import unittest
from unittest.mock import patch


class DemoServerTests(unittest.TestCase):
    def test_fastapi_health_endpoint(self):
        from fastapi.testclient import TestClient

        from src.demo_server import app

        client = TestClient(app)
        response = client.get("/api/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_fastapi_demo_run_endpoint_returns_runner_payload(self):
        from fastapi.testclient import TestClient

        from src.demo_server import app

        payload = {
            "scoreboard": {"total": 3},
            "decisions": [],
            "graph_png": "graph_out/hiregraph.png",
        }

        with patch("src.demo_server.run_hiregraph_scenarios", return_value=payload):
            client = TestClient(app)
            response = client.get("/api/demo/run")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), payload)

    def test_fastapi_resume_endpoint_returns_updated_payload(self):
        from fastapi.testclient import TestClient

        from src.demo_server import app

        payload = {
            "scoreboard": {"total": 3},
            "decisions": [{"scenario": "borderline", "run_status": "completed"}],
        }

        with patch("src.demo_server.resume_hiregraph_scenario", return_value=payload):
            client = TestClient(app)
            response = client.post(
                "/api/demo/resume",
                json={
                    "scenario": "borderline",
                    "decision": "approved",
                    "notes": "Looks good.",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), payload)


if __name__ == "__main__":
    unittest.main()
