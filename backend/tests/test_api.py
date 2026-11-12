import tempfile
import time
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from o_an_quan.api import store
from o_an_quan.api.app import app

RANDOM_GAME = {"agent_a": {"type": "random"}, "agent_b": {"type": "random"}, "max_rounds": 25, "seed": 7}


class TestApi(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patch = mock.patch.object(store, "RESULTS_DIR", self.tmp.name)
        self.patch.start()
        self.client = TestClient(app)

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def wait_for(self, job_id: str) -> dict:
        for _ in range(100):
            job = self.client.get(f"/api/jobs/{job_id}").json()
            if job["status"] != "running":
                return job
            time.sleep(0.05)
        self.fail("job did not finish")

    def test_config_and_personas(self):
        config = self.client.get("/api/config").json()
        self.assertIn("BalancedAgent", config["personas"])
        personas = self.client.get("/api/personas").json()
        self.assertEqual({p["name"] for p in personas}, set(config["personas"]))

    def test_empty_results(self):
        self.assertEqual(self.client.get("/api/games").json(), [])
        stats = self.client.get("/api/stats").json()
        self.assertEqual(stats["totals"]["games"], 0)

    def test_random_game_job_is_saved_and_listed(self):
        response = self.client.post("/api/games", json=RANDOM_GAME)
        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["initial_board"]["A1"], {"peasants": 5, "mandarin": False})

        job = self.wait_for(response.json()["id"])
        self.assertEqual(job["status"], "finished", job["error"])
        self.assertTrue(job["turns"])

        game = self.client.get(f"/api/games/{job['game_id']}").json()
        self.assertEqual(len(game["turns"]), len(job["turns"]))
        self.assertIn("board", game["turns"][0])

        games = self.client.get("/api/games").json()
        self.assertEqual(games[0]["id"], job["game_id"])
        self.assertEqual(self.client.get("/api/stats").json()["totals"]["games"], 1)

    def test_errors(self):
        self.assertEqual(self.client.get("/api/games/nope-3").status_code, 404)
        self.assertEqual(self.client.get("/api/jobs/missing").status_code, 404)
        bad = {**RANDOM_GAME, "agent_a": {"type": "llm", "persona": "Nobody"}}
        with mock.patch("o_an_quan.api.app.OPENROUTER_API_KEY", "key"):
            self.assertEqual(self.client.post("/api/games", json=bad).status_code, 400)
        with mock.patch("o_an_quan.api.app.OPENROUTER_API_KEY", None):
            self.assertEqual(self.client.post("/api/games", json=bad).status_code, 400)


if __name__ == "__main__":
    unittest.main()
