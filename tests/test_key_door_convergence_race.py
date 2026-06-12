from __future__ import annotations

import unittest
from types import SimpleNamespace

from examples.key_door_grid_convergence_race import (
    RaceRunResult,
    estimate_seconds_to_stable,
    format_seconds,
    format_solve_episode,
    run_convergence_race,
    stable_solve_episode,
    summarize_race,
)


class KeyDoorConvergenceRaceTest(unittest.TestCase):
    def test_stable_solve_episode_requires_all_later_evaluations_to_solve(self) -> None:
        evaluations = [
            SimpleNamespace(episode=10, solved=False),
            SimpleNamespace(episode=20, solved=True),
            SimpleNamespace(episode=30, solved=False),
            SimpleNamespace(episode=40, solved=True),
            SimpleNamespace(episode=50, solved=True),
        ]

        self.assertEqual(stable_solve_episode(evaluations), 40)

    def test_summarize_race_orders_by_estimated_wall_clock(self) -> None:
        summaries = summarize_race(
            {
                "sample_efficient_but_expensive": [
                    RaceRunResult(
                        "sample_efficient_but_expensive",
                        1,
                        20,
                        True,
                        1.0,
                        9,
                        10.0,
                        2.0,
                    ),
                    RaceRunResult(
                        "sample_efficient_but_expensive",
                        2,
                        30,
                        True,
                        1.0,
                        9,
                        10.0,
                        3.0,
                    ),
                ],
                "wall_clock_fast": [
                    RaceRunResult(
                        "wall_clock_fast",
                        1,
                        40,
                        True,
                        1.1,
                        9,
                        1.0,
                        0.4,
                    ),
                    RaceRunResult(
                        "wall_clock_fast",
                        2,
                        60,
                        True,
                        1.1,
                        9,
                        1.0,
                        0.6,
                    ),
                ],
                "unstable": [
                    RaceRunResult("unstable", 1, None, False, -0.1, 45),
                    RaceRunResult("unstable", 2, None, False, -0.1, 45),
                ],
            }
        )

        self.assertEqual(
            [summary.learner for summary in summaries],
            ["wall_clock_fast", "sample_efficient_but_expensive", "unstable"],
        )
        self.assertEqual(summaries[0].median_stable_solve_episode, 50)
        self.assertEqual(summaries[0].median_estimated_seconds_to_stable, 0.5)
        self.assertEqual(summaries[-1].median_stable_solve_episode, None)

    def test_runs_tiny_smoke_race(self) -> None:
        results = run_convergence_race(
            learners=("q_learning",),
            seeds=(1,),
            layout="classic",
            episodes=1,
            eval_every=1,
        )

        self.assertEqual(set(results), {"q_learning"})
        self.assertEqual(len(results["q_learning"]), 1)
        self.assertEqual(results["q_learning"][0].learner, "q_learning")

    def test_rejects_invalid_race_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown learners"):
            run_convergence_race(learners=("ppo",))
        with self.assertRaisesRegex(ValueError, "seeds must contain"):
            run_convergence_race(seeds=())
        with self.assertRaisesRegex(ValueError, "episodes must be"):
            run_convergence_race(episodes=0)
        self.assertEqual(
            estimate_seconds_to_stable(
                wall_clock_seconds=2.0,
                stable_episode=50,
                total_episodes=100,
            ),
            1.0,
        )
        self.assertEqual(
            estimate_seconds_to_stable(
                wall_clock_seconds=2.0,
                stable_episode=None,
                total_episodes=100,
            ),
            None,
        )
        self.assertEqual(format_solve_episode(None), "not stable")
        self.assertEqual(format_solve_episode(20), "20")
        self.assertEqual(format_seconds(None), "-")
        self.assertEqual(format_seconds(1.2345), "1.234")


if __name__ == "__main__":
    unittest.main()
