"""Explain policy optimizers that sit beyond Q-learning."""

from __future__ import annotations

import argparse
from collections.abc import Iterable

from policy_optimizer_catalog import (
    POLICY_OPTIMIZER_TECHNIQUES,
    PolicyOptimizerTechnique,
    technique_by_name,
)


def print_techniques(techniques: Iterable[PolicyOptimizerTechnique]) -> None:
    print("Policy optimizers beyond Q-learning")
    print(
        "KeyDoorGridEnv would still only provide transitions and rewards; "
        "these methods update a policy network from rollout data."
    )
    print()
    for technique in techniques:
        print(f"{technique.name}: {technique.family}")
        print(f"  learns: {technique.learns}")
        print(f"  idea: {technique.core_idea}")
        print(f"  update: {technique.mechanism}")
        print(f"  vs Q-learning: {technique.q_learning_contrast}")
        print(f"  KeyDoorGrid next step: {technique.keydoor_next_step}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print PPO/TRPO/GRPO policy-optimizer teaching notes."
    )
    parser.add_argument(
        "--technique",
        action="append",
        choices=[technique.name.lower() for technique in POLICY_OPTIMIZER_TECHNIQUES],
        help="Technique to print; repeat to show multiple. Defaults to all.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    techniques = (
        POLICY_OPTIMIZER_TECHNIQUES
        if args.technique is None
        else tuple(technique_by_name(name) for name in args.technique)
    )
    print_techniques(techniques)


if __name__ == "__main__":
    main()
