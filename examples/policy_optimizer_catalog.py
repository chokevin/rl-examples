"""Small catalog of policy-optimization techniques beyond Q-learning."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyOptimizerTechnique:
    name: str
    family: str
    learns: str
    core_idea: str
    mechanism: str
    q_learning_contrast: str
    keydoor_next_step: str


POLICY_OPTIMIZER_TECHNIQUES: tuple[PolicyOptimizerTechnique, ...] = (
    PolicyOptimizerTechnique(
        name="PPO",
        family="clipped policy optimization",
        learns="a stochastic policy, usually with a value-function baseline",
        core_idea=(
            "Improve the policy with gradient steps, but clip the probability "
            "ratio so each update stays near the policy that collected the data."
        ),
        mechanism="Adam or SGD on a clipped surrogate policy-gradient loss.",
        q_learning_contrast=(
            "Q-learning backs up action values; PPO directly adjusts action "
            "probabilities."
        ),
        keydoor_next_step=(
            "A small KeyDoorGrid PPO example would add an actor plus critic and "
            "train on batches of rollouts."
        ),
    ),
    PolicyOptimizerTechnique(
        name="TRPO",
        family="trust-region policy optimization",
        learns="a stochastic policy with a constrained policy update",
        core_idea=(
            "Maximize a policy-gradient surrogate while constraining KL "
            "divergence from the old policy."
        ),
        mechanism=(
            "Natural-gradient style update, commonly via conjugate gradient and "
            "line search rather than a plain Adam step."
        ),
        q_learning_contrast=(
            "Q-learning has no trust-region constraint because it updates value "
            "estimates, not a policy distribution."
        ),
        keydoor_next_step=(
            "A faithful KeyDoorGrid TRPO example would need KL calculations, "
            "conjugate gradient, and line search, so it is larger than PPO."
        ),
    ),
    PolicyOptimizerTechnique(
        name="GRPO",
        family="group-relative policy optimization",
        learns="a stochastic policy from grouped rollouts and relative rewards",
        core_idea=(
            "Score multiple sampled answers or rollouts for the same prompt/task "
            "and update from each sample's reward relative to its group."
        ),
        mechanism=(
            "PPO-like policy-ratio objective using group-normalized advantages; "
            "often used to avoid a separate critic in LLM RL setups."
        ),
        q_learning_contrast=(
            "Q-learning learns state-action values from individual transitions; "
            "GRPO learns a policy from grouped trajectory-level comparisons."
        ),
        keydoor_next_step=(
            "A KeyDoorGrid GRPO teaching version would group several rollouts "
            "from the same start state and compare their returns."
        ),
    ),
)


def technique_by_name(name: str) -> PolicyOptimizerTechnique:
    normalized = name.casefold()
    for technique in POLICY_OPTIMIZER_TECHNIQUES:
        if technique.name.casefold() == normalized:
            return technique
    valid = ", ".join(technique.name for technique in POLICY_OPTIMIZER_TECHNIQUES)
    raise ValueError(f"name must be one of: {valid}")


__all__ = [
    "POLICY_OPTIMIZER_TECHNIQUES",
    "PolicyOptimizerTechnique",
    "technique_by_name",
]
