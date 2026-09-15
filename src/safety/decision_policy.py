"""
Automation Decision Policy
==========================

Conservative AUTO/HUMAN decision layer.

Policy:
    HIGH risk   -> HUMAN
    MEDIUM risk -> HUMAN

    LOW risk + no investigation + query <= 25 words
        -> AUTO

    Everything else
        -> HUMAN

The Trust Checker remains a grounding/evidence signal and is not
used as the primary automation classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


# ============================================================================
# CONFIGURATION
# ============================================================================

POLICY_NAME = "conservative_v1"

MAX_QUERY_WORDS = 25

LOW_RISK = "LOW"
MEDIUM_RISK = "MEDIUM"
HIGH_RISK = "HIGH"

AUTO = "AUTO"
HUMAN = "HUMAN"


# ============================================================================
# RESULT
# ============================================================================

@dataclass
class DecisionResult:
    """Structured automation decision."""

    decision: str
    policy_name: str
    reason: str
    risk_level: str
    investigation_rate: float
    query_words: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision,
            "policy_name": self.policy_name,
            "reason": self.reason,
            "risk_level": self.risk_level,
            "investigation_rate": self.investigation_rate,
            "query_words": self.query_words,
        }


# ============================================================================
# HELPERS
# ============================================================================

def calculate_query_words(query: str) -> int:
    """Count whitespace-separated query words."""

    if not isinstance(query, str):
        return 0

    return len(query.strip().split())


def normalize_risk(risk_level: str) -> str:
    """Normalize risk level."""

    if not isinstance(risk_level, str):
        return MEDIUM_RISK

    return risk_level.strip().upper()


# ============================================================================
# POLICY
# ============================================================================

def decide_automation(
    risk_level: str,
    investigation_rate: float,
    query: str,
) -> DecisionResult:
    """
    Apply the conservative automation policy.

    AUTO requires ALL of:

        risk == LOW
        investigation_rate == 0
        query_words <= 25

    Otherwise HUMAN.
    """

    risk = normalize_risk(risk_level)

    try:
        investigation = float(investigation_rate)
    except (TypeError, ValueError):
        investigation = 1.0

    query_words = calculate_query_words(query)

    # HIGH risk -> HUMAN
    if risk == HIGH_RISK:
        return DecisionResult(
            decision=HUMAN,
            policy_name=POLICY_NAME,
            reason="HIGH_RISK_REQUIRES_HUMAN",
            risk_level=risk,
            investigation_rate=investigation,
            query_words=query_words,
        )

    # MEDIUM risk -> HUMAN
    if risk == MEDIUM_RISK:
        return DecisionResult(
            decision=HUMAN,
            policy_name=POLICY_NAME,
            reason="MEDIUM_RISK_REQUIRES_HUMAN",
            risk_level=risk,
            investigation_rate=investigation,
            query_words=query_words,
        )

    # Unknown risk -> fail safe
    if risk != LOW_RISK:
        return DecisionResult(
            decision=HUMAN,
            policy_name=POLICY_NAME,
            reason="UNKNOWN_RISK_FAIL_SAFE",
            risk_level=risk,
            investigation_rate=investigation,
            query_words=query_words,
        )

    # Investigation required
    if investigation != 0.0:
        return DecisionResult(
            decision=HUMAN,
            policy_name=POLICY_NAME,
            reason="INVESTIGATION_REQUIRED",
            risk_level=risk,
            investigation_rate=investigation,
            query_words=query_words,
        )

    # Query too long
    if query_words > MAX_QUERY_WORDS:
        return DecisionResult(
            decision=HUMAN,
            policy_name=POLICY_NAME,
            reason="QUERY_TOO_COMPLEX_FOR_AUTO",
            risk_level=risk,
            investigation_rate=investigation,
            query_words=query_words,
        )

    # AUTO
    return DecisionResult(
        decision=AUTO,
        policy_name=POLICY_NAME,
        reason="LOW_RISK_NO_INVESTIGATION_SHORT_QUERY",
        risk_level=risk,
        investigation_rate=investigation,
        query_words=query_words,
    )


# ============================================================================
# UNIT TESTS
# ============================================================================

def _run_tests() -> None:

    # ------------------------------------------------------------------------
    # 1. Normal LOW-risk case -> AUTO
    # ------------------------------------------------------------------------

    result = decide_automation(
        risk_level="LOW",
        investigation_rate=0,
        query="Where is my package?",
    )

    assert result.decision == AUTO

    # ------------------------------------------------------------------------
    # 2. HIGH -> HUMAN
    # ------------------------------------------------------------------------

    result = decide_automation(
        risk_level="HIGH",
        investigation_rate=0,
        query="Where is my package?",
    )

    assert result.decision == HUMAN
    assert result.reason == "HIGH_RISK_REQUIRES_HUMAN"

    # ------------------------------------------------------------------------
    # 3. MEDIUM -> HUMAN
    # ------------------------------------------------------------------------

    result = decide_automation(
        risk_level="MEDIUM",
        investigation_rate=0,
        query="Where is my package?",
    )

    assert result.decision == HUMAN
    assert result.reason == "MEDIUM_RISK_REQUIRES_HUMAN"

    # ------------------------------------------------------------------------
    # 4. Investigation required -> HUMAN
    # ------------------------------------------------------------------------

    result = decide_automation(
        risk_level="LOW",
        investigation_rate=0.5,
        query="Where is my package?",
    )

    assert result.decision == HUMAN
    assert result.reason == "INVESTIGATION_REQUIRED"

    # ------------------------------------------------------------------------
    # 5. More than 25 words -> HUMAN
    #
    # Generate exactly 26 words instead of manually counting a sentence.
    # ------------------------------------------------------------------------

    long_query = " ".join(
        f"word{i}" for i in range(MAX_QUERY_WORDS + 1)
    )

    result = decide_automation(
        risk_level="LOW",
        investigation_rate=0,
        query=long_query,
    )

    assert result.query_words == 26
    assert result.query_words > MAX_QUERY_WORDS
    assert result.decision == HUMAN
    assert result.reason == "QUERY_TOO_COMPLEX_FOR_AUTO"

    # ------------------------------------------------------------------------
    # 6. Exactly 25 words -> AUTO
    #
    # Generate exactly 25 words programmatically.
    # ------------------------------------------------------------------------

    twenty_five_words = " ".join(
        f"word{i}" for i in range(MAX_QUERY_WORDS)
    )

    result = decide_automation(
        risk_level="LOW",
        investigation_rate=0,
        query=twenty_five_words,
    )

    assert result.query_words == MAX_QUERY_WORDS
    assert result.decision == AUTO
    assert result.reason == "LOW_RISK_NO_INVESTIGATION_SHORT_QUERY"

    # ------------------------------------------------------------------------
    # 7. Unknown risk -> HUMAN
    # ------------------------------------------------------------------------

    result = decide_automation(
        risk_level="UNKNOWN",
        investigation_rate=0,
        query="Where is my package?",
    )

    assert result.decision == HUMAN
    assert result.reason == "UNKNOWN_RISK_FAIL_SAFE"

    # ------------------------------------------------------------------------
    # 8. Invalid investigation value -> HUMAN
    # ------------------------------------------------------------------------

    result = decide_automation(
        risk_level="LOW",
        investigation_rate="invalid",
        query="Where is my package?",
    )

    assert result.decision == HUMAN
    assert result.reason == "INVESTIGATION_REQUIRED"

    # ------------------------------------------------------------------------
    # 9. Case-insensitive risk -> AUTO
    # ------------------------------------------------------------------------

    result = decide_automation(
        risk_level="low",
        investigation_rate=0,
        query="Where is my package?",
    )

    assert result.decision == AUTO

    # ------------------------------------------------------------------------
    # 10. Empty query -> AUTO under current policy
    # ------------------------------------------------------------------------

    result = decide_automation(
        risk_level="LOW",
        investigation_rate=0,
        query="",
    )

    assert result.decision == AUTO
    assert result.query_words == 0

    # ------------------------------------------------------------------------
    # PASS
    # ------------------------------------------------------------------------

    print("=" * 100)
    print("AUTOMATION DECISION POLICY TEST: PASS")
    print("=" * 100)
    print("All 10 local tests passed.")
    print("No API or LLM calls were made.")


if __name__ == "__main__":
    _run_tests()