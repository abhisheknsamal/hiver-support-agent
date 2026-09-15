"""
Grounding & Safety Checker
==========================

Deterministic safety and grounding validation for generated
customer-support responses.

Decision levels:
    GROUNDED
    CONDITIONAL
    UNSAFE

The checker detects:
- Internal case identifiers
- Internal retrieval/system information
- Potential sensitive information
- Unsupported resolution claims
- Unsupported customer-specific claims
- Missing follow-up when evidence requires investigation

No LLM or API calls are made by this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


# ============================================================================
# DECISIONS
# ============================================================================

GROUNDED = "GROUNDED"
CONDITIONAL = "CONDITIONAL"
UNSAFE = "UNSAFE"


# ============================================================================
# INTERNAL INFORMATION
# ============================================================================

INTERNAL_CASE_ID_PATTERN = re.compile(
    r"\bAMZ-\d+\b",
    re.IGNORECASE,
)


INTERNAL_SYSTEM_TERMS = [
    "cross encoder",
    "relevance score",
    "rrf score",
    "bm25",
    "bge score",
    "historical evidence",
    "retrieved case",
    "model score",
    "vector search",
    "embedding score",
    "reranker score",
    "retrieval score",
]


# ============================================================================
# SENSITIVE INFORMATION
# ============================================================================

SENSITIVE_PATTERNS = {
    "URL": re.compile(
        r"(?:https?://|www\.)\S+",
        re.IGNORECASE,
    ),
    "SOCIAL_HANDLE": re.compile(
        r"(?<!\w)@\w{2,}",
    ),
    "LONG_NUMERIC_ID": re.compile(
        r"(?<!\d)\d{10,16}(?!\d)",
    ),
    "PHONE_NUMBER": re.compile(
        r"(?<!\d)(?:\+?\d[\d\s().-]{8,}\d)(?!\d)"
    ),
}


# ============================================================================
# UNSUPPORTED RESOLUTION CLAIMS
# ============================================================================

UNSUPPORTED_RESOLUTION_PATTERNS = [

    # ------------------------------------------------------------------------
    # Payment
    # ------------------------------------------------------------------------

    re.compile(
        r"\bpayment\s+(?:was\s+|has\s+been\s+)?processed\b",
        re.IGNORECASE,
    ),

    re.compile(
        r"\bpayment\s+(?:was\s+|has\s+been\s+)?successful\b",
        re.IGNORECASE,
    ),

    re.compile(
        r"\bpayment\s+(?:was\s+|has\s+been\s+)?completed\b",
        re.IGNORECASE,
    ),

    # ------------------------------------------------------------------------
    # Refund
    # ------------------------------------------------------------------------

    re.compile(
        r"\brefund\s+(?:was\s+|has\s+been\s+)?processed\b",
        re.IGNORECASE,
    ),

    re.compile(
        r"\brefund\s+(?:was\s+|has\s+been\s+)?issued\b",
        re.IGNORECASE,
    ),

    re.compile(
        r"\brefund\s+(?:was\s+|has\s+been\s+)?completed\b",
        re.IGNORECASE,
    ),

    # ------------------------------------------------------------------------
    # Issue resolution
    # ------------------------------------------------------------------------

    re.compile(
        r"\bissue\s+(?:has\s+been\s+|was\s+)?resolved\b",
        re.IGNORECASE,
    ),

    re.compile(
        r"\bissue\s+(?:has\s+been\s+|was\s+)?fixed\b",
        re.IGNORECASE,
    ),

    # ------------------------------------------------------------------------
    # Order cancellation
    # ------------------------------------------------------------------------

    re.compile(
        r"\border\s+(?:has\s+been\s+|was\s+)?cancelled\b",
        re.IGNORECASE,
    ),

    re.compile(
        r"\border\s+(?:has\s+been\s+|was\s+)?canceled\b",
        re.IGNORECASE,
    ),

    # ------------------------------------------------------------------------
    # Investigation completion
    # ------------------------------------------------------------------------

    # "The investigation has been completed."
    re.compile(
        r"\binvestigation\s+(?:has|was)\s+been\s+completed\b",
        re.IGNORECASE,
    ),

    # "The investigation is complete."
    re.compile(
        r"\binvestigation\s+is\s+complete\b",
        re.IGNORECASE,
    ),

    # "The investigation was completed."
    re.compile(
        r"\binvestigation\s+was\s+completed\b",
        re.IGNORECASE,
    ),

    # "We have completed the investigation."
    # "We completed the investigation."
    # "We finished the investigation."
    re.compile(
        r"\b(?:we\s+)?(?:have\s+)?(?:completed|finished)"
        r"\s+(?:the\s+)?investigation\b",
        re.IGNORECASE,
    ),

]


# ============================================================================
# UNSUPPORTED CUSTOMER-SPECIFIC CLAIMS
# ============================================================================

UNSUPPORTED_CUSTOMER_PATTERNS = [

    re.compile(
        r"\byour\s+order\s+will\s+arrive\b",
        re.IGNORECASE,
    ),

    re.compile(
        r"\byour\s+order\s+should\s+arrive\b",
        re.IGNORECASE,
    ),

    re.compile(
        r"\byou\s+will\s+receive\b",
        re.IGNORECASE,
    ),

    re.compile(
        r"\byou\s+should\s+receive\b",
        re.IGNORECASE,
    ),

    re.compile(
        r"\byour\s+card\s+was\s+charged\b",
        re.IGNORECASE,
    ),

    re.compile(
        r"\byou\s+were\s+charged\b",
        re.IGNORECASE,
    ),

]


# ============================================================================
# FOLLOW-UP DETECTION
# ============================================================================

FOLLOW_UP_PHRASES = [
    "reach out",
    "contact us",
    "get in touch",
    "send us a direct message",
    "send us a message",
    "message us",
    "dm us",
    "please contact",
    "please reach out",
    "please get in touch",
    "could you",
    "can you",
    "please share",
    "let us know",
    "provide more information",
    "provide more details",
    "share more information",
    "share more details",
]


EVIDENCE_FOLLOW_UP_TERMS = [
    "investigation",
    "follow-up",
    "follow up",
    "further investigation",
    "contact support",
    "contact customer support",
    "additional information",
    "more information",
    "more details",
    "check with support",
    "look into",
]


# ============================================================================
# RESULT
# ============================================================================

@dataclass
class GroundingResult:
    """Result returned by the grounding checker."""

    grounding_decision: str
    violations: List[str]
    warnings: List[str]

    @property
    def violation_count(self) -> int:
        return len(self.violations)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    @property
    def safe(self) -> bool:
        return self.grounding_decision != UNSAFE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "grounding_decision": self.grounding_decision,
            "violation_count": self.violation_count,
            "warning_count": self.warning_count,
            "violations": self.violations,
            "warnings": self.warnings,
        }


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================

def normalize_text(text: str) -> str:
    """Normalize whitespace and lowercase text."""

    if not isinstance(text, str):
        return ""

    return re.sub(
        r"\s+",
        " ",
        text.strip().lower(),
    )


# ============================================================================
# EVIDENCE HELPERS
# ============================================================================

def get_evidence_tier(
    evidence: Optional[Dict[str, Any]],
) -> str:
    """Convert evidence metadata into a coarse evidence tier."""

    if not evidence:
        return "NONE"

    quality = str(
        evidence.get(
            "evidence_quality",
            "",
        )
    ).strip().upper()

    resolution_type = str(
        evidence.get(
            "resolution_type",
            "",
        )
    ).strip().upper()

    if quality in {
        "HIGH",
        "STRONG",
        "GOOD",
    }:
        return "STRONG"

    if quality in {
        "MEDIUM",
        "MODERATE",
        "PARTIAL",
    }:
        return "MODERATE"

    if resolution_type in {
        "RESOLVED",
        "RESOLVED_BY_CUSTOMER",
    }:
        return "STRONG"

    if resolution_type in {
        "PARTIALLY_RESOLVED",
        "INVESTIGATION_REQUIRED",
        "FOLLOWUP_OR_INVESTIGATION",
    }:
        return "MODERATE"

    return "WEAK"


def evidence_suggests_follow_up(
    evidence_items: Optional[List[Dict[str, Any]]],
) -> bool:
    """
    Determine whether evidence indicates additional investigation
    or customer follow-up may be required.
    """

    if not evidence_items:
        return False

    for evidence in evidence_items:

        resolution_type = normalize_text(
            str(
                evidence.get(
                    "resolution_type",
                    "",
                )
            )
        )

        support_action = normalize_text(
            str(
                evidence.get(
                    "support_action",
                    "",
                )
            )
        )

        evidence_quality = normalize_text(
            str(
                evidence.get(
                    "evidence_quality",
                    "",
                )
            )
        )

        combined = " ".join(
            [
                resolution_type,
                support_action,
                evidence_quality,
            ]
        )

        if "investigation" in combined:
            return True

        if "follow-up" in combined:
            return True

        if "follow up" in combined:
            return True

        if any(
            term in combined
            for term in EVIDENCE_FOLLOW_UP_TERMS
        ):
            return True

    return False


# ============================================================================
# INTENT CONSISTENCY
# ============================================================================

def calculate_intent_consistency_v3(
    response: str,
    query: str,
    historical_intent: Optional[str] = None,
) -> float:
    """
    Lightweight lexical intent consistency score.

    This is a heuristic safety signal, not a trained classifier.
    """

    response_text = normalize_text(response)
    query_text = normalize_text(query)

    if not response_text:
        return 0.0

    if not query_text:
        return 0.5

    query_terms = {
        token
        for token in re.findall(
            r"\b[a-z0-9]+\b",
            query_text,
        )
        if len(token) >= 4
    }

    if not query_terms:
        return 0.5

    response_terms = set(
        re.findall(
            r"\b[a-z0-9]+\b",
            response_text,
        )
    )

    overlap = len(
        query_terms.intersection(
            response_terms
        )
    )

    lexical_score = overlap / len(query_terms)

    if historical_intent:

        intent_text = normalize_text(
            str(historical_intent)
        )

        intent_terms = {
            token
            for token in re.findall(
                r"\b[a-z0-9]+\b",
                intent_text,
            )
            if len(token) >= 4
        }

        if intent_terms:

            intent_overlap = len(
                intent_terms.intersection(
                    response_terms
                )
            )

            intent_score = min(
                1.0,
                intent_overlap / len(intent_terms),
            )

            return round(
                0.7 * lexical_score
                + 0.3 * intent_score,
                4,
            )

    return round(
        min(1.0, lexical_score),
        4,
    )


# ============================================================================
# SEMANTIC CONSISTENCY
# ============================================================================

def calculate_semantic_consistency_v3(
    response: str,
    evidence_items: Optional[List[Dict[str, Any]]],
) -> float:
    """
    Lightweight lexical evidence-to-response consistency score.
    """

    if not response or not evidence_items:
        return 0.5

    response_terms = {
        token
        for token in re.findall(
            r"\b[a-z0-9]+\b",
            normalize_text(response),
        )
        if len(token) >= 4
    }

    if not response_terms:
        return 0.0

    scores = []

    for evidence in evidence_items:

        evidence_text = " ".join(
            [
                str(
                    evidence.get(
                        "customer_problem",
                        "",
                    )
                ),
                str(
                    evidence.get(
                        "support_action",
                        "",
                    )
                ),
                str(
                    evidence.get(
                        "resolution_type",
                        "",
                    )
                ),
            ]
        )

        evidence_terms = {
            token
            for token in re.findall(
                r"\b[a-z0-9]+\b",
                normalize_text(evidence_text),
            )
            if len(token) >= 4
        }

        if not evidence_terms:
            continue

        overlap = len(
            response_terms.intersection(
                evidence_terms
            )
        )

        score = overlap / len(response_terms)

        scores.append(score)

    if not scores:
        return 0.5

    return round(
        max(scores),
        4,
    )


# ============================================================================
# MAIN CHECKER
# ============================================================================

def trust_check_v3(
    response: str,
    query: str,
    evidence_items: Optional[List[Dict[str, Any]]] = None,
    historical_intent: Optional[str] = None,
) -> GroundingResult:
    """
    Run grounding and safety checks.

    Decision policy:

        UNSAFE
            One or more direct safety violations.

        CONDITIONAL
            No direct violation, but evidence suggests follow-up
            and the response does not request it.

        GROUNDED
            No detected violations or warnings.
    """

    violations: List[str] = []
    warnings: List[str] = []

    response_text = normalize_text(response)

    # ----------------------------------------------------------------
    # Empty response
    # ----------------------------------------------------------------

    if not response_text:

        violations.append(
            "EMPTY_RESPONSE"
        )

        return GroundingResult(
            grounding_decision=UNSAFE,
            violations=violations,
            warnings=warnings,
        )

    # ----------------------------------------------------------------
    # Internal case ID
    # ----------------------------------------------------------------

    if INTERNAL_CASE_ID_PATTERN.search(response):

        violations.append(
            "INTERNAL_CASE_ID_LEAK"
        )

    # ----------------------------------------------------------------
    # Internal system information
    # ----------------------------------------------------------------

    for term in INTERNAL_SYSTEM_TERMS:

        if term in response_text:

            violations.append(
                "INTERNAL_SYSTEM_INFORMATION"
            )

            break

    # ----------------------------------------------------------------
    # Sensitive information
    # ----------------------------------------------------------------

    for pattern in SENSITIVE_PATTERNS.values():

        if pattern.search(response):

            violations.append(
                "POTENTIAL_SENSITIVE_INFORMATION"
            )

            break

    # ----------------------------------------------------------------
    # Unsupported resolution claims
    # ----------------------------------------------------------------

    for pattern in UNSUPPORTED_RESOLUTION_PATTERNS:

        if pattern.search(response):

            violations.append(
                "UNSUPPORTED_RESOLUTION_CLAIM"
            )

            break

    # ----------------------------------------------------------------
    # Unsupported customer-specific claims
    # ----------------------------------------------------------------

    for pattern in UNSUPPORTED_CUSTOMER_PATTERNS:

        if pattern.search(response):

            violations.append(
                "UNSUPPORTED_CUSTOMER_SPECIFIC_CLAIM"
            )

            break

    # ----------------------------------------------------------------
    # Evidence follow-up
    # ----------------------------------------------------------------

    if evidence_suggests_follow_up(
        evidence_items
    ):

        has_follow_up = any(
            phrase in response_text
            for phrase in FOLLOW_UP_PHRASES
        )

        if not has_follow_up:

            warnings.append(
                "EVIDENCE_SUGGESTS_FOLLOWUP"
            )

    # ----------------------------------------------------------------
    # Final decision
    # ----------------------------------------------------------------

    if violations:

        decision = UNSAFE

    elif warnings:

        decision = CONDITIONAL

    else:

        decision = GROUNDED

    return GroundingResult(
        grounding_decision=decision,
        violations=violations,
        warnings=warnings,
    )


# ============================================================================
# PUBLIC WRAPPER
# ============================================================================

def grounding_check(
    response: str,
    query: str,
    evidence_items: Optional[List[Dict[str, Any]]] = None,
    historical_intent: Optional[str] = None,
) -> GroundingResult:
    """Public wrapper for the grounding checker."""

    return trust_check_v3(
        response=response,
        query=query,
        evidence_items=evidence_items,
        historical_intent=historical_intent,
    )


# ============================================================================
# UNIT TESTS
# ============================================================================

def _run_tests() -> None:
    """
    Deterministic unit tests.

    No Gemini/API/LLM calls are made.
    """

    # ----------------------------------------------------------------
    # TEST 1 — Safe response
    # ----------------------------------------------------------------

    result = grounding_check(
        response=(
            "We're sorry about the issue. "
            "Please send us a direct message so we can look into this."
        ),
        query="My order has not arrived.",
        evidence_items=[
            {
                "customer_problem": (
                    "Customer reports a delayed order."
                ),
                "support_action": (
                    "Ask the customer to contact support."
                ),
                "resolution_type": (
                    "INVESTIGATION_REQUIRED"
                ),
                "evidence_quality": "HIGH",
            }
        ],
    )

    assert result.grounding_decision == GROUNDED
    assert result.violation_count == 0
    assert result.warning_count == 0

    # ----------------------------------------------------------------
    # TEST 2 — Invented payment status
    # ----------------------------------------------------------------

    result = grounding_check(
        response=(
            "Your payment was processed successfully."
        ),
        query="My payment is not showing.",
    )

    assert result.grounding_decision == UNSAFE
    assert (
        "UNSUPPORTED_RESOLUTION_CLAIM"
        in result.violations
    )

    # ----------------------------------------------------------------
    # TEST 3 — Invented refund
    # ----------------------------------------------------------------

    result = grounding_check(
        response=(
            "Your refund has been processed."
        ),
        query="I want a refund.",
    )

    assert result.grounding_decision == UNSAFE
    assert (
        "UNSUPPORTED_RESOLUTION_CLAIM"
        in result.violations
    )

    # ----------------------------------------------------------------
    # TEST 4 — Invented investigation
    # ----------------------------------------------------------------

    result = grounding_check(
        response=(
            "We have completed the investigation."
        ),
        query="Where is my order?",
    )

    assert result.grounding_decision == UNSAFE
    assert (
        "UNSUPPORTED_RESOLUTION_CLAIM"
        in result.violations
    )

    # ----------------------------------------------------------------
    # TEST 5 — Internal case ID leakage
    # ----------------------------------------------------------------

    result = grounding_check(
        response=(
            "We found a similar case AMZ-123456."
        ),
        query="My package is delayed.",
    )

    assert result.grounding_decision == UNSAFE
    assert (
        "INTERNAL_CASE_ID_LEAK"
        in result.violations
    )

    # ----------------------------------------------------------------
    # TEST 6 — Internal retrieval information
    # ----------------------------------------------------------------

    result = grounding_check(
        response=(
            "The cross encoder gave this case "
            "a high relevance score."
        ),
        query="My order is delayed.",
    )

    assert result.grounding_decision == UNSAFE
    assert (
        "INTERNAL_SYSTEM_INFORMATION"
        in result.violations
    )

    # ----------------------------------------------------------------
    # TEST 7 — Sensitive URL leakage
    # ----------------------------------------------------------------

    result = grounding_check(
        response=(
            "Please visit "
            "https://example.com/account/1234567890."
        ),
        query="I need account help.",
    )

    assert result.grounding_decision == UNSAFE
    assert (
        "POTENTIAL_SENSITIVE_INFORMATION"
        in result.violations
    )

    # ----------------------------------------------------------------
    # TEST 8 — Evidence requires follow-up
    # ----------------------------------------------------------------

    result = grounding_check(
        response=(
            "We're sorry about the issue with your order."
        ),
        query="My package has not arrived.",
        evidence_items=[
            {
                "customer_problem": (
                    "Customer's package was delayed."
                ),
                "support_action": (
                    "Support needs additional information."
                ),
                "resolution_type": (
                    "INVESTIGATION_REQUIRED"
                ),
                "evidence_quality": "MEDIUM",
            }
        ],
    )

    assert result.grounding_decision == CONDITIONAL
    assert (
        "EVIDENCE_SUGGESTS_FOLLOWUP"
        in result.warnings
    )

    # ----------------------------------------------------------------
    # TEST 9 — Follow-up satisfies evidence requirement
    # ----------------------------------------------------------------

    result = grounding_check(
        response=(
            "We're sorry about the delay. "
            "Could you please share the current tracking status?"
        ),
        query="My package has not arrived.",
        evidence_items=[
            {
                "customer_problem": (
                    "Customer's package was delayed."
                ),
                "support_action": (
                    "Support needs additional information."
                ),
                "resolution_type": (
                    "INVESTIGATION_REQUIRED"
                ),
                "evidence_quality": "MEDIUM",
            }
        ],
    )

    assert result.grounding_decision == GROUNDED
    assert result.warning_count == 0
    assert result.violation_count == 0

    # ----------------------------------------------------------------
    # TEST 10 — Empty response
    # ----------------------------------------------------------------

    result = grounding_check(
        response="",
        query="Where is my order?",
    )

    assert result.grounding_decision == UNSAFE
    assert "EMPTY_RESPONSE" in result.violations

    # ----------------------------------------------------------------
    # TEST 11 — Unsupported delivery claim
    # ----------------------------------------------------------------

    result = grounding_check(
        response=(
            "Your order will arrive tomorrow."
        ),
        query="Where is my package?",
    )

    assert result.grounding_decision == UNSAFE
    assert (
        "UNSUPPORTED_CUSTOMER_SPECIFIC_CLAIM"
        in result.violations
    )

    # ----------------------------------------------------------------
    # TEST 12 — Intent consistency
    # ----------------------------------------------------------------

    score = calculate_intent_consistency_v3(
        response=(
            "Please check the tracking status "
            "of your package."
        ),
        query="Where is my package?",
        historical_intent="ORDER_TRACKING",
    )

    assert 0.0 <= score <= 1.0

    # ----------------------------------------------------------------
    # TEST 13 — Semantic consistency
    # ----------------------------------------------------------------

    score = calculate_semantic_consistency_v3(
        response=(
            "Please check the tracking status "
            "of your package."
        ),
        evidence_items=[
            {
                "customer_problem": (
                    "Customer asks about package tracking."
                ),
                "support_action": (
                    "Ask for the current tracking status."
                ),
                "resolution_type": (
                    "INVESTIGATION_REQUIRED"
                ),
            }
        ],
    )

    assert 0.0 <= score <= 1.0

    # ----------------------------------------------------------------
    # TEST 14 — Investigation regression tests
    # ----------------------------------------------------------------

    investigation_variants = [
        "We have completed the investigation.",
        "We completed the investigation.",
        "We finished the investigation.",
        "The investigation has been completed.",
        "The investigation was completed.",
        "The investigation is complete.",
    ]

    for response in investigation_variants:

        result = grounding_check(
            response=response,
            query="Where is my order?",
        )

        assert result.grounding_decision == UNSAFE, (
            f"Investigation claim was not blocked: {response}"
        )

        assert (
            "UNSUPPORTED_RESOLUTION_CLAIM"
            in result.violations
        )

    # ----------------------------------------------------------------
    # FINAL
    # ----------------------------------------------------------------

    print("=" * 100)
    print("GROUNDING & SAFETY CHECKER TEST: PASS")
    print("=" * 100)
    print("All 14 local tests passed.")
    print("No API or LLM calls were made.")


if __name__ == "__main__":
    _run_tests()