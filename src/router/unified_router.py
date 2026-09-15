"""
Unified Router
--------------
Combines:

1. Intent classification
2. Risk classification
3. Query complexity estimation

into one validated routing result.

The router is designed to fail safely:
    - Invalid intent -> GENERAL_SUPPORT
    - Invalid risk -> MEDIUM
    - Invalid complexity -> COMPLEX
    - Runtime failure -> fail-safe routing
"""

from typing import Dict

from intent_router import SUPPORTED_INTENTS, intent_router
from risk_router import risk_router


# ============================================================
# SUPPORTED VALUES
# ============================================================

SUPPORTED_RISKS = {
    "LOW",
    "MEDIUM",
    "HIGH",
}

SUPPORTED_COMPLEXITIES = {
    "SIMPLE",
    "MEDIUM",
    "COMPLEX",
}


# ============================================================
# QUERY COMPLEXITY
# ============================================================

def calculate_query_complexity(query: str) -> Dict[str, object]:
    """
    Estimate query complexity using deterministic features.

    Rules:

    COMPLEX:
        - multiple questions
        - more than 25 words
        - more than 160 characters

    MEDIUM:
        - more than 12 words
        - exactly one question

    SIMPLE:
        - 12 words or fewer
        - no question
    """

    try:
        text = str(query).strip()

        words = text.split()
        word_count = len(words)
        char_count = len(text)
        question_count = text.count("?")

        if (
            question_count > 1
            or word_count > 25
            or char_count > 160
        ):
            complexity = "COMPLEX"

        elif (
            word_count > 12
            or question_count == 1
        ):
            complexity = "MEDIUM"

        else:
            complexity = "SIMPLE"

        return {
            "complexity": complexity,
            "word_count": word_count,
            "char_count": char_count,
            "question_count": question_count,
        }

    except Exception:
        return {
            "complexity": "COMPLEX",
            "word_count": 0,
            "char_count": 0,
            "question_count": 0,
        }


# ============================================================
# ROUTER OUTPUT VALIDATION
# ============================================================

def validate_router_output(
    intent: str,
    risk_level: str,
    complexity: str,
) -> Dict[str, object]:
    """
    Validate all router outputs against the supported taxonomy.
    """

    errors = []

    if intent not in SUPPORTED_INTENTS:
        errors.append(
            f"Invalid intent: {intent}"
        )

    if risk_level not in SUPPORTED_RISKS:
        errors.append(
            f"Invalid risk level: {risk_level}"
        )

    if complexity not in SUPPORTED_COMPLEXITIES:
        errors.append(
            f"Invalid complexity: {complexity}"
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }


# ============================================================
# FAIL-SAFE FALLBACK
# ============================================================

def safe_router_fallback(
    reason: str = "UNKNOWN_ERROR",
) -> Dict[str, object]:
    """
    Conservative fail-safe routing.

    Unknown failures must never silently become LOW risk.
    """

    return {
        "intent": "GENERAL_SUPPORT",
        "risk_level": "MEDIUM",
        "complexity": "COMPLEX",
        "word_count": 0,
        "char_count": 0,
        "question_count": 0,
        "router_status": "FAIL_SAFE",
        "fallback_reason": reason,
    }


# ============================================================
# RESULT BUILDER
# ============================================================

def build_router_result(
    query: str,
    intent: str,
    risk_level: str,
    complexity_info: Dict[str, object],
) -> Dict[str, object]:
    """
    Build and validate the unified router result.
    """

    complexity = complexity_info["complexity"]

    validation = validate_router_output(
        intent=intent,
        risk_level=risk_level,
        complexity=complexity,
    )

    if not validation["valid"]:
        return safe_router_fallback(
            reason="INVALID_ROUTER_OUTPUT"
        )

    return {
        "query": query,
        "intent": intent,
        "risk_level": risk_level,
        "complexity": complexity,
        "word_count": complexity_info["word_count"],
        "char_count": complexity_info["char_count"],
        "question_count": complexity_info["question_count"],
        "router_status": "SUCCESS",
        "fallback_reason": None,
    }


# ============================================================
# UNIFIED ROUTER
# ============================================================

def run_unified_router(query: str) -> Dict[str, object]:
    """
    Run the complete routing pipeline:

        Query
          ↓
        Intent Router
          ↓
        Risk Router
          ↓
        Complexity Estimator
          ↓
        Validation
          ↓
        Unified Result
    """

    try:
        intent_result = intent_router(query)
        risk_result = risk_router(query)
        complexity_result = calculate_query_complexity(query)

        return build_router_result(
            query=query,
            intent=intent_result["intent"],
            risk_level=risk_result["risk_level"],
            complexity_info=complexity_result,
        )

    except Exception as exc:
        return safe_router_fallback(
            reason=f"ROUTER_EXCEPTION:{type(exc).__name__}"
        )


# ============================================================
# LOCAL TEST SUITE
# ============================================================

if __name__ == "__main__":

    test_cases = [
        (
            "Where is my package?",
            "ORDER_TRACKING",
            "LOW",
            "MEDIUM",
        ),
        (
            "My order delivery is delayed",
            "ORDER_DELIVERY",
            "LOW",
            "SIMPLE",
        ),
        (
            "I forgot my password",
            "ACCOUNT_ACCESS",
            "MEDIUM",
            "SIMPLE",
        ),
        (
            "My account was hacked",
            "ACCOUNT_SECURITY",
            "HIGH",
            "SIMPLE",
        ),
        (
            "There is an unauthorized charge",
            "PAYMENT_BILLING",
            "HIGH",
            "SIMPLE",
        ),
        (
            "I want to cancel my order",
            "ORDER_CANCELLATION",
            "MEDIUM",
            "SIMPLE",
        ),
        (
            "I want to buy this product",
            "ORDER_PURCHASE",
            "LOW",
            "SIMPLE",
        ),
        (
            "Are you leaking customer personal information?",
            "GENERAL_SUPPORT",
            "HIGH",
            "MEDIUM",
        ),
    ]

    print("=" * 100)
    print("HIVER SUPPORT AGENT — UNIFIED ROUTER TEST")
    print("=" * 100)

    passed = 0

    for (
        query,
        expected_intent,
        expected_risk,
        expected_complexity,
    ) in test_cases:

        result = run_unified_router(query)

        intent_ok = result["intent"] == expected_intent
        risk_ok = result["risk_level"] == expected_risk
        complexity_ok = result["complexity"] == expected_complexity

        status = (
            "PASS"
            if intent_ok and risk_ok and complexity_ok
            else "FAIL"
        )

        if status == "PASS":
            passed += 1

        print(
            f"{status:4} | "
            f"Intent={result['intent']:22} | "
            f"Risk={result['risk_level']:6} | "
            f"Complexity={result['complexity']:7} | "
            f"{query}"
        )

    print("=" * 100)
    print(
        f"Tests passed: {passed}/{len(test_cases)}"
    )
    print("=" * 100)

    # --------------------------------------------------------
    # FAIL-SAFE TEST
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("FAIL-SAFE VALIDATION TEST")
    print("=" * 100)

    invalid_result = validate_router_output(
        intent="INVALID_INTENT",
        risk_level="INVALID_RISK",
        complexity="INVALID_COMPLEXITY",
    )

    fallback = safe_router_fallback(
        reason="UNIT_TEST_INVALID_OUTPUT"
    )

    print("Validation result:")
    print(invalid_result)

    print()
    print("Fallback result:")
    print(fallback)

    assert invalid_result["valid"] is False
    assert fallback["intent"] == "GENERAL_SUPPORT"
    assert fallback["risk_level"] == "MEDIUM"
    assert fallback["complexity"] == "COMPLEX"

    print()
    print("FAIL-SAFE TEST: PASS")
    print("=" * 100)