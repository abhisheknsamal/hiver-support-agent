"""
Risk Router
-----------
Conservative, fail-closed risk classification for the Hiver Support Agent.

Risk levels:
    HIGH   -> security, fraud, unauthorized payment, privacy/data exposure
    MEDIUM -> payment, account access, cancellation, marketplace/seller
    LOW    -> normal support requests

Safety principle:
    If classification fails, return MEDIUM rather than LOW.
"""

import re
from typing import Dict


# ============================================================
# HIGH-RISK PATTERNS
# ============================================================

HIGH_RISK_PATTERNS = [
    # Account security / compromise
    r"\bhacked\b",
    r"\bhack(ed|ing)?\b",
    r"\bcompromised\b",
    r"\bunauthorized access\b",
    r"\bstolen account\b",
    r"\bsomeone accessed my account\b",
    r"\bsomeone got into my account\b",

    # Fraud / unauthorized financial activity
    r"\bunauthorized charge\b",
    r"\bunauthorized payment\b",
    r"\bunauthorized transaction\b",
    r"\bfraud\b",
    r"\bfraudulent\b",
    r"\bcharged without permission\b",
    r"\bcharged without authorization\b",

    # Privacy / personal-data exposure
    r"\bleaking (customer )?(personal )?information\b",
    r"\bleak(ed|ing)? (customer )?(personal )?information\b",
    r"\bpersonal data exposed\b",
    r"\bpersonal data has been exposed\b",
    r"\bpersonal data was exposed\b",
    r"\bpersonal information exposed\b",
    r"\bpersonal information has been exposed\b",
    r"\bpersonal information was exposed\b",
    r"\bdata breach\b",
    r"\bprivacy breach\b",
    r"\bexposed (my|customer) data\b",
    r"\bexposed (my|customer) personal information\b",
    r"\bprivate information exposed\b",
]


# ============================================================
# MEDIUM-RISK PATTERNS
# ============================================================

MEDIUM_RISK_PATTERNS = [
    # Payment / billing
    r"\bpayment\b",
    r"\bpaid\b",
    r"\bcharged\b",
    r"\bcharge\b",
    r"\bbilling\b",
    r"\bcredit card\b",
    r"\bdebit card\b",

    # Account access
    r"\bforgot (my )?password\b",
    r"\bpassword\b",
    r"\blogin\b",
    r"\blog in\b",
    r"\bsign in\b",
    r"\baccount access\b",

    # Cancellation
    r"\bcancel\b",
    r"\bcancelled\b",
    r"\bcancellation\b",

    # Marketplace / seller
    r"\bseller\b",
    r"\bmarketplace\b",
]


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def _normalize(text: str) -> str:
    """Normalize input text for rule matching."""
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def _contains_any(text: str, patterns: list[str]) -> bool:
    """Return True when any supplied regex matches the text."""
    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in patterns
    )


# ============================================================
# CORE RISK CLASSIFIER
# ============================================================

def classify_risk(query: str) -> str:
    """
    Classify a support query as LOW, MEDIUM, or HIGH.

    HIGH is checked first because safety-critical signals must
    override lower-risk categories.

    Unexpected failures fail closed to MEDIUM.
    """
    try:
        text = _normalize(query)

        if not text:
            return "MEDIUM"

        # Safety-critical signals take priority.
        if _contains_any(text, HIGH_RISK_PATTERNS):
            return "HIGH"

        # Potentially sensitive but not immediately critical.
        if _contains_any(text, MEDIUM_RISK_PATTERNS):
            return "MEDIUM"

        # Normal support request.
        return "LOW"

    except Exception:
        # Fail-safe behavior.
        return "MEDIUM"


# ============================================================
# STRUCTURED ROUTER OUTPUT
# ============================================================

def risk_router(query: str) -> Dict[str, str]:
    """
    Return a structured risk-routing result.

    Example:
        {
            "risk_level": "HIGH",
            "router_status": "SUCCESS"
        }
    """
    try:
        risk_level = classify_risk(query)

        return {
            "risk_level": risk_level,
            "router_status": "SUCCESS",
        }

    except Exception:
        return {
            "risk_level": "MEDIUM",
            "router_status": "FAIL_SAFE",
        }


# ============================================================
# LOCAL SAFETY TESTS
# ============================================================

if __name__ == "__main__":

    test_queries = [
        # LOW
        "Where is my package?",
        "My delivery is delayed",
        "I want to buy this product",

        # MEDIUM
        "I forgot my password",
        "I was charged twice",
        "I want to cancel my order",

        # HIGH
        "My account was hacked",
        "There is an unauthorized charge",
        "Are you leaking customer personal information?",
        "My personal data has been exposed",
    ]

    print("=" * 75)
    print("HIVER SUPPORT AGENT — RISK ROUTER TEST")
    print("=" * 75)

    for query in test_queries:
        result = risk_router(query)

        print(
            f"{result['risk_level']:6} | "
            f"{result['router_status']:9} | "
            f"{query}"
        )

    print("=" * 75)