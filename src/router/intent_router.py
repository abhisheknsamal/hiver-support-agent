"""
Intent Router
-------------
Deterministic intent classification layer for the Hiver Support Agent.

This module defines the project's supported intent taxonomy and provides
a conservative rule-based intent router.

IMPORTANT:
This is the repository runtime fallback/router implementation.
It is NOT the previously evaluated TF-IDF + Logistic Regression model.
The trained classifier was not persisted as a reusable model artifact.

The router is designed behind a simple interface so an SLM or trained
classifier can replace it later without changing downstream components.
"""

import re
from typing import Dict


# ============================================================
# SUPPORTED INTENT TAXONOMY
# ============================================================

SUPPORTED_INTENTS = [
    "ORDER_DELIVERY",
    "ORDER_TRACKING",
    "PAYMENT_BILLING",
    "RETURNS_REFUNDS",
    "SELLER_AUTHENTICITY",
    "ACCOUNT_ACCESS",
    "ACCOUNT_SECURITY",
    "DEVICE_TECHNICAL",
    "DIGITAL_CONTENT",
    "PURCHASE_CONTROL",
    "ORDER_CANCELLATION",
    "PREORDER_DELIVERY",
    "GENERAL_SUPPORT",
    "GENERAL_SOCIAL",
    "MARKETPLACE_SELLING",
    "ORDER_PURCHASE",
]


# ============================================================
# HIGH-CONFIDENCE INTENT PATTERNS
# ============================================================

INTENT_PATTERNS = {

    "ACCOUNT_SECURITY": [
        r"\bhacked\b",
        r"\bcompromised\b",
        r"\bunauthorized access\b",
        r"\bsuspicious activity\b",
        r"\bsecurity\b",
        r"\bstolen account\b",
        r"\bsomeone accessed my account\b",
    ],

    "SELLER_AUTHENTICITY": [
        r"\bis this seller legit\b",
        r"\blegit seller\b",
        r"\btrusted seller\b",
        r"\bauthentic seller\b",
        r"\bfake seller\b",
        r"\bscam seller\b",
        r"\bis this seller genuine\b",
        r"\bauthenticity\b",
    ],

    "MARKETPLACE_SELLING": [
        r"\bsell on amazon\b",
        r"\bsell through amazon\b",
        r"\bmarketplace seller\b",
        r"\bbecome a seller\b",
        r"\bstart selling\b",
        r"\bseller account\b",
    ],

    "PURCHASE_CONTROL": [
        r"\bchild ordered\b",
        r"\bchild bought\b",
        r"\bordered by mistake\b",
        r"\bpurchased by mistake\b",
        r"\baccidental purchase\b",
        r"\bvoice purchase\b",
        r"\balexa.*purchase\b",
    ],

    "ACCOUNT_ACCESS": [
        r"\bforgot (my )?password\b",
        r"\breset (my )?password\b",
        r"\bpassword reset\b",
        r"\bcannot log in\b",
        r"\bcan't log in\b",
        r"\bcannot login\b",
        r"\bcan't login\b",
        r"\bunable to login\b",
        r"\bunable to log in\b",
        r"\bcannot access my account\b",
        r"\bcan't access my account\b",
        r"\bunable to access my account\b",
        r"\baccount access\b",
        r"\blogin problem\b",
        r"\bsign in problem\b",
    ],

    "DEVICE_TECHNICAL": [
        r"\bphone\b",
        r"\biphone\b",
        r"\bandroid\b",
        r"\boneplus\b",
        r"\bsamsung\b",
        r"\bdevice\b",
        r"\bapp.*not working\b",
        r"\bapplication.*not working\b",
        r"\bapp.*issue\b",
        r"\btechnical issue\b",
        r"\btechnical problem\b",
        r"\bfeature.*not working\b",
    ],

    "PAYMENT_BILLING": [
        r"\bpayment\b",
        r"\bpaid\b",
        r"\bcharged\b",
        r"\bcharge\b",
        r"\bbilling\b",
        r"\bcredit card\b",
        r"\bdebit card\b",
        r"\bpayment failed\b",
        r"\bpayment problem\b",
        r"\bcharged twice\b",
        r"\bdouble charged\b",
    ],

    "RETURNS_REFUNDS": [
        r"\brefund\b",
        r"\brefunded\b",
        r"\breturn\b",
        r"\breturned\b",
        r"\breturning\b",
        r"\bmoney back\b",
        r"\bdefective product\b",
        r"\bdamaged product\b",
        r"\bdamaged item\b",
    ],

    "ORDER_CANCELLATION": [
        r"\bcancel my order\b",
        r"\bcancel order\b",
        r"\bcancelled order\b",
        r"\bcancellation\b",
        r"\bwant to cancel\b",
    ],

    "PREORDER_DELIVERY": [
        r"\bpreorder\b",
        r"\bpre-order\b",
        r"\bpre order\b",
        r"\bpreordered\b",
        r"\bpre-ordered\b",
    ],

    "ORDER_TRACKING": [
        r"\btrack my order\b",
        r"\btrack my package\b",
        r"\btracking number\b",
        r"\btracking status\b",
        r"\bwhere is my package\b",
        r"\bwhere is my order\b",
        r"\bpackage location\b",
        r"\border location\b",
    ],

    "ORDER_DELIVERY": [
        r"\bdelivery\b",
        r"\bdelivered\b",
        r"\bdelivery delay\b",
        r"\bdelayed delivery\b",
        r"\blate delivery\b",
        r"\blate package\b",
        r"\bpackage.*late\b",
        r"\bpackage.*delayed\b",
        r"\border.*late\b",
        r"\border.*delayed\b",
        r"\bwhen will.*arrive\b",
        r"\bwhen.*arrive\b",
        r"\bdelivery date\b",
        r"\bdelivery time\b",
    ],

    "DIGITAL_CONTENT": [
        r"\bdigital content\b",
        r"\bkindle\b",
        r"\bebook\b",
        r"\be-book\b",
        r"\baudible\b",
        r"\bdigital book\b",
        r"\bdigital purchase\b",
        r"\bmovie\b",
        r"\bvideo content\b",
    ],

    "ORDER_PURCHASE": [
        r"\bhow do i buy\b",
        r"\bhow can i buy\b",
        r"\bwant to buy\b",
        r"\bwant to purchase\b",
        r"\bhow to purchase\b",
        r"\bplace an order\b",
        r"\bmake an order\b",
        r"\bpurchase\b",
    ],
}


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_query(query: str) -> str:
    """Normalize a customer query for intent matching."""
    return re.sub(r"\s+", " ", str(query).lower()).strip()


# ============================================================
# PATTERN MATCHING
# ============================================================

def _match_intent(text: str, intent: str) -> bool:
    """Check whether an intent's patterns match the query."""
    patterns = INTENT_PATTERNS.get(intent, [])

    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in patterns
    )


# ============================================================
# INTENT CLASSIFICATION
# ============================================================

def classify_intent(query: str) -> str:
    """
    Classify a customer query into the supported intent taxonomy.

    Specific intents are evaluated before broader intents.
    Unknown or ambiguous queries fall back to GENERAL_SUPPORT.
    """
    try:
        text = normalize_query(query)

        if not text:
            return "GENERAL_SUPPORT"

        priority_order = [
            "ACCOUNT_SECURITY",
            "SELLER_AUTHENTICITY",
            "MARKETPLACE_SELLING",
            "PURCHASE_CONTROL",
            "ACCOUNT_ACCESS",
            "ORDER_CANCELLATION",
            "PREORDER_DELIVERY",
            "ORDER_TRACKING",
            "PAYMENT_BILLING",
            "RETURNS_REFUNDS",
            "DEVICE_TECHNICAL",
            "DIGITAL_CONTENT",
            "ORDER_DELIVERY",
            "ORDER_PURCHASE",
        ]

        for intent in priority_order:
            if _match_intent(text, intent):
                return intent

        return "GENERAL_SUPPORT"

    except Exception:
        return "GENERAL_SUPPORT"


# ============================================================
# STRUCTURED ROUTER OUTPUT
# ============================================================

def intent_router(query: str) -> Dict[str, str]:
    """
    Return a structured intent-routing result.
    """
    try:
        intent = classify_intent(query)

        return {
            "intent": intent,
            "router_status": "SUCCESS",
        }

    except Exception:
        return {
            "intent": "GENERAL_SUPPORT",
            "router_status": "FAIL_SAFE",
        }


# ============================================================
# LOCAL TEST SUITE
# ============================================================

if __name__ == "__main__":

    test_cases = [
        ("Where is my package?", "ORDER_TRACKING"),
        ("My order delivery is delayed", "ORDER_DELIVERY"),
        ("I want to buy this product", "ORDER_PURCHASE"),
        ("I forgot my password", "ACCOUNT_ACCESS"),
        ("I was charged twice", "PAYMENT_BILLING"),
        ("I want a refund", "RETURNS_REFUNDS"),
        ("I want to cancel my order", "ORDER_CANCELLATION"),
        ("My account was hacked", "ACCOUNT_SECURITY"),
        ("Is this seller legit?", "SELLER_AUTHENTICITY"),
        ("I want to become a seller", "MARKETPLACE_SELLING"),
        ("My child ordered something using Alexa", "PURCHASE_CONTROL"),
        ("My OnePlus app is not working", "DEVICE_TECHNICAL"),
        ("When will my preorder arrive?", "PREORDER_DELIVERY"),
        ("I cannot access my account", "ACCOUNT_ACCESS"),
        ("I have a Kindle ebook problem", "DIGITAL_CONTENT"),
        ("Can you help me?", "GENERAL_SUPPORT"),
    ]

    print("=" * 85)
    print("HIVER SUPPORT AGENT — INTENT ROUTER TEST")
    print("=" * 85)

    passed = 0

    for query, expected in test_cases:
        result = intent_router(query)

        predicted = result["intent"]

        status = "PASS" if predicted == expected else "FAIL"

        if status == "PASS":
            passed += 1

        print(
            f"{status:4} | "
            f"Expected: {expected:22} | "
            f"Predicted: {predicted:22} | "
            f"{query}"
        )

    print("=" * 85)
    print(f"Tests passed: {passed}/{len(test_cases)}")
    print("=" * 85)
