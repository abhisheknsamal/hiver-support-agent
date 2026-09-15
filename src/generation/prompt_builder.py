"""
Generation Prompt Builder
-------------------------

Builds a strict evidence-grounded prompt for the response generator.
"""

from typing import Dict, List


# ============================================================
# SYSTEM INSTRUCTIONS
# ============================================================

SYSTEM_INSTRUCTIONS = """
You are a customer-support response assistant.

Your task is to draft a concise, polite customer-facing response
using the customer's current query and historical support evidence.

IMPORTANT:

1. Historical evidence is guidance only.
   It describes how similar historical cases were handled.
   It does NOT prove the current customer's order status,
   payment status, refund status, delivery date, account state,
   investigation result, or any action taken.

2. NEVER invent:
   - order status
   - delivery date
   - tracking status
   - refund amount
   - payment status
   - account information
   - investigation results
   - policies not present in the evidence
   - actions already taken

3. If the historical evidence suggests that additional details,
   investigation, or follow-up are required, ask the customer for
   the appropriate information or direct them to an appropriate
   support channel.

4. NEVER expose internal system information such as:
   - case IDs
   - retrieval scores
   - BM25 scores
   - BGE scores
   - RRF scores
   - cross-encoder scores
   - ranking information
   - internal model names
   - historical evidence terminology

5. NEVER repeat:
   - phone numbers
   - email addresses
   - order IDs
   - account IDs
   - payment card numbers
   - URLs
   - social-media handles
   - other sensitive identifiers

6. Do not copy historical support responses verbatim.
   Use the evidence only to understand the appropriate type of
   response and next step.

7. Be concise, natural, professional, and empathetic.

8. Output ONLY the customer-facing response.
"""


# ============================================================
# EVIDENCE FORMATTER
# ============================================================

def format_evidence(
    evidence: List[Dict[str, object]],
) -> str:

    if not evidence:
        return (
            "No historical evidence is available."
        )

    sections = []

    for index, item in enumerate(
        evidence,
        start=1,
    ):

        customer_problem = str(
            item.get(
                "customer_problem",
                "",
            )
        ).strip()

        support_action = str(
            item.get(
                "support_action",
                "",
            )
        ).strip()

        resolution_type = str(
            item.get(
                "resolution_type",
                "",
            )
        ).strip()

        evidence_quality = str(
            item.get(
                "evidence_quality",
                "",
            )
        ).strip()

        sections.append(
            f"""
Historical example {index}:

Customer problem:
{customer_problem}

Historical support action:
{support_action}

Resolution type:
{resolution_type}

Evidence quality:
{evidence_quality}
""".strip()
        )

    return "\n\n".join(
        sections
    )


# ============================================================
# PROMPT BUILDER
# ============================================================

def build_generation_prompt(
    customer_query: str,
    evidence: List[Dict[str, object]],
) -> str:

    if not isinstance(
        customer_query,
        str,
    ):
        customer_query = str(
            customer_query
        )

    customer_query = customer_query.strip()

    if not customer_query:
        raise ValueError(
            "Customer query cannot be empty."
        )

    evidence_text = format_evidence(
        evidence
    )

    prompt = f"""
{SYSTEM_INSTRUCTIONS.strip()}

CURRENT CUSTOMER QUERY
----------------------

{customer_query}

HISTORICAL SUPPORT EVIDENCE
---------------------------

{evidence_text}

TASK
----

Draft the safest and most useful customer-facing response.

Remember:

- Do not claim that a current action has already happened.
- Do not invent information missing from the query or evidence.
- If more information is needed, ask for it.
- If the issue requires investigation or human support, say so
  without claiming that an investigation has already occurred.
- Do not mention the historical examples or internal system.
- Output only the customer-facing response.
""".strip()

    return prompt


# ============================================================
# CONTEXT BUILDER
# ============================================================

def build_generation_context(
    customer_query: str,
    retrieval_result: Dict[str, object],
) -> Dict[str, object]:

    if not isinstance(
        retrieval_result,
        dict,
    ):
        raise TypeError(
            "retrieval_result must be a dictionary."
        )

    final_evidence = retrieval_result.get(
        "final_evidence"
    )

    if final_evidence is None:
        raise ValueError(
            "retrieval_result must contain final_evidence."
        )

    evidence_records = []

    if hasattr(
        final_evidence,
        "to_dict",
    ):
        records = final_evidence.to_dict(
            orient="records"
        )
    else:
        records = final_evidence

    for item in records:

        evidence_records.append(
            {
                "customer_problem": item.get(
                    "customer_problem",
                    "",
                ),
                "support_action": item.get(
                    "support_action",
                    "",
                ),
                "resolution_type": item.get(
                    "resolution_type",
                    "",
                ),
                "evidence_quality": item.get(
                    "evidence_quality",
                    "",
                ),
            }
        )

    prompt = build_generation_prompt(
        customer_query=customer_query,
        evidence=evidence_records,
    )

    return {
        "customer_query": customer_query,
        "evidence": evidence_records,
        "prompt": prompt,
    }


# ============================================================
# LOCAL TESTS
# ============================================================

if __name__ == "__main__":

    print("=" * 100)
    print("HIVER SUPPORT AGENT — GENERATION PROMPT BUILDER TEST")
    print("=" * 100)

    evidence = [
        {
            "case_id": "INTERNAL-CASE-001",
            "customer_problem": (
                "Customer could not locate their package."
            ),
            "support_action": (
                "Provided delivery tracking guidance."
            ),
            "resolution_type": "ORDER_TRACKING",
            "evidence_quality": "HIGH",
            "ce_score": 0.99,
            "rrf_score": 0.03,
        },
        {
            "case_id": "INTERNAL-CASE-002",
            "customer_problem": (
                "Customer reported a delayed delivery."
            ),
            "support_action": (
                "Requested current tracking information."
            ),
            "resolution_type": "ORDER_DELIVERY",
            "evidence_quality": "HIGH",
            "ce_score": 0.95,
            "rrf_score": 0.02,
        },
    ]

    # --------------------------------------------------------
    # Evidence formatting
    # --------------------------------------------------------

    evidence_text = format_evidence(
        evidence
    )

    assert (
        "Customer could not locate their package."
        in evidence_text
    )

    assert (
        "Provided delivery tracking guidance."
        in evidence_text
    )

    print(
        "PASS | Evidence formatting"
    )

    # --------------------------------------------------------
    # Internal information isolation
    # --------------------------------------------------------

    assert (
        "INTERNAL-CASE-001"
        not in evidence_text
    )

    assert (
        "0.99"
        not in evidence_text
    )

    assert (
        "0.03"
        not in evidence_text
    )

    print(
        "PASS | Retrieval metadata isolation"
    )

    # --------------------------------------------------------
    # Prompt construction
    # --------------------------------------------------------

    prompt = build_generation_prompt(
        customer_query=(
            "Where is my package?"
        ),
        evidence=evidence,
    )

    assert (
        "Where is my package?"
        in prompt
    )

    assert (
        "Historical support action:"
        in prompt
    )

    # Case-insensitive test because the safety instruction
    # intentionally uses uppercase "NEVER".
    assert (
        "never invent"
        in prompt.lower()
    )

    print(
        "PASS | Prompt construction"
    )

    # --------------------------------------------------------
    # Sensitive instruction presence
    # --------------------------------------------------------

    prompt_lower = prompt.lower()

    assert (
        "order status"
        in prompt_lower
    )

    assert (
        "refund"
        in prompt_lower
    )

    assert (
        "payment status"
        in prompt_lower
    )

    assert (
        "case ids"
        in prompt_lower
    )

    print(
        "PASS | Grounding safety instructions"
    )

    # --------------------------------------------------------
    # Empty query validation
    # --------------------------------------------------------

    try:

        build_generation_prompt(
            customer_query="",
            evidence=evidence,
        )

        raise AssertionError(
            "Empty query should raise ValueError."
        )

    except ValueError:
        pass

    print(
        "PASS | Empty-query validation"
    )

    # --------------------------------------------------------
    # DataFrame context test
    # --------------------------------------------------------

    import pandas as pd

    retrieval_df = pd.DataFrame(
        [
            {
                "case_id": "CASE-001",
                "customer_problem": (
                    "Package is delayed."
                ),
                "support_action": (
                    "Requested tracking information."
                ),
                "resolution_type": "ORDER_DELIVERY",
                "evidence_quality": "HIGH",
                "ce_score": 0.99,
            }
        ]
    )

    retrieval_result = {
        "query": "Where is my package?",
        "bm25_results": pd.DataFrame(),
        "bge_results": pd.DataFrame(),
        "rrf_results": pd.DataFrame(),
        "final_evidence": retrieval_df,
    }

    context = build_generation_context(
        customer_query="Where is my package?",
        retrieval_result=retrieval_result,
    )

    assert (
        context["customer_query"]
        == "Where is my package?"
    )

    assert len(
        context["evidence"]
    ) == 1

    assert (
        "Package is delayed."
        in context["prompt"]
    )

    print(
        "PASS | Retrieval-to-generation context"
    )

    # --------------------------------------------------------
    # Output-only instruction
    # --------------------------------------------------------

    assert (
        "Output ONLY the customer-facing response."
        in context["prompt"]
    )

    print(
        "PASS | Customer-facing output constraint"
    )

    print("=" * 100)
    print("GENERATION PROMPT BUILDER TEST: PASS")
    print("=" * 100)