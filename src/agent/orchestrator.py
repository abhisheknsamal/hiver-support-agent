"""
Hiver Support Agent
===================

End-to-end orchestration layer for the Hiver Support Agent.

Pipeline:

    Customer Query
          |
          v
    Unified Router
          |
          +--> Intent
          +--> Risk
          +--> Complexity
          |
          v
    Hybrid Retrieval
          |
          v
    Historical Evidence
          |
          v
    LLM Generation
          |
          v
    Grounding / Safety Check
          |
          v
    Automation Decision Policy
          |
          +------+
          |      |
         AUTO   HUMAN

Design goals:
    - Dependency injection
    - No API/LLM calls during local tests
    - Fail-safe behavior
    - Structured result
    - No silent automation on failures
    - Compatible with direct PowerShell execution
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ============================================================================
# PROJECT PATH
# ============================================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)

SRC_ROOT = os.path.join(
    PROJECT_ROOT,
    "src",
)

# Repository root is required for:
#
#     from src.router...
#
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Existing modules in this repository currently use some direct sibling
# imports such as:
#
#     from intent_router import ...
#
# Add their directories so direct execution remains compatible.
SOURCE_DIRECTORIES = [
    os.path.join(SRC_ROOT, "router"),
    os.path.join(SRC_ROOT, "retrieval"),
    os.path.join(SRC_ROOT, "generation"),
    os.path.join(SRC_ROOT, "safety"),
]

for directory in SOURCE_DIRECTORIES:
    if directory not in sys.path:
        sys.path.insert(0, directory)


# ============================================================================
# IMPORTS
# ============================================================================

from src.router.unified_router import run_unified_router
from src.safety.grounding_checker import grounding_check
from src.safety.decision_policy import decide_automation


# ============================================================================
# CONSTANTS
# ============================================================================

AUTO = "AUTO"
HUMAN = "HUMAN"

GROUNDED = "GROUNDED"
CONDITIONAL = "CONDITIONAL"
UNSAFE = "UNSAFE"


# ============================================================================
# RESULT OBJECT
# ============================================================================

@dataclass
class AgentResult:
    """Final structured result returned by the orchestrator."""

    query: str
    intent: str
    risk_level: str
    complexity: str

    evidence: List[Dict[str, Any]] = field(
        default_factory=list
    )

    draft_response: str = ""

    grounding_status: str = ""
    grounding_reason: str = ""

    automation_decision: str = HUMAN
    decision_reason: str = ""

    success: bool = True
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the result into a serializable dictionary."""

        return {
            "query": self.query,
            "intent": self.intent,
            "risk_level": self.risk_level,
            "complexity": self.complexity,
            "evidence": self.evidence,
            "draft_response": self.draft_response,
            "grounding_status": self.grounding_status,
            "grounding_reason": self.grounding_reason,
            "automation_decision": self.automation_decision,
            "decision_reason": self.decision_reason,
            "success": self.success,
            "error": self.error,
        }


# ============================================================================
# ORCHESTRATOR
# ============================================================================

class SupportAgent:
    """
    Main Hiver Support Agent orchestrator.

    Dependencies are injected:

        retriever
            Historical evidence retriever.

        generator
            Customer-facing response generator.

    This allows the orchestrator to be tested without loading
    embedding models, rerankers, or calling an LLM API.
    """

    def __init__(
        self,
        retriever: Any,
        generator: Any,
    ) -> None:

        if retriever is None:
            raise ValueError(
                "retriever cannot be None"
            )

        if generator is None:
            raise ValueError(
                "generator cannot be None"
            )

        self.retriever = retriever
        self.generator = generator

    # ========================================================================
    # ROUTER
    # ========================================================================

    def route(
        self,
        query: str,
    ) -> Dict[str, Any]:
        """Run the unified router."""

        result = run_unified_router(query)

        if result is None:
            raise RuntimeError(
                "Router returned no result"
            )

        if not isinstance(result, dict):
            raise TypeError(
                "Router must return a dictionary, "
                f"got {type(result).__name__}"
            )

        return result

    # ========================================================================
    # RETRIEVAL
    # ========================================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve historical support evidence.

        Expected interface:

            retriever.search(query, top_k=5)
        """

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        try:
            results = self.retriever.search(
                query,
                top_k=top_k,
            )
        except TypeError:
            results = self.retriever.search(
                query,
                top_k,
            )

        if results is None:
            return []

        if not isinstance(results, list):
            results = list(results)

        # Runtime-safe evidence only.
        cleaned_results = []

        for item in results[:top_k]:

            if isinstance(item, dict):
                cleaned_results.append(item)

        return cleaned_results

    # ========================================================================
    # GENERATION
    # ========================================================================

    def generate(
        self,
        query: str,
        route_result: Dict[str, Any],
        evidence: List[Dict[str, Any]],
    ) -> str:
        """
        Generate the customer-facing response.

        Expected interface:

            generator.generate(
                query=query,
                evidence=evidence,
                metadata=metadata,
            )
        """

        metadata = {
            "intent": route_result.get(
                "intent",
                "GENERAL_SUPPORT",
            ),
            "risk_level": route_result.get(
                "risk_level",
                "MEDIUM",
            ),
            "complexity": route_result.get(
                "complexity",
                "MEDIUM",
            ),
        }

        try:

            result = self.generator.generate(
                query=query,
                evidence=evidence,
                metadata=metadata,
            )

        except TypeError:

            # Compatibility with simple test doubles.
            result = self.generator.generate(
                query,
                evidence,
                metadata,
            )

        # GenerationResult-style object.
        if hasattr(result, "text"):
            return str(
                result.text or ""
            ).strip()

        # Dictionary-style result.
        if isinstance(result, dict):
            return str(
                result.get(
                    "text",
                    result.get(
                        "response",
                        "",
                    ),
                )
                or ""
            ).strip()

        # Plain string.
        return str(
            result or ""
        ).strip()

    # ========================================================================
    # GROUNDING
    # ========================================================================

    def check_grounding(
        self,
        query: str,
        response: str,
        evidence: List[Dict[str, Any]],
        intent: str,
    ) -> Any:
        """
        Run the grounding and safety checker.

        IMPORTANT:

        grounding_check() expects:

            response
            query
            evidence_items
            historical_intent
        """

        return grounding_check(
            response=response,
            query=query,
            evidence_items=evidence,
            historical_intent=intent,
        )

    # ========================================================================
    # DECISION
    # ========================================================================

    def decide(
        self,
        route_result: Dict[str, Any],
        query: str,
        grounding_status: str,
        investigation_rate: float,
    ) -> Any:
        """
        Apply grounding safety gate followed by automation policy.

        Anything other than GROUNDED is forced to HUMAN.
        """

        risk_level = route_result.get(
            "risk_level",
            "MEDIUM",
        )

        # --------------------------------------------------------------------
        # HARD SAFETY GATE
        # --------------------------------------------------------------------

        if grounding_status != GROUNDED:

            class SafetyDecision:
                decision = HUMAN
                reason = (
                    "GROUNDING_CHECK_REQUIRES_HUMAN"
                )

            return SafetyDecision()

        # --------------------------------------------------------------------
        # CONSERVATIVE AUTOMATION POLICY
        # --------------------------------------------------------------------

        return decide_automation(
            risk_level=risk_level,
            investigation_rate=investigation_rate,
            query=query,
        )

    # ========================================================================
    # MAIN PIPELINE
    # ========================================================================

    def run(
        self,
        query: str,
        top_k: int = 5,
    ) -> AgentResult:
        """
        Execute the complete support-agent pipeline.

        Router
            ↓
        Hybrid Retrieval
            ↓
        Generation
            ↓
        Grounding / Safety
            ↓
        Investigation Assessment
            ↓
        Automation Decision
        """

        # --------------------------------------------------------------------
        # INPUT VALIDATION
        # --------------------------------------------------------------------

        if not isinstance(
            query,
            str,
        ) or not query.strip():

            return AgentResult(
                query=(
                    query
                    if isinstance(query, str)
                    else ""
                ),
                intent="GENERAL_SUPPORT",
                risk_level="MEDIUM",
                complexity="SIMPLE",
                automation_decision=HUMAN,
                decision_reason=(
                    "EMPTY_QUERY_REQUIRES_HUMAN"
                ),
                success=False,
                error=(
                    "Query must be a non-empty string."
                ),
            )

        query = query.strip()

        # --------------------------------------------------------------------
        # TOP-K VALIDATION
        # --------------------------------------------------------------------

        if (
            not isinstance(top_k, int)
            or top_k <= 0
        ):

            return AgentResult(
                query=query,
                intent="GENERAL_SUPPORT",
                risk_level="MEDIUM",
                complexity="MEDIUM",
                automation_decision=HUMAN,
                decision_reason=(
                    "INVALID_TOP_K_FAIL_SAFE"
                ),
                success=False,
                error=(
                    "top_k must be a positive integer."
                ),
            )

        try:

            # ================================================================
            # STEP 1 — ROUTER
            # ================================================================

            route_result = self.route(query)

            intent = route_result.get(
                "intent",
                "GENERAL_SUPPORT",
            )

            risk_level = route_result.get(
                "risk_level",
                "MEDIUM",
            )

            complexity = route_result.get(
                "complexity",
                "MEDIUM",
            )

            # ================================================================
            # STEP 2 — RETRIEVAL
            # ================================================================

            evidence = self.retrieve(
                query=query,
                top_k=top_k,
            )

            # ================================================================
            # STEP 3 — GENERATION
            # ================================================================

            draft_response = self.generate(
                query=query,
                route_result=route_result,
                evidence=evidence,
            )

            # ================================================================
            # STEP 4 — GROUNDING / SAFETY
            # ================================================================

            grounding_result = self.check_grounding(
                query=query,
                response=draft_response,
                evidence=evidence,
                intent=intent,
            )

            # IMPORTANT:
            #
            # GroundingResult uses:
            #
            #     grounding_decision
            #     violations
            #     warnings
            #
            # NOT:
            #
            #     status
            #     reason
            #
            grounding_status = getattr(
                grounding_result,
                "grounding_decision",
                UNSAFE,
            )

            violations = getattr(
                grounding_result,
                "violations",
                [],
            )

            warnings = getattr(
                grounding_result,
                "warnings",
                [],
            )

            if violations:
                grounding_reason = "; ".join(
                    str(item)
                    for item in violations
                )
            elif warnings:
                grounding_reason = "; ".join(
                    str(item)
                    for item in warnings
                )
            else:
                grounding_reason = ""

            # ================================================================
            # STEP 5 — INVESTIGATION RATE
            # ================================================================

            investigation_rate = (
                self._calculate_investigation_rate(
                    evidence=evidence,
                )
            )

            # ================================================================
            # STEP 6 — AUTOMATION DECISION
            # ================================================================

            decision_result = self.decide(
                route_result=route_result,
                query=query,
                grounding_status=grounding_status,
                investigation_rate=investigation_rate,
            )

            automation_decision = getattr(
                decision_result,
                "decision",
                HUMAN,
            )

            decision_reason = getattr(
                decision_result,
                "reason",
                "FAIL_SAFE",
            )

            # ================================================================
            # FINAL RESULT
            # ================================================================

            return AgentResult(
                query=query,
                intent=intent,
                risk_level=risk_level,
                complexity=complexity,
                evidence=evidence,
                draft_response=draft_response,
                grounding_status=grounding_status,
                grounding_reason=grounding_reason,
                automation_decision=(
                    automation_decision
                ),
                decision_reason=decision_reason,
                success=True,
                error=None,
            )

        except Exception as exc:

            # ================================================================
            # GLOBAL FAIL-SAFE
            # ================================================================

            return AgentResult(
                query=query,
                intent="GENERAL_SUPPORT",
                risk_level="MEDIUM",
                complexity="MEDIUM",
                evidence=[],
                draft_response="",
                grounding_status=UNSAFE,
                grounding_reason=(
                    "ORCHESTRATOR_FAILURE"
                ),
                automation_decision=HUMAN,
                decision_reason=(
                    "ORCHESTRATOR_FAILURE_FAIL_SAFE"
                ),
                success=False,
                error=(
                    f"{type(exc).__name__}: {exc}"
                ),
            )

    # ========================================================================
    # INVESTIGATION RATE
    # ========================================================================

    @staticmethod
    def _calculate_investigation_rate(
        evidence: List[Dict[str, Any]],
    ) -> float:
        """
        Calculate the fraction of retrieved evidence that requires
        investigation or escalation.

        Empty evidence is conservatively treated as 100% investigation.
        """

        if not evidence:
            return 1.0

        investigation_count = 0

        for item in evidence:

            if not isinstance(item, dict):

                investigation_count += 1
                continue

            resolution_type = str(
                item.get(
                    "resolution_type",
                    "",
                )
            ).strip().upper()

            evidence_quality = str(
                item.get(
                    "evidence_quality",
                    "",
                )
            ).strip().upper()

            if (
                "INVESTIGATION"
                in resolution_type
                or "ESCALATION"
                in resolution_type
                or evidence_quality
                in {
                    "WEAK",
                    "IRRELEVANT",
                }
            ):

                investigation_count += 1

        return (
            investigation_count
            / len(evidence)
        )


# ============================================================================
# LOCAL TEST DOUBLES
# ============================================================================

class _FakeRetriever:
    """Fake historical retriever for local tests."""

    def __init__(
        self,
        results: Optional[
            List[Dict[str, Any]]
        ] = None,
    ):

        self.results = (
            results
            if results is not None
            else [
                {
                    "case_id": "AMZ-0001",
                    "customer_problem": (
                        "Customer asks where their package is."
                    ),
                    "support_action": (
                        "Provided tracking guidance."
                    ),
                    "resolution_type": "RESOLVED",
                    "evidence_quality": "HIGH",
                    "source": "historical",
                }
            ]
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:

        return self.results[:top_k]


class _FakeGenerator:
    """Fake generator for local tests."""

    def __init__(
        self,
        response: str = (
            "Please check your tracking information."
        ),
    ):

        self.response = response

    def generate(
        self,
        query: str,
        evidence: List[Dict[str, Any]],
        metadata: Dict[str, Any],
    ) -> str:

        return self.response


# ============================================================================
# LOCAL TESTS
# ============================================================================

def _run_tests() -> None:

    # ------------------------------------------------------------------------
    # 1. Basic successful pipeline
    # ------------------------------------------------------------------------

    agent = SupportAgent(
        retriever=_FakeRetriever(),
        generator=_FakeGenerator(),
    )

    result = agent.run(
        "Where is my package?",
    )

    assert result.success is True
    assert result.query == "Where is my package?"
    assert result.intent != ""
    assert result.risk_level != ""
    assert result.complexity != ""
    assert len(result.evidence) == 1
    assert result.draft_response != ""
    assert result.grounding_status == GROUNDED

    # ------------------------------------------------------------------------
    # 2. Empty query -> HUMAN
    # ------------------------------------------------------------------------

    result = agent.run("")

    assert result.success is False
    assert result.automation_decision == HUMAN
    assert result.error is not None

    # ------------------------------------------------------------------------
    # 3. HIGH risk -> HUMAN
    # ------------------------------------------------------------------------

    class HighRiskAgent(SupportAgent):

        def route(
            self,
            query: str,
        ) -> Dict[str, Any]:

            return {
                "intent": "ACCOUNT_SECURITY",
                "risk_level": "HIGH",
                "complexity": "SIMPLE",
            }

    agent = HighRiskAgent(
        retriever=_FakeRetriever(),
        generator=_FakeGenerator(),
    )

    result = agent.run(
        "Someone hacked my account",
    )

    assert result.success is True
    assert result.risk_level == "HIGH"
    assert result.automation_decision == HUMAN

    # ------------------------------------------------------------------------
    # 4. MEDIUM risk -> HUMAN
    # ------------------------------------------------------------------------

    class MediumRiskAgent(SupportAgent):

        def route(
            self,
            query: str,
        ) -> Dict[str, Any]:

            return {
                "intent": "ACCOUNT_ACCESS",
                "risk_level": "MEDIUM",
                "complexity": "SIMPLE",
            }

    agent = MediumRiskAgent(
        retriever=_FakeRetriever(),
        generator=_FakeGenerator(),
    )

    result = agent.run(
        "I forgot my password",
    )

    assert result.success is True
    assert result.risk_level == "MEDIUM"
    assert result.automation_decision == HUMAN

    # ------------------------------------------------------------------------
    # 5. LOW risk + grounded + no investigation -> AUTO
    # ------------------------------------------------------------------------

    class LowRiskAgent(SupportAgent):

        def route(
            self,
            query: str,
        ) -> Dict[str, Any]:

            return {
                "intent": "ORDER_TRACKING",
                "risk_level": "LOW",
                "complexity": "SIMPLE",
            }

    agent = LowRiskAgent(
        retriever=_FakeRetriever(),
        generator=_FakeGenerator(),
    )

    result = agent.run(
        "Where is my package?",
    )

    assert result.success is True
    assert result.risk_level == "LOW"
    assert result.grounding_status == GROUNDED
    assert result.automation_decision == AUTO

    # ------------------------------------------------------------------------
    # 6. Investigation evidence -> HUMAN
    # ------------------------------------------------------------------------

    investigation_evidence = [
        {
            "case_id": "AMZ-0002",
            "customer_problem": (
                "Customer asks where their package is."
            ),
            "support_action": (
                "Investigated the package delivery issue."
            ),
            "resolution_type": (
                "INVESTIGATION_REQUIRED"
            ),
            "evidence_quality": "HIGH",
            "source": "historical",
        }
    ]

    class InvestigationAgent(SupportAgent):

        def route(
            self,
            query: str,
        ) -> Dict[str, Any]:

            return {
                "intent": "ORDER_DELIVERY",
                "risk_level": "LOW",
                "complexity": "SIMPLE",
            }

    agent = InvestigationAgent(
        retriever=_FakeRetriever(
            investigation_evidence
        ),
        generator=_FakeGenerator(),
    )

    result = agent.run(
        "Where is my package?",
    )

    assert result.success is True
    assert result.automation_decision == HUMAN
   

    # ------------------------------------------------------------------------
    # 7. Unsafe generated response -> HUMAN
    # ------------------------------------------------------------------------

    unsafe_generator = _FakeGenerator(
        response="Your order has been cancelled."
    )

    agent = SupportAgent(
        retriever=_FakeRetriever(),
        generator=unsafe_generator,
    )

    result = agent.run(
        "Where is my package?",
    )

    assert result.success is True
    assert result.grounding_status == UNSAFE
    assert result.automation_decision == HUMAN
    assert (
        result.decision_reason
        == "GROUNDING_CHECK_REQUIRES_HUMAN"
    )

    # ------------------------------------------------------------------------
    # 8. Empty retrieval -> HUMAN
    # ------------------------------------------------------------------------

    agent = SupportAgent(
        retriever=_FakeRetriever(
            results=[]
        ),
        generator=_FakeGenerator(),
    )

    result = agent.run(
        "Where is my package?",
    )

    assert result.success is True
    assert result.automation_decision == HUMAN

    # ------------------------------------------------------------------------
    # 9. Retriever failure -> fail-safe HUMAN
    # ------------------------------------------------------------------------

    class BrokenRetriever:

        def search(
            self,
            query: str,
            top_k: int = 5,
        ):

            raise RuntimeError(
                "retrieval failure"
            )

    agent = SupportAgent(
        retriever=BrokenRetriever(),
        generator=_FakeGenerator(),
    )

    result = agent.run(
        "Where is my package?",
    )

    assert result.success is False
    assert result.automation_decision == HUMAN
    assert result.error is not None
    assert "retrieval failure" in result.error

    # ------------------------------------------------------------------------
    # 10. Generator failure -> fail-safe HUMAN
    # ------------------------------------------------------------------------

    class BrokenGenerator:

        def generate(
            self,
            query: str,
            evidence: List[Dict[str, Any]],
            metadata: Dict[str, Any],
        ):

            raise RuntimeError(
                "generation failure"
            )

    agent = SupportAgent(
        retriever=_FakeRetriever(),
        generator=BrokenGenerator(),
    )

    result = agent.run(
        "Where is my package?",
    )

    assert result.success is False
    assert result.automation_decision == HUMAN
    assert result.error is not None
    assert "generation failure" in result.error

    # ------------------------------------------------------------------------
    # 11. Result serialization
    # ------------------------------------------------------------------------

    agent = SupportAgent(
        retriever=_FakeRetriever(),
        generator=_FakeGenerator(),
    )

    result = agent.run(
        "Where is my package?",
    )

    data = result.to_dict()

    assert isinstance(data, dict)
    assert (
        data["query"]
        == "Where is my package?"
    )
    assert "evidence" in data
    assert "automation_decision" in data
    assert "grounding_status" in data

    # ------------------------------------------------------------------------
    # 12. Invalid top_k -> HUMAN
    # ------------------------------------------------------------------------

    result = agent.run(
        "Where is my package?",
        top_k=0,
    )

    assert result.success is False
    assert result.automation_decision == HUMAN
    assert (
        result.decision_reason
        == "INVALID_TOP_K_FAIL_SAFE"
    )

    # ------------------------------------------------------------------------
    # 13. Investigation rate calculation
    # ------------------------------------------------------------------------

    evidence = [
        {
            "resolution_type": "RESOLVED",
            "evidence_quality": "HIGH",
        },
        {
            "resolution_type": (
                "INVESTIGATION_REQUIRED"
            ),
            "evidence_quality": "HIGH",
        },
    ]

    rate = (
        SupportAgent
        ._calculate_investigation_rate(
            evidence
        )
    )

    assert rate == 0.5

    # ------------------------------------------------------------------------
    # 14. Empty evidence -> investigation required
    # ------------------------------------------------------------------------

    rate = (
        SupportAgent
        ._calculate_investigation_rate([])
    )

    assert rate == 1.0

    # ------------------------------------------------------------------------
    # PASS
    # ------------------------------------------------------------------------

    print("=" * 100)
    print("AGENT ORCHESTRATOR TEST: PASS")
    print("=" * 100)
    print("All 14 local tests passed.")
    print("No API or LLM calls were made.")


if __name__ == "__main__":
    _run_tests()