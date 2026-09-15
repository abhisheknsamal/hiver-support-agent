\# Hiver Support Agent — System Architecture



\## 1. Architecture Overview



The Hiver Support Agent is designed as a safety-first, evidence-grounded customer-support pipeline.



The architecture separates five major responsibilities:



1\. \*\*Understanding\*\* — intent, risk, and complexity routing.

2\. \*\*Retrieval\*\* — finding relevant historical support evidence.

3\. \*\*Generation\*\* — drafting a customer-facing response.

4\. \*\*Verification\*\* — checking grounding and safety.

5\. \*\*Decision\*\* — deciding AUTO vs HUMAN.



```mermaid

flowchart TD

&#x20;   A\[Customer Query]



&#x20;   A --> B\[Unified Router]



&#x20;   B --> B1\[Intent]

&#x20;   B --> B2\[Risk]

&#x20;   B --> B3\[Complexity]



&#x20;   B --> C\[Hybrid Retrieval]



&#x20;   C --> C1\[BM25]

&#x20;   C --> C2\[BGE-large Vector Search]



&#x20;   C1 --> D\[RRF Fusion]

&#x20;   C2 --> D



&#x20;   D --> E\[Cross-Encoder Reranker]



&#x20;   E --> F\[Historical Resolution Evidence]



&#x20;   F --> G\[LLM Response Generator]



&#x20;   G --> H\[Grounding \& Safety Checker]



&#x20;   H -->|GROUNDED| I\[Automation Policy]

&#x20;   H -->|CONDITIONAL / UNSAFE| J\[HUMAN]



&#x20;   I -->|AUTO| K\[Customer Response]

&#x20;   I -->|HUMAN| J

2\. Design Philosophy



The central design principle is:



LLM confidence is not automation permission.



The LLM is responsible for generating a response draft.



It is not responsible for deciding whether that response should be automatically sent.



The decision is made by independent safety and policy layers.



&#x20;                LLM

&#x20;                 |

&#x20;                 v

&#x20;            Draft Response

&#x20;                 |

&#x20;                 v

&#x20;       Grounding / Safety Check

&#x20;                 |

&#x20;                 v

&#x20;         Automation Policy

&#x20;                 |

&#x20;          +------+------+

&#x20;          |             |

&#x20;         AUTO         HUMAN



This separation reduces the chance that a fluent but unsupported response is automatically delivered to a customer.



3\. End-to-End Runtime Flow

4\. Layered Architecture



The system can be viewed as six logical layers.



┌─────────────────────────────────────────────┐

│                 CUSTOMER                    │

└──────────────────────┬──────────────────────┘

&#x20;                      │

&#x20;                      v

┌─────────────────────────────────────────────┐

│              1. ROUTING LAYER               │

│  Intent + Risk + Complexity                 │

└──────────────────────┬──────────────────────┘

&#x20;                      │

&#x20;                      v

┌─────────────────────────────────────────────┐

│             2. RETRIEVAL LAYER              │

│  BM25 + BGE + RRF + Cross-Encoder           │

└──────────────────────┬──────────────────────┘

&#x20;                      │

&#x20;                      v

┌─────────────────────────────────────────────┐

│               3. EVIDENCE                   │

│  Historical Customer → Support Cases        │

└──────────────────────┬──────────────────────┘

&#x20;                      │

&#x20;                      v

┌─────────────────────────────────────────────┐

│             4. GENERATION                   │

│  Evidence-grounded LLM response             │

└──────────────────────┬──────────────────────┘

&#x20;                      │

&#x20;                      v

┌─────────────────────────────────────────────┐

│          5. SAFETY / VERIFICATION           │

│  Grounding + Trust + Unsupported Claims     │

└──────────────────────┬──────────────────────┘

&#x20;                      │

&#x20;                      v

┌─────────────────────────────────────────────┐

│              6. DECISION                    │

│             AUTO or HUMAN                   │

└─────────────────────────────────────────────┘

5\. Routing Architecture



The unified router produces three outputs:



Intent

Risk

Complexity

Intent



The intent layer categorizes the customer's request.



Supported intents:



ORDER\_DELIVERY

ORDER\_TRACKING

PAYMENT\_BILLING

RETURNS\_REFUNDS

SELLER\_AUTHENTICITY

ACCOUNT\_ACCESS

ACCOUNT\_SECURITY

DEVICE\_TECHNICAL

DIGITAL\_CONTENT

PURCHASE\_CONTROL

ORDER\_CANCELLATION

PREORDER\_DELIVERY

GENERAL\_SUPPORT

GENERAL\_SOCIAL

MARKETPLACE\_SELLING

ORDER\_PURCHASE

Risk



Risk determines how conservatively the system should behave.



HIGH

MEDIUM

LOW



Examples of HIGH-risk areas include:



Account security

Payment fraud / unauthorized charges

Sensitive-data exposure



The final policy forces HIGH and MEDIUM cases to HUMAN.



Complexity



Queries are classified as:



SIMPLE

MEDIUM

COMPLEX



Complexity is used as an additional safety/routing signal.



6\. Retrieval Architecture



The retrieval system uses multiple complementary signals.



7\. BM25



BM25 performs lexical retrieval.



It is useful when important terms in the customer query overlap with historical support conversations.



Examples:



refund

tracking

password

charge

delivery

order



BM25 provides strong exact/term-based matching.



8\. BGE Semantic Retrieval



The semantic retrieval layer uses:



BAAI/bge-large-en-v1.5



Embeddings are 1024-dimensional.



The semantic retriever allows the system to identify historical conversations with similar meaning even when the customer's wording differs.



Customer:

"Where is the package I ordered?"



Historical:

"My delivery hasn't arrived yet."



Lexical similarity may be limited.



Semantic similarity can still be high.

9\. Reciprocal Rank Fusion



BM25 and BGE rankings are combined using Reciprocal Rank Fusion.



BM25 Top-50

&#x20;    |

&#x20;    |

&#x20;    +----------+

&#x20;               |

&#x20;               v

&#x20;            RRF Fusion

&#x20;               ^

&#x20;               |

&#x20;    +----------+

&#x20;    |

BGE Top-50



RRF allows both retrieval strategies to contribute candidates without requiring their raw scores to be directly comparable.



10\. Cross-Encoder Reranking



After RRF, the candidate set is reranked with:



BAAI/bge-reranker-base



The cross-encoder evaluates:



Customer Query

&#x20;     +

Historical Evidence



and produces a relevance score.



The final evidence returned to the generation layer is the highest-ranked historical evidence.



Query

&#x20; |

&#x20; v

BM25 + BGE

&#x20; |

&#x20; v

RRF

&#x20; |

&#x20; v

Candidate Pool

&#x20; |

&#x20; v

Cross-Encoder

&#x20; |

&#x20; v

Top-5 Evidence

11\. Historical Evidence Representation



Raw conversations are converted into structured resolution objects.



case\_id

customer\_problem

support\_action

resolution\_type

evidence\_quality

source



The important distinction is that the generator receives historical support evidence, not merely arbitrary similar text.



Example:



Customer Problem:

Customer asks where their package is.



Support Action:

Provided tracking guidance.



Resolution Type:

RESOLVED



Evidence Quality:

HIGH

12\. Generation Architecture



The generation layer receives:



Customer Query

\+

Historical Evidence

\+

Relevant Metadata



The prompt explicitly instructs the model to use evidence as guidance and avoid unsupported claims.



13\. Generation Safety Constraints



The model must not invent:



Order status

Delivery dates

Tracking information

Refund amounts

Refund status

Payment status

Account information

Investigation results

Policies

Unsupported actions



The model must also avoid exposing:



Internal case IDs

Retrieval scores

Model names

Internal system terms

Sensitive identifiers



The generated output is customer-facing only.



14\. Grounding Architecture



The generated response passes through an independent grounding checker.



The checker produces:



GROUNDED

CONDITIONAL

UNSAFE



Only GROUNDED responses continue toward automatic handling.



15\. Trust Architecture



The trust checker provides an evidence-quality signal.



Evidence Tier

&#x20;     +

Intent Consistency

&#x20;     +

Semantic Consistency

&#x20;     |

&#x20;     v

Trust Signal



Trust levels:



TRUSTED      >= 0.80

CONDITIONAL  >= 0.60

WEAK         < 0.60



Trust is intentionally not the sole automation classifier.



16\. Automation Decision Architecture



The automation policy is deliberately conservative.



The frozen policy is:



LOW risk

AND

investigation\_rate == 0

AND

query\_words <= 25



&#x20;       ↓



&#x20;     AUTO



Otherwise:



HUMAN

17\. Safety Boundary



The decision hierarchy is:



&#x20;                Risk

&#x20;                 |

&#x20;         +-------+-------+

&#x20;         |               |

&#x20;      HIGH/MEDIUM        LOW

&#x20;         |               |

&#x20;         v               v

&#x20;       HUMAN       Grounding Check

&#x20;                         |

&#x20;                 +-------+-------+

&#x20;                 |               |

&#x20;             Unsafe/Other     GROUNDED

&#x20;                 |               |

&#x20;                 v               v

&#x20;               HUMAN       Investigation?

&#x20;                                 |

&#x20;                        +--------+--------+

&#x20;                        |                 |

&#x20;                       YES                NO

&#x20;                        |                 |

&#x20;                        v                 v

&#x20;                      HUMAN        Query Length

&#x20;                                        |

&#x20;                                 +------+------+

&#x20;                                 |             |

&#x20;                               >25           <=25

&#x20;                                 |             |

&#x20;                                 v             v

&#x20;                               HUMAN         AUTO



This makes the safety boundaries explicit.



18\. Fail-Safe Architecture



Every major component has a safer failure path.



The system does not interpret failure as permission to automate.



19\. End-to-End Orchestrator



The repository's orchestrator connects all layers.



run(query)

&#x20;  |

&#x20;  +--> route()

&#x20;  |

&#x20;  +--> retrieve()

&#x20;  |

&#x20;  +--> generate()

&#x20;  |

&#x20;  +--> grounding\_check()

&#x20;  |

&#x20;  +--> investigation analysis

&#x20;  |

&#x20;  +--> decide\_automation()

&#x20;  |

&#x20;  v

AgentResult



The result contains:



query

intent

risk\_level

complexity

evidence

draft\_response

grounding\_status

grounding\_reason

automation\_decision

decision\_reason

success

error

20\. Evaluation Architecture



Evaluation data is kept separate from training and retrieval data.



&#x20;                   Historical Dataset

&#x20;                          |

&#x20;                          v

&#x20;                 Conversation Cases

&#x20;                          |

&#x20;             +------------+------------+

&#x20;             |                         |

&#x20;             v                         v

&#x20;      Training / RAG             Golden Set

&#x20;             |                         |

&#x20;             |                    200 cases

&#x20;             |                         |

&#x20;             |                    Evaluation

&#x20;             |                         |

&#x20;             +------------X------------+

&#x20;                      NO LEAKAGE



The 200-case golden set is not used as retrieval evidence or training data.



21\. Retrieval Evaluation Flow

Golden Query

&#x20;    |

&#x20;    v

Candidate Retrieval

&#x20;    |

&#x20;    +--> BM25

&#x20;    +--> BGE

&#x20;    |

&#x20;    v

RRF

&#x20;    |

&#x20;    v

Cross-Encoder

&#x20;    |

&#x20;    v

Ranked Historical Cases

&#x20;    |

&#x20;    v

Intent Agreement Evaluation



This provides a controlled benchmark for comparing retrieval strategies.



22\. Human Evidence Evaluation



A separate 50-case review evaluates whether retrieved evidence is actually useful.



Retrieved Evidence

&#x20;       |

&#x20;       v

Human Review

&#x20;       |

&#x20;       +--> RELEVANT

&#x20;       |

&#x20;       +--> PARTIAL

&#x20;       |

&#x20;       +--> IRRELEVANT



Results:



RELEVANT     = 36

PARTIAL      = 13

IRRELEVANT   = 1



Useful Evidence = 49 / 50 = 98%

23\. Automation Evaluation



The automation policy is evaluated against the golden annotations.



Golden Automation Label

&#x20;         |

&#x20;         v

&#x20;     AUTO / HUMAN

&#x20;         ^

&#x20;         |

Predicted Automation



The evaluation is performed independently from the generation layer.



This allows automation quality to be measured separately from language-model fluency.



24\. Component Responsibilities

Component	Responsibility

intent\_router.py	Intent classification/fallback routing

risk\_router.py	Risk classification

unified\_router.py	Unified intent/risk/complexity output

corpus.py	Structured corpus loading and validation

bm25\_retriever.py	Lexical retrieval

vector\_retriever.py	BGE semantic retrieval

rrf.py	Reciprocal Rank Fusion

reranker.py	Cross-encoder reranking

hybrid\_retriever.py	End-to-end retrieval pipeline

prompt\_builder.py	Evidence-grounded generation prompt

llm\_generator.py	Fail-safe LLM interface

grounding\_checker.py	Grounding and safety validation

decision\_policy.py	Conservative AUTO/HUMAN decision

orchestrator.py	End-to-end agent coordination

25\. Repository Architecture

hiver-support-agent/

│

├── notebooks/

│   ├── 01\_amazon\_dataset\_exploration.ipynb

│   └── 02\_agent\_evaluation\_pipeline.ipynb

│

├── src/

│   │

│   ├── agent/

│   │   └── orchestrator.py

│   │

│   ├── router/

│   │   ├── intent\_router.py

│   │   ├── risk\_router.py

│   │   └── unified\_router.py

│   │

│   ├── retrieval/

│   │   ├── corpus.py

│   │   ├── bm25\_retriever.py

│   │   ├── vector\_retriever.py

│   │   ├── rrf.py

│   │   ├── reranker.py

│   │   └── hybrid\_retriever.py

│   │

│   ├── generation/

│   │   ├── prompt\_builder.py

│   │   └── llm\_generator.py

│   │

│   └── safety/

│       ├── grounding\_checker.py

│       └── decision\_policy.py

│

├── .gitignore

├── README.md

└── ARCHITECTURE.md

26\. Data Flow



The complete data flow is:



Raw Tweets

&#x20;   |

&#x20;   v

Conversation Reconstruction

&#x20;   |

&#x20;   v

Case-Level Conversations

&#x20;   |

&#x20;   v

Structured Historical Evidence

&#x20;   |

&#x20;   +-----------------------+

&#x20;   |                       |

&#x20;   v                       v

RAG Corpus             Evaluation Set

&#x20;   |                       |

&#x20;   v                       v

Retrieval              Golden Evaluation

&#x20;   |

&#x20;   v

Evidence

&#x20;   |

&#x20;   v

Generation

&#x20;   |

&#x20;   v

Grounding

&#x20;   |

&#x20;   v

Automation Decision

27\. Security Boundaries



The architecture intentionally keeps several boundaries between internal data and customer-facing output.



Historical Data

&#x20;     |

&#x20;     v

PII Redaction

&#x20;     |

&#x20;     v

Structured Evidence

&#x20;     |

&#x20;     v

LLM Context

&#x20;     |

&#x20;     v

Customer Response



The generation prompt prevents internal retrieval information from being exposed.



The grounding checker provides a second independent boundary.



28\. Why Hybrid RAG?



No single retrieval strategy is ideal for customer support.



BM25



Strong for exact terminology.



BGE



Strong for semantic similarity.



RRF



Combines independent ranking signals.



Cross-Encoder



Provides a more detailed relevance judgment over the candidate set.



Therefore:



Lexical Recall

&#x20;     +

Semantic Recall

&#x20;     +

Rank Fusion

&#x20;     +

Deep Reranking

&#x20;     |

&#x20;     v

Better Evidence Selection

29\. Why Historical Resolution Evidence?



A generic document RAG system retrieves informational documents.



Customer support is different.



The most useful evidence is often:



Customer Problem

&#x20;     +

Support Action

&#x20;     +

Resolution Outcome



Therefore the corpus is structured around historical support cases and actions, rather than only isolated tweets.



This makes retrieved evidence more actionable for response generation.



30\. Why Conservative Automation?



The golden evaluation contains a large number of cases requiring investigation or human handling.



The policy therefore intentionally prioritizes:



Safety > Automation Coverage



A system that automatically handles fewer cases with higher precision is preferable to a system that confidently sends unsupported answers.



31\. Observability Opportunities



The architecture provides natural monitoring points:



Router

&#x20; |

&#x20; +--> Intent distribution

&#x20; +--> Risk distribution

&#x20; +--> Complexity distribution



Retriever

&#x20; |

&#x20; +--> Retrieval scores

&#x20; +--> Candidate counts

&#x20; +--> Evidence quality



Generator

&#x20; |

&#x20; +--> Generation failures

&#x20; +--> Response length

&#x20; +--> Grounding violations



Decision

&#x20; |

&#x20; +--> AUTO rate

&#x20; +--> HUMAN rate

&#x20; +--> False AUTO

&#x20; +--> False HUMAN



These signals can later support production monitoring and model/data drift detection.



32\. Future Architecture



A future production version could add:



&#x20;                   Customer

&#x20;                      |

&#x20;                      v

&#x20;                Intent / Risk

&#x20;                      |

&#x20;                      v

&#x20;             Query Understanding

&#x20;                      |

&#x20;         +------------+-------------+

&#x20;         |                          |

&#x20;         v                          v

&#x20;  Historical RAG              Current Policy RAG

&#x20;         |                          |

&#x20;         +------------+-------------+

&#x20;                      |

&#x20;                      v

&#x20;               Evidence Fusion

&#x20;                      |

&#x20;                      v

&#x20;                LLM Generator

&#x20;                      |

&#x20;                      v

&#x20;             Safety / Grounding

&#x20;                      |

&#x20;                      v

&#x20;             Decision Policy

&#x20;                      |

&#x20;                +-----+-----+

&#x20;                |           |

&#x20;               AUTO       HUMAN

&#x20;                |           |

&#x20;                v           v

&#x20;            Response    Escalation

&#x20;                            |

&#x20;                            v

&#x20;                      Human Feedback

&#x20;                            |

&#x20;                            v

&#x20;                      Model / Policy

&#x20;                        Improvement

33\. Final System Principle



The complete architecture can be summarized as:



UNDERSTAND

&#x20;   ↓

RETRIEVE

&#x20;   ↓

GROUND

&#x20;   ↓

GENERATE

&#x20;   ↓

VERIFY

&#x20;   ↓

DECIDE

&#x20;   ↓

AUTO / HUMAN



Or, more simply:



The router decides what the request is.

The retriever finds what has worked before.

The LLM drafts the response.

The safety layer checks the response.

The policy decides whether automation is allowed.



The LLM drafts. Evidence grounds. Safety verifies. Policy decides.

