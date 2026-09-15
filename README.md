\# Hiver Support Agent



> \*\*An Evidence-Grounded Adaptive Support Agent for safe customer-support automation\*\*



A production-oriented customer support agent that combines \*\*intent routing, risk detection, query complexity analysis, hybrid RAG, Reciprocal Rank Fusion, cross-encoder reranking, evidence-grounded generation, grounding checks, trust signals, and conservative AUTO/HUMAN decisions\*\*.



The central design principle is:



> \*\*LLM confidence is not automation permission.\*\*



The LLM generates a response draft. Independent safety and policy layers decide whether that response is safe enough to automate.



\---



\# 1. Problem Statement



A customer-support AI system should not simply generate a plausible answer.



A reliable support agent needs to answer four questions:



1\. \*\*What is the customer asking?\*\*

2\. \*\*How risky is the request?\*\*

3\. \*\*Is there useful historical evidence for this problem?\*\*

4\. \*\*Should the generated response actually be sent automatically?\*\*



This project addresses these questions through a multi-stage architecture:



```text

Customer Query

&#x20;     |

&#x20;     v

Intent + Risk + Complexity Router

&#x20;     |

&#x20;     v

Hybrid Historical Retrieval

(BM25 + BGE Vector Search)

&#x20;     |

&#x20;     v

RRF Fusion

&#x20;     |

&#x20;     v

Cross-Encoder Reranking

&#x20;     |

&#x20;     v

Historical Resolution Evidence

&#x20;     |

&#x20;     v

LLM Response Generation

&#x20;     |

&#x20;     v

Grounding + Safety Checker

&#x20;     |

&#x20;     v

Conservative Automation Policy

&#x20;     |

&#x20;     +-------------> AUTO

&#x20;     |

&#x20;     +-------------> HUMAN

2\. Key Idea



The system separates answer generation from automation authorization.



&#x20;            GENERATION

&#x20;                |

&#x20;                v

&#x20;       "What could we say?"

&#x20;                |

&#x20;                v

&#x20;       Grounding / Safety

&#x20;                |

&#x20;                v

&#x20;           DECISION

&#x20;                |

&#x20;                v

&#x20;       "Should we send it?"



This prevents a language model from being the final authority on whether a customer-facing response should be automatically sent.



3\. Key Differentiators

Evidence-Grounded Generation



The LLM receives retrieved historical customer-support evidence and is explicitly instructed not to invent unsupported information.



Hybrid Retrieval



The system combines:



BM25 lexical retrieval

BGE-large semantic retrieval

Reciprocal Rank Fusion

Cross-Encoder reranking

Explicit Risk Routing



Requests are classified into LOW, MEDIUM, and HIGH risk.



High-risk requests are never automatically handled by the final conservative policy.



Independent Grounding Gate



Generated responses are checked for:



unsupported claims

internal identifiers

sensitive information

unsupported resolution claims

unsupported customer-specific claims

Conservative Automation



The system intentionally prefers HUMAN escalation when evidence or safety is uncertain.



Evaluation-First Design



A frozen 200-case golden dataset is kept separate from the RAG corpus and training data.



4\. Dataset



The project uses the Kaggle Customer Support on Twitter dataset.



The raw dataset contains approximately 2.8 million tweets with fields including:



tweet\_id

author\_id

inbound

created\_at

text

response\_tweet\_id

in\_response\_to\_tweet\_id



The project focuses on AmazonHelp support conversations.



5\. Conversation Reconstruction



Tweets are connected using:



response\_tweet\_id

in\_response\_to\_tweet\_id



The resulting conversation graph is reconstructed into case-level objects.



Reconstructed dataset

Metric	Value

AmazonHelp support tweets	169,840

Parent tweet IDs	155,445

Response tweet IDs	100,785

Connected tweet IDs	359,705

Recovered connected tweets	358,973

Customer inbound tweets	189,132

Support tweets	169,841

Unique authors	71,669

Conversation roots	83,572

Two-sided cases	83,418



The case-level representation contains:



case\_id

root\_tweet\_id

turn\_count

customer\_turns

support\_turns

first\_time

last\_time

duration\_hours

final\_role

messages



Historical resolution evidence is transformed into:



case\_id

customer\_problem

support\_action

resolution\_type

evidence\_quality

source



PII is redacted before evidence is used by the agent.



6\. Golden Evaluation Set



A separate frozen 200-case golden evaluation set was manually reviewed and annotated.



The golden set is:



excluded from RAG retrieval

excluded from weak-label training

used only for evaluation



This prevents evaluation leakage.



Risk Distribution

Risk	Cases

LOW	106

MEDIUM	68

HIGH	26

Automation Distribution

Decision	Cases

AUTO	52

HUMAN	148

Resolution Distribution

Resolution Status	Cases

INVESTIGATION\_REQUIRED	121

RESOLVED	21

PARTIALLY\_RESOLVED	20

NOT\_APPLICABLE	17

UNRESOLVED	10

ESCALATION\_REQUIRED	6

RESOLVED\_BY\_CUSTOMER	5

Methodology



The golden set is the authoritative evaluation source.



Earlier provisional resolution buckets and weak labels are not treated as human ground truth.



7\. Intent Taxonomy



The router supports 16 intents:



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

8\. Unified Router



The unified router produces:



intent

risk\_level

complexity



The routing layer combines three signals.



Intent



Determines the customer's primary support category.



Risk



Determines how sensitive the request is.



Complexity



Determines how difficult the query is to handle safely.



Complexity levels:



SIMPLE

MEDIUM

COMPLEX



Complexity is used as a routing signal and is not treated as a direct quality metric.



9\. Risk Routing



The risk router follows a conservative approach.



HIGH



Examples:



ACCOUNT\_SECURITY

payment fraud / unauthorized charges

privacy or sensitive-data exposure



These cases are forced toward HUMAN handling.



MEDIUM



Examples:



PAYMENT\_BILLING

ACCOUNT\_ACCESS

ORDER\_CANCELLATION

MARKETPLACE\_SELLING



These cases are also kept out of automatic handling by the final conservative policy.



LOW



Normal support queries that do not match higher-risk categories.



If routing fails, the system uses a safer fallback instead of silently permitting automation.



10\. Hybrid Retrieval Architecture



The retrieval system combines lexical and semantic retrieval.



&#x20;                   Customer Query

&#x20;                        |

&#x20;            +-----------+-----------+

&#x20;            |                       |

&#x20;            v                       v

&#x20;         BM25                  BGE-large

&#x20;      Lexical Search        Semantic Search

&#x20;            |                       |

&#x20;         Top 50                   Top 50

&#x20;            |                       |

&#x20;            +-----------+-----------+

&#x20;                        |

&#x20;                        v

&#x20;                 RRF Fusion

&#x20;                        |

&#x20;                        v

&#x20;               Candidate Set

&#x20;                        |

&#x20;                        v

&#x20;             Cross-Encoder Reranker

&#x20;                        |

&#x20;                        v

&#x20;                   Top-5 Evidence

11\. BM25 Retrieval



BM25 provides lexical matching.



This is useful when customer terminology overlaps strongly with historical support terminology.



Examples:



refund

tracking

password

charge

delivery

order

12\. BGE Semantic Retrieval



The semantic retriever uses:



BAAI/bge-large-en-v1.5



with 1024-dimensional embeddings.



This helps retrieve historical cases where the customer's wording differs from the wording used in previous conversations.



13\. Reciprocal Rank Fusion



BM25 and BGE retrieval results are combined using Reciprocal Rank Fusion.



BM25 Top-50

&#x20;    |

&#x20;    +------+

&#x20;           |

&#x20;           v

&#x20;         RRF

&#x20;           ^

&#x20;           |

&#x20;    +------+

&#x20;    |

BGE Top-50



RRF allows candidates supported by either lexical or semantic retrieval to remain in the candidate pool.



14\. Cross-Encoder Reranking



The fused candidate set is reranked using:



BAAI/bge-reranker-base



The cross-encoder evaluates:



Customer Query

&#x20;     +

Historical Evidence



and produces a relevance score.



The final runtime retrieval returns the highest-ranked evidence items.



15\. Retrieval Evaluation



Intent agreement was used as a retrieval evaluation proxy.



Retriever	@1	@3	@5	@10

TF-IDF	16.0%	25.0%	31.0%	—

BM25	17.0%	29.0%	32.5%	—

BGE-large	21.5%	32.0%	36.5%	—

BM25 + BGE + RRF	22.0%	33.0%	38.0%	42.5%

RRF + Cross-Encoder	25.0%	35.0%	39.0%	44.0%



The results show progressive improvement as lexical, semantic, fusion, and reranking signals are combined.



Important evaluation note



These are retrieval intent-agreement proxies, not end-to-end response-quality metrics.



16\. Human Evidence Review



A stratified 50-case human evidence review was completed.



Evidence Assessment	Cases

RELEVANT	36

PARTIAL	13

IRRELEVANT	1



Therefore:



Useful evidence = 49 / 50

&#x20;                = 98%



For this review:



RELEVANT



The historical case contains the same or a very similar customer problem and the historical action is useful/applicable.



PARTIAL



The historical case is related but requires adaptation.



IRRELEVANT



The similarity is superficial and the historical evidence is not useful.



17\. Generation Layer



The generation layer creates the customer-facing response using retrieved historical evidence.



The prompt explicitly instructs the model to:



use historical evidence as guidance

avoid unsupported claims

ask for details when required

avoid exposing internal information

remain concise and professional

avoid copying historical responses directly



The generator is specifically prevented from inventing:



order status

delivery date

tracking information

refund amount/status

payment status

account information

investigation results

policies

unsupported actions

18\. Fail-Safe LLM Generation



The LLM wrapper uses dependency injection so the generation layer can be tested without making real API calls.



The generator returns a structured result.



Generation failure does not automatically produce an AUTO decision.



Instead:



Generation Failure

&#x20;      |

&#x20;      v

&#x20;    HUMAN



This follows the project's fail-safe principle.



19\. Grounding \& Safety Checker



The generated response is independently checked before automation.



The checker detects:



Internal information

internal case IDs

model/system terminology

retrieval information

Sensitive information

URLs

social handles

long numeric identifiers

phone-like patterns

Unsupported claims



Examples include unsupported claims that:



payment was completed

refund was processed

order was cancelled

issue was resolved

investigation was completed

customer will receive an item

customer's card was charged



The grounding result is:



GROUNDED

CONDITIONAL

UNSAFE



Only GROUNDED responses can continue to the automation decision layer.



20\. Trust Checker



The system also calculates evidence-quality signals using:



Evidence Tier

Intent Consistency

Semantic Consistency



The resulting trust levels are:



TRUSTED      >= 0.80

CONDITIONAL  >= 0.60

WEAK         < 0.60



The trust checker is intentionally treated as an evidence-quality signal, not as the sole automation classifier.



21\. Conservative Automation Policy



The final frozen automation policy is:



LOW risk

&#x20;   AND

investigation\_rate == 0

&#x20;   AND

query\_words <= 25



&#x20;           |

&#x20;           v



&#x20;          AUTO



Everything else goes to HUMAN.



Therefore:



HIGH   -> HUMAN

MEDIUM -> HUMAN



LOW + investigation evidence

&#x20;      -> HUMAN



LOW + long/complex query

&#x20;      -> HUMAN



LOW + safe/simple query

&#x20;      -> AUTO



The objective is not maximum automation.



The objective is safe automation.



22\. Automation Evaluation



The conservative policy was evaluated on the 174 non-HIGH golden cases.



Metric	Result

AUTO Precision	81.1%

AUTO Recall	57.7%

AUTO F1	67.4%

Overall Accuracy	83.3%

AUTO Coverage	21.3%

False AUTO	7

False HUMAN	22



All HIGH-risk cases remain HUMAN.



The seven false-AUTO cases demonstrate an important limitation:



A short query with no obvious investigation signal can still require investigation.



This is why the current policy is intentionally conservative.



23\. End-to-End Orchestrator



The orchestrator connects the entire system.



run(query)

&#x20;  |

&#x20;  v

route()

&#x20;  |

&#x20;  +--> intent

&#x20;  +--> risk

&#x20;  +--> complexity

&#x20;  |

&#x20;  v

retrieve()

&#x20;  |

&#x20;  v

generate()

&#x20;  |

&#x20;  v

grounding\_check()

&#x20;  |

&#x20;  v

investigation analysis

&#x20;  |

&#x20;  v

decide\_automation()

&#x20;  |

&#x20;  v

AgentResult



The resulting AgentResult contains:



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

24\. Fail-Safe Behavior



The system defaults toward HUMAN handling whenever uncertainty occurs.



Examples:



Empty query

&#x20;    -> HUMAN



Router failure

&#x20;    -> safe fallback



Retrieval failure

&#x20;    -> HUMAN



Empty evidence

&#x20;    -> HUMAN



Generation failure

&#x20;    -> HUMAN



Unsafe response

&#x20;    -> HUMAN



Non-grounded response

&#x20;    -> HUMAN



HIGH risk

&#x20;    -> HUMAN



MEDIUM risk

&#x20;    -> HUMAN



Investigation-required evidence

&#x20;    -> HUMAN



This is intentional.



In customer support automation:



False AUTO



can be substantially more damaging than:



False HUMAN

25\. Repository Structure

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

└── README.md

26\. Running the Project



From the repository root:



python src\\agent\\orchestrator.py



The current local orchestrator test suite covers:



successful pipeline

empty query

HIGH risk

MEDIUM risk

LOW grounded query

investigation evidence

unsafe generated response

empty retrieval

retriever failure

generator failure

serialization

invalid top\_k

investigation-rate calculation

empty-evidence investigation rate



Current result:



AGENT ORCHESTRATOR TEST: PASS



All 14 local tests passed.

No API or LLM calls were made.

27\. Environment Variables



The generation layer expects:



GEMINI\_API\_KEY



The API key should be supplied through the environment.



Example:



$env:GEMINI\_API\_KEY="YOUR\_KEY"



Never commit API keys to GitHub.



28\. Evaluation Artifacts



Important evaluation artifacts include:



data/evaluation/golden\_evaluation\_set.csv

data/evaluation/golden\_evaluation\_set\_frozen.csv

data/evaluation/retrieval\_benchmark.csv

data/evaluation/top5\_historical\_evidence.csv

data/evaluation/top1\_evidence\_review\_50\_final.csv

data/evaluation/human\_evidence\_review\_50\_summary.json

data/evaluation/final\_automation\_policy\_evaluation.csv



Large datasets, embeddings, checkpoints, and model artifacts should remain outside the Git repository where appropriate.



29\. Limitations



The project intentionally documents its limitations rather than presenting experimental signals as production truth.



Weak Intent Labels



Historical weak labels are rule-derived.



They are not human ground truth.



A classifier trained on weak labels achieved strong validation performance on the weak-label split, but performance on the frozen golden set was substantially lower.



Therefore weak-label validation accuracy is not presented as real-world intent accuracy.



Retrieval Evaluation



The retrieval benchmark uses intent agreement as a proxy.



It does not prove that every retrieved historical response can be directly reused.



Automation Classifier



The current automation policy is conservative and rule-based.



It achieves strong precision at limited coverage but still misses some investigation-required cases.



Runtime Retrieval Scope



The saved cross-encoder benchmark artifact was generated for the benchmark candidate set.



It should not be represented as arbitrary-query production retrieval evaluation.



30\. Future Improvements



Potential next steps include:



Expand the human-reviewed intent dataset.

Train and calibrate a stronger intent classifier.

Learn automation decisions from the golden annotations.

Add confidence calibration.

Add temporal weighting so stale historical cases receive lower priority.

Add current policy/knowledge retrieval.

Add conversation-level retrieval.

Evaluate generated responses with a larger human-reviewed dataset.

Add production observability and drift monitoring.

Add human feedback loops for incorrect AUTO decisions.

Introduce explicit model abstention when evidence is insufficient.

Add online policy/version management.

31\. Engineering Philosophy



The most important design decision is:



The agent should know when not to answer automatically.



A support agent should not optimize only for generation quality.



The complete decision process is:



Understand

&#x20;   |

&#x20;   v

Retrieve

&#x20;   |

&#x20;   v

Ground

&#x20;   |

&#x20;   v

Generate

&#x20;   |

&#x20;   v

Verify

&#x20;   |

&#x20;   v

Decide

&#x20;   |

&#x20;   +------> AUTO

&#x20;   |

&#x20;   +------> HUMAN



The architecture therefore separates:



"What should we say?"



from:



"Are we allowed to automatically say it?"

32\. Current Implementation Status

\[x] Dataset exploration

\[x] Conversation reconstruction

\[x] Golden evaluation set

\[x] Intent taxonomy

\[x] Risk router

\[x] Complexity router

\[x] BM25 retrieval

\[x] BGE vector retrieval

\[x] RRF fusion

\[x] Cross-encoder reranking

\[x] Structured historical evidence

\[x] Evidence-grounded prompt builder

\[x] Fail-safe LLM generation wrapper

\[x] Grounding checker

\[x] Trust checker

\[x] Conservative automation policy

\[x] End-to-end orchestrator

\[x] Local orchestrator tests

\[x] Retrieval evaluation

\[x] Human evidence review

\[x] Automation evaluation

\[x] GitHub repository

33\. Final Architecture

&#x20;                        CUSTOMER

&#x20;                           |

&#x20;                           v

&#x20;                +----------------------+

&#x20;                |   UNIFIED ROUTER     |

&#x20;                |----------------------|

&#x20;                | Intent               |

&#x20;                | Risk                 |

&#x20;                | Complexity           |

&#x20;                +----------+-----------+

&#x20;                           |

&#x20;                           v

&#x20;                +----------------------+

&#x20;                |   HYBRID RETRIEVER   |

&#x20;                |----------------------|

&#x20;                | BM25                 |

&#x20;                | BGE-large            |

&#x20;                +----------+-----------+

&#x20;                           |

&#x20;                           v

&#x20;                +----------------------+

&#x20;                |        RRF           |

&#x20;                +----------+-----------+

&#x20;                           |

&#x20;                           v

&#x20;                +----------------------+

&#x20;                | CROSS-ENCODER        |

&#x20;                | RERANKER             |

&#x20;                +----------+-----------+

&#x20;                           |

&#x20;                           v

&#x20;                +----------------------+

&#x20;                | HISTORICAL EVIDENCE  |

&#x20;                +----------+-----------+

&#x20;                           |

&#x20;                           v

&#x20;                +----------------------+

&#x20;                |   LLM GENERATOR      |

&#x20;                +----------+-----------+

&#x20;                           |

&#x20;                           v

&#x20;                +----------------------+

&#x20;                | GROUNDING + SAFETY   |

&#x20;                +----------+-----------+

&#x20;                           |

&#x20;                 +---------+---------+

&#x20;                 |                   |

&#x20;             GROUNDED          UNSAFE / UNCERTAIN

&#x20;                 |                   |

&#x20;                 v                   v

&#x20;       +----------------+        HUMAN

&#x20;       | AUTOMATION     |

&#x20;       | POLICY         |

&#x20;       +-------+--------+

&#x20;               |

&#x20;         +-----+-----+

&#x20;         |           |

&#x20;        AUTO       HUMAN

&#x20;         |           |

&#x20;         v           v

&#x20;      RESPONSE    ESCALATION

34\. Core Principle

&#x20;                LLM

&#x20;                 |

&#x20;             generates

&#x20;                 |

&#x20;                 v

&#x20;             DRAFT

&#x20;                 |

&#x20;                 v

&#x20;         SAFETY / GROUNDING

&#x20;                 |

&#x20;                 v

&#x20;            POLICY GATE

&#x20;                 |

&#x20;         +-------+-------+

&#x20;         |               |

&#x20;        AUTO           HUMAN



The LLM drafts.

Evidence grounds.

Safety verifies.

Policy decides.



That is the core architecture of the Hiver Support Agent.

