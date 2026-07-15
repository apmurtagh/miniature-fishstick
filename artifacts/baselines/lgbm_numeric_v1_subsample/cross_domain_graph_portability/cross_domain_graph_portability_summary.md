# Cross-Domain EO and Entity-Graph Portability Extension

## Purpose

This extension addresses three optional dissertation items together:

1. Cyber portability slice.
2. Crypto portability slice.
3. Optional entity-graph / GNN enhancement.

The extension is deliberately protocol-level. It does not claim a new cyber model, crypto model, XRP model or GNN result. It shows how the Evidence Object (EO) contract can receive evidence from adjacent alerting domains and from future graph-derived models.

## Scope and Non-Claims

This extension supports structural portability of the EO-to-narrative protocol. It does not claim:

- cross-domain predictive lift;
- transfer of IEEE-CIS fraud thresholds;
- trained cyber, crypto or GNN performance;
- production readiness in cyber or crypto monitoring.

## Dataset Anchors

Cyber portability is anchored to public labelled cyber-flow datasets such as CICIDS2017, which includes benign and attack network traffic, labelled flows, timestamps, source and destination IPs, ports, protocols and attack labels.

Crypto portability is anchored to Elliptic Bitcoin as the labelled crypto AML benchmark, where transactions are nodes, Bitcoin flows are edges, and labels distinguish licit, illicit and unknown entities.

XRP Ledger is treated as a plausible future transaction-monitoring source because public XRPL transaction data is available through sources such as AWS Public Blockchain Datasets. However, no standard public XRP labelled illicit/lict benchmark equivalent to Elliptic Bitcoin is used here.

## Unified EO Mapping

| EO field | Fraud transaction monitoring | Cyber alert / flow monitoring | Crypto transaction monitoring | Entity-graph / GNN extension |
|---|---|---|---|---|
| event_id | TransactionID | Flow ID or alert ID | Transaction hash or ledger sequence | Node ID |
| score | Fraud probability | Intrusion or anomaly score | AML / risk score | GNN or graph risk score |
| risk_band | LOW / MED / HIGH | LOW / MED / HIGH | LOW / MED / HIGH | LOW / MED / HIGH |
| top_drivers | SHAP transaction drivers | Flow, host, protocol or service drivers | Wallet, counterparty or attribution drivers | Graph-derived drivers |
| entity_context | Card, address, device or identity indicators | Source IP, destination IP, host, service | Sender, receiver, wallet or asset | Node / neighbourhood metadata |
| graph_context | Optional entity linkages | IP-flow-host graph | Transaction/account graph | Explicit graph features |
| evidence_strength | Coverage and sparsity of transaction evidence | Telemetry completeness and sensor coverage | Wallet-history and attribution coverage | Graph coverage and neighbourhood completeness |
| recommended_action | allow, step-up, review | suppress, monitor, investigate | allow, enhanced review, hold or investigate | Domain-specific action |
| required_disclosure | Thin-file or low-evidence warning | Sparse telemetry warning | Limited attribution warning | Sparse-neighbourhood or graph-coverage warning |

## Cyber Portability Example

A cyber EO can represent a labelled intrusion-detection flow or alert. The event identifier becomes a flow ID or alert ID. Drivers become network or host indicators, such as failed login rate, byte volume, destination reputation, protocol or connection burst features.

Example template narrative:

> Risk is HIGH. The alert is mainly driven by elevated failed login rate and connection burst behaviour. Destination reputation also increases risk. Evidence strength is MED because flow telemetry is available, so the recommended action is investigate.

## Crypto Portability Example

A crypto EO can represent a transaction or wallet event. The event identifier becomes a transaction hash or ledger reference. Drivers become wallet age, counterparty risk, transaction amount, attribution coverage, transaction frequency or community-risk indicators.

Example template narrative:

> Risk is MED. The transaction is mainly driven by elevated counterparty risk and limited wallet history. Prior clean activity reduces risk slightly. Evidence strength is LOW because attribution coverage is limited, so the recommended action is enhanced review.

## Entity-Graph / GNN Extension

The EO can be extended with a graph_context block. This allows graph-derived evidence to be passed into the same narrative and validator layer.

Example graph_context fields:

- node_type;
- edge_types;
- neighbourhood_depth;
- degree;
- community_risk_rate;
- pagerank_bucket;
- graph_score;
- gnn_score_available;
- sparse_neighbourhood_flag.

The GNN is therefore treated as a future evidence-source adapter. A future GNN or entity-graph model could populate EO fields with graph risk score, graph-derived drivers and neighbourhood evidence. The dissertation does not claim trained GNN performance.

## Validator Portability

The same validator hierarchy transfers across domains:

1. Schema gate: EO and narrative conform to the expected structure.
2. Closed-world gate: narrative uses only EO-provided facts.
3. Driver gate: mentioned drivers are drawn from EO top_drivers.
4. Action gate: recommendation matches recommended_action_class.
5. Disclosure gate: LOW evidence, sparse telemetry, limited attribution or sparse graph coverage requires explicit disclosure.

## Conclusion

The cyber, crypto and graph examples collectively show that the EO-to-narrative protocol is structurally portable beyond tabular fraud decisioning. The protocol can receive evidence from cyber-flow datasets, crypto transaction graphs and future entity-graph/GNN models. The claim is protocol portability only, not cross-domain model performance.
