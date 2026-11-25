# Data pipeline

Download public CSV exports via the [CPSC open data index](https://www.cpsc.gov/Data). Local data paths are configurable. Preserve the export dates and hashes for repeat experiments.

```bash
recallguard prepare --incidents data/raw/IncidentReports.csv \
  --recalls data/raw/recalls_recall_listing.csv --out data/processed
```

The reader recognizes the incident disclaimer preamble, retains only supported columns, preserves string identifiers, parses mixed date formats, and logs invalid dates. Exact duplicates collapse and conflicting IDs are excluded. The recall export supports `Recall Number`, `Date`, `Name of product`, `Recall Heading`, and `Manufacturers`. Optional incident fields are filled consistently. Normalization applies Unicode NFKC, case folding, punctuation and placeholder handling, and common manufacturer suffix removal.

For each incident, the index votes on recall candidates sharing informative words. Token blocks are capped at 75 recalls and each report generates at most 18 pairs. The scorer combines product token Jaccard (0.50), normalized string similarity (0.30), exact nonempty manufacturer (0.15) and brand tokens in recall product or heading (0.05). The current files lack reliably structured recall-side UPCs, brands, product categories and product types. Product categories and types support model features, while identity in linkage depends on the shared evidence. Candidate date differences must be strictly positive and within the configured horizon, 365 days by default.

The default high threshold is 0.82 and requires manufacturer or brand agreement. A score of at least 0.57 gives review status when date and product-token checks pass. `--high`, `--review`, and `--horizon` configure those rules. A unique high candidate produces label 1. Multiple high candidates, review candidates, rejects, and incidents with no candidates remain unresolved. The thresholds are heuristic. Shared products can be missed when candidate blocks drop common tokens or when name, brand or manufacturer values differ.

Outputs include `candidates.csv` with match evidence and rejection reasons, `observations.csv` with a nullable target, `review_sample.csv` with high-scoring examples per status, and `summary.json` with input hashes, configuration and counts. On the local exports, ingestion read 65,058 incidents and 9,666 recall listings. After ID validation, 9,624 recall rows remained. The run produced 1,144,376 reviewable pairs and four proxy positives. This low yield calls for adjudication and improved identity extraction before a reconstructed training set can be benchmarked. It does not establish linkage precision or recall. The earlier capstone files used different rules and a different sample. The CLI's supervised entry point uses the original CPSC-shaped pre-labeled CSV.

The SaferProducts terms allow copying and distribution of posted public information while excluding personally identifiable information. Public reports can contain narrative and contextual identifiers and carry accuracy limitations. Keep raw and derived exports local and review narrative exposure before any redistribution or public demo. The existing MIT license applies to project software. [SaferProducts Terms of Use](https://www.saferproducts.gov/Business/Acknowledge?source=signIn) · [CPSC Privacy and Security Notice](https://www.cpsc.gov/About-CPSC/Policies-Statements-and-Directives/Privacy-Policy)
