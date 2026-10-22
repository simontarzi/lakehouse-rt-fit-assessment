# Lakehouse//RT fit assessment

A read-only Databricks notebook that checks which SQL warehouse workloads running on DBSQL today are a good fit for
Lakehouse//RT (Real-Time warehouses), how confident that verdict is, and what holds the rest back.

## What it reports

- Per warehouse: fit (strong / good / after tuning / weak / not a fit), whether the whole warehouse or only a subset of its
  queries can move, a 0–100% confidence score, whether it writes, and a one-sentence `why`.
- The share of executed SELECTs that qualify today, and the blockers for the rest (Genie / Jobs, unsupported SQL, unsupported
  tables or governance, preview guardrails).
- A query plan check: the most frequent query patterns are planned with `EXPLAIN` (never run) to find blockers hidden inside
  views, such as AI functions, Python UDFs, row filters / column masks, or foreign, external and Delta Sharing tables.
- Latency, queueing and load per warehouse, an indicative cost comparison at list price, connector readiness, table layout
  tips, and the top query patterns as proof-of-concept candidates.
- A copy/paste summary at the end.

## How to run

1. Import `lakehouse_rt_fit_assessment.py` into a workspace (Workspace → Import, or `databricks workspace import`).
2. Attach serverless notebook compute (or a Unity Catalog cluster on DBR 15.4+).
3. Adjust the widgets if needed (lookback days, workspace / warehouse filters, plan check on/off) and run all cells.

Run it as an account or metastore admin with `SELECT` on `system.query`, `system.access`, `system.compute`, `system.billing`
and `system.information_schema`. The query plan check also needs read access to the tables the queries use; patterns that
can't be planned are reported as "could not check". It takes about 5 minutes for a few million statements.

## Read-only

The notebook only runs `SELECT`, `DESCRIBE` and `EXPLAIN` statements against system tables and table metadata. It creates,
modifies or caches nothing and changes no configuration. Its own queries carry the session query tag
`use_case = lakehouse-rt-fit-assessment` so Databricks can count how often the notebook is used; the tag holds no account data.

Costs are at list price. Results are a heuristic assessment; confirm with a proof of concept.
