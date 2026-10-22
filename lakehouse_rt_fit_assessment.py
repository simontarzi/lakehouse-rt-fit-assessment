# Databricks notebook source
# MAGIC %md
# MAGIC # Lakehouse//RT (Reyden) — workload fit assessment
# MAGIC
# MAGIC This notebook checks the SQL warehouses in this account and estimates:
# MAGIC
# MAGIC 1. **Which warehouses — and how much of their load — are a good fit for Lakehouse//RT**
# MAGIC 2. **How confident we are per warehouse (0–100%)**, whether the **whole warehouse** can move (full fit) or only a
# MAGIC    **subset** of its queries (partial fit, with the share that could move), whether it **writes**, and **why**
# MAGIC 3. **What percentage of queries qualify today**, and what blocks the rest
# MAGIC 4. **The expected benefits**: latency, concurrency/queueing, and an indicative cost comparison
# MAGIC
# MAGIC **It is read-only.** It only runs `SELECT`, `DESCRIBE`, and `EXPLAIN` statements against system tables and table metadata.
# MAGIC `EXPLAIN` plans the most frequent warehouse queries in this notebook session without running them, so no warehouse is used.
# MAGIC It does not create, modify, or cache any table, view, or file, and it does not change any configuration.
# MAGIC Its own queries carry the session query tag `use_case = lakehouse-rt-fit-assessment`, so Databricks can count how often
# MAGIC the notebook is used. The tag holds no data from this account.
# MAGIC
# MAGIC | Requirement | Details |
# MAGIC |---|---|
# MAGIC | Compute | Serverless notebook compute, or a Unity Catalog cluster (DBR 15.4+). Python is required, so not a SQL warehouse. |
# MAGIC | Permissions | `SELECT` on `system.query`, `system.access`, `system.compute`, `system.billing`, and `system.information_schema` (typically an account or metastore admin). Any section without access is skipped. The query plan check also needs `SELECT` on the tables the queries read; queries you can't read are reported as "could not check". |
# MAGIC | Runtime | A few minutes. Most of the time goes to one pass over `system.query.history`. |
# MAGIC
# MAGIC **How a query is judged** (based on the public [Lakehouse//RT limitations](https://docs.databricks.com/aws/en/compute/sql-warehouse/real-time)):
# MAGIC a finished `SELECT` qualifies if it ran in ≤ 20 s, didn't come from Genie or a Job, uses no unsupported SQL
# MAGIC (AI functions, Python UDFs, spatial/GEO, XML/XPath, `input_file_*`, file-path or streaming reads),
# MAGIC reads only supported tables (UC managed Delta/Iceberg tables, materialized views, streaming tables, metric views —
# MAGIC no external, foreign, Delta Sharing, Hive metastore, or system tables, and no row filters or column masks),
# MAGIC and stays within the preview guardrails (2 min runtime, 300 GB shuffle, 30 GB spill, 50K files).
# MAGIC Queries running 20 s – 2 min are reported as **"after tuning"**: they could move after layout tuning
# MAGIC (liquid clustering, OPTIMIZE, ANALYZE).
# MAGIC The SQL-text check can't see inside views, so the **query plan check** then plans the most frequent query patterns with
# MAGIC `EXPLAIN` and moves patterns whose plan shows a blocker (for example an AI function or external table inside a view) to blocked.
# MAGIC
# MAGIC Percentages are of SELECTs the engine actually executed. Result-cache hits are counted separately, because they cost nothing
# MAGIC today and gain nothing from a faster engine. Queries on tables the current user can't see in `information_schema` are shown as
# MAGIC **"probably qualifies"** (an upper bound). Run the notebook as a metastore admin to keep that bucket small.
# MAGIC
# MAGIC Run all cells top to bottom. Adjust the widgets at the top if needed.

# COMMAND ----------

dbutils.widgets.text("lookback_days", "30", "01 Lookback days")
dbutils.widgets.text("workspace_ids", "", "02 Workspace IDs (comma-separated, empty = all)")
dbutils.widgets.text("warehouse_ids", "", "03 Warehouse IDs (comma-separated, empty = all)")
dbutils.widgets.text("eligible_max_seconds", "20", "04 Max runtime to qualify (seconds)")
dbutils.widgets.dropdown("show_query_text", "yes", ["yes", "no"], "05 Show sample query text")
dbutils.widgets.text("efficiency_factors", "1,2,4", "06 Cost scenarios: compute reduction factors")
dbutils.widgets.text("rt_price_override", "", "07 Lakehouse//RT $/DBU override (empty = list price)")
dbutils.widgets.text("top_patterns", "10", "08 Top query patterns per warehouse")
dbutils.widgets.dropdown("inspect_tables", "yes", ["yes", "no"], "09 Inspect layout of top tables")
dbutils.widgets.dropdown("plan_check", "yes", ["yes", "no"], "10 Check query plans of top patterns (EXPLAIN)")

# COMMAND ----------

import math
import re
import time

import pandas as pd

pd.set_option("display.max_colwidth", 400)

NOTEBOOK_VERSION = "0.2.0"
# Usage tag on this session's queries (visible in system.query.history.query_tags). Session setting only.
try:
    spark.sql("SET QUERY_TAGS['use_case'] = 'lakehouse-rt-fit-assessment', "
              f"QUERY_TAGS['version'] = '{NOTEBOOK_VERSION}'")
except Exception:  # compute without query-tag support: run untagged
    pass


def _int_widget(name, lo, hi):
    v = int(dbutils.widgets.get(name).strip())
    if not lo <= v <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}")
    return v


def _id_list(name):
    vals = [v.strip() for v in dbutils.widgets.get(name).split(",") if v.strip()]
    for v in vals:
        if not re.fullmatch(r"[0-9A-Za-z_-]+", v):
            raise ValueError(f"Invalid value in {name}: {v!r}")
    return vals


DAYS = _int_widget("lookback_days", 1, 365)
ELIG_S = _int_widget("eligible_max_seconds", 1, 120)
ELIG_MS = ELIG_S * 1000
TOP_PATTERNS = _int_widget("top_patterns", 0, 100)
WORKSPACES = _id_list("workspace_ids")
WAREHOUSES = _id_list("warehouse_ids")
SHOW_TEXT = dbutils.widgets.get("show_query_text") == "yes"
INSPECT_TABLES = dbutils.widgets.get("inspect_tables") == "yes"
PLAN_CHECK = dbutils.widgets.get("plan_check") == "yes" and TOP_PATTERNS > 0
PLAN_MAX_PATTERNS, PLAN_BUDGET_S, PLAN_MAX_CHARS = 200, 600, 50000  # plan check limits: patterns, seconds, query length
FACTORS = [float(f) for f in dbutils.widgets.get("efficiency_factors").split(",") if f.strip()] or [1.0]
RT_PRICE_OVERRIDE = float(dbutils.widgets.get("rt_price_override")) if dbutils.widgets.get("rt_price_override").strip() else None
MONTH = 30.0 / DAYS  # scales the lookback window to a 30-day month


def in_list(col, vals):
    return f" AND {col} IN ({', '.join(repr(v) for v in vals)})" if vals else ""


QH_FILTER = in_list("q.workspace_id", WORKSPACES) + in_list("q.compute.warehouse_id", WAREHOUSES)


def run(label, sql):
    """Runs one read-only statement and returns a pandas DataFrame (None if the source isn't accessible)."""
    t0 = time.time()
    try:
        pdf = spark.sql(sql).toPandas()
        print(f"✓ {label}: {len(pdf):,} rows in {time.time() - t0:.0f}s")
        return pdf
    except Exception as e:  # missing table or privilege → skip that section
        print(f"✗ {label} skipped: {str(e).splitlines()[0][:300]}")
        return None


def table_exists(name):
    try:
        spark.table(name).columns  # resolves the schema only, reads no data
        return True
    except Exception:
        return False


_schemas = {}


def has_field(table, path):
    t = _schemas.setdefault(table, spark.table(table).schema)
    for part in path.split("."):
        names = {f.name: f for f in t.fields} if hasattr(t, "fields") else {}
        if part not in names:
            return False
        t = names[part].dataType
    return True


SRC = {t: table_exists(t) for t in [
    "system.query.history",
    "system.access.table_lineage",
    "system.information_schema.tables",
    "system.information_schema.row_filters",
    "system.information_schema.column_masks",
    "system.information_schema.routines",
    "system.compute.warehouses",
    "system.billing.usage",
    "system.billing.list_prices",
]}
for t, ok in SRC.items():
    print(("✓ " if ok else "✗ ") + t)
assert SRC["system.query.history"], "system.query.history is required. Ask an admin for SELECT on system.query."

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Collect (read-only)

# COMMAND ----------

# --- Warehouse inventory (latest configuration per warehouse) ---
inventory = run("warehouse inventory", """
  SELECT warehouse_id, workspace_id, warehouse_name, warehouse_type, warehouse_size,
         min_clusters, max_clusters, auto_stop_minutes, delete_time IS NOT NULL AS deleted
  FROM system.compute.warehouses
  QUALIFY row_number() OVER (PARTITION BY warehouse_id ORDER BY change_time DESC) = 1
""") if SRC["system.compute.warehouses"] else None

# --- DBUs and list cost per warehouse, plus the Lakehouse//RT list price for the same region/tier ---
billing = None
if SRC["system.billing.usage"] and SRC["system.billing.list_prices"]:
    billing = run("billing per warehouse", f"""
      WITH prices AS (
        SELECT sku_name, cloud, usage_unit, price_start_time, price_end_time,
               coalesce(pricing.effective_list.default, pricing.default) AS price
        FROM system.billing.list_prices
        WHERE currency_code = 'USD'
      ),
      u AS (
        SELECT u.usage_metadata.warehouse_id AS warehouse_id, u.sku_name, u.cloud,
               date_trunc('hour', u.usage_start_time) AS hr,
               u.usage_quantity AS dbu, u.usage_quantity * p.price AS list_cost
        FROM system.billing.usage u
        LEFT JOIN prices p
          ON p.sku_name = u.sku_name AND p.cloud = u.cloud AND p.usage_unit = u.usage_unit
         AND u.usage_start_time >= p.price_start_time
         AND (p.price_end_time IS NULL OR u.usage_start_time < p.price_end_time)
        WHERE u.usage_date >= date_sub(current_date(), {DAYS})
          AND u.usage_metadata.warehouse_id IS NOT NULL
          AND u.usage_unit = 'DBU' {in_list("u.workspace_id", WORKSPACES)} {in_list("u.usage_metadata.warehouse_id", WAREHOUSES)}
      ),
      h AS (
        SELECT warehouse_id, hr, sum(dbu) AS dbu, sum(list_cost) AS list_cost,
               max_by(sku_name, dbu) AS sku_name, any_value(cloud) AS cloud
        FROM u GROUP BY warehouse_id, hr
      ),
      w AS (
        SELECT warehouse_id, sum(dbu) AS dbus, sum(list_cost) AS list_cost,
               max(dbu) AS peak_hour_dbu, count(*) AS billed_hours,
               max_by(sku_name, dbu) AS sku_name, any_value(cloud) AS cloud
        FROM h GROUP BY warehouse_id
      ),
      cur AS (SELECT sku_name, cloud, price FROM prices WHERE price_end_time IS NULL OR price_end_time > current_timestamp())
      SELECT w.*,
             rt.price AS rt_list_price,      -- Lakehouse//RT SKU in the same region and tier, if published
             sl.price AS serverless_price    -- Serverless SQL SKU in the same region and tier (fallback)
      FROM w
      LEFT JOIN cur rt ON rt.cloud = w.cloud
        AND rt.sku_name = regexp_replace(w.sku_name, '(SERVERLESS_SQL_COMPUTE|SQL_PRO_COMPUTE|SQL_COMPUTE)', 'LAKEHOUSE_SERVERLESS_COMPUTE')
      LEFT JOIN cur sl ON sl.cloud = w.cloud
        AND sl.sku_name = regexp_replace(w.sku_name, '(LAKEHOUSE_SERVERLESS_COMPUTE|SERVERLESS_SQL_COMPUTE|SQL_PRO_COMPUTE|SQL_COMPUTE)', 'SERVERLESS_SQL_COMPUTE')
    """)

# --- Names of Python UDFs (unsupported on Lakehouse//RT), used to flag queries that call them ---
udf_names = []
if SRC["system.information_schema.routines"]:
    r = run("Python UDF names", """
      SELECT DISTINCT lower(routine_name) AS name
      FROM system.information_schema.routines
      WHERE upper(external_language) = 'PYTHON'
      LIMIT 500
    """)
    if r is not None:
        udf_names = [n for n in r["name"] if re.fullmatch(r"[a-z_][a-z0-9_]*", n or "")]

# COMMAND ----------

# --- Shared SQL fragments ---

# Table support status, from information_schema (+ row filters / column masks).
if SRC["system.information_schema.row_filters"] and SRC["system.information_schema.column_masks"]:
    GOVERNED = """
      SELECT lower(concat_ws('.', table_catalog, table_schema, table_name)) AS full_name FROM system.information_schema.row_filters
      UNION
      SELECT lower(concat_ws('.', table_catalog, table_schema, table_name)) FROM system.information_schema.column_masks"""
else:
    GOVERNED = "SELECT CAST(NULL AS STRING) AS full_name WHERE false"

if SRC["system.information_schema.tables"]:
    TBL = f"""
    governed AS ({GOVERNED}),
    tbl AS (
      SELECT lower(concat_ws('.', t.table_catalog, t.table_schema, t.table_name)) AS full_name,
             t.table_type, t.data_source_format,
             CASE
               WHEN upper(t.table_type) LIKE '%EXTERNAL%' THEN 'external table'
               WHEN upper(t.table_type) = 'FOREIGN' THEN 'foreign / federated table'
               WHEN upper(t.data_source_format) = 'DELTASHARING' THEN 'Delta Sharing table'
               WHEN upper(t.table_type) LIKE 'MANAGED%' AND upper(t.data_source_format) NOT IN ('DELTA', 'ICEBERG') THEN 'non-Delta/Iceberg format'
               WHEN g.full_name IS NOT NULL THEN 'row filter / column mask'
             END AS table_blocker
      FROM system.information_schema.tables t
      LEFT JOIN (SELECT DISTINCT full_name FROM governed) g
        ON g.full_name = lower(concat_ws('.', t.table_catalog, t.table_schema, t.table_name))
    )"""
else:
    TBL = """
    tbl AS (SELECT CAST(NULL AS STRING) AS full_name, CAST(NULL AS STRING) AS table_type,
                   CAST(NULL AS STRING) AS data_source_format, CAST(NULL AS STRING) AS table_blocker WHERE false)"""

# Tables read by each statement, from lineage.
if SRC["system.access.table_lineage"]:
    LIN = f"""
    lin AS (
      SELECT DISTINCT statement_id, lower(source_table_full_name) AS src
      FROM system.access.table_lineage
      WHERE event_date >= date_sub(current_date(), {DAYS})
        AND statement_id IS NOT NULL AND source_table_full_name IS NOT NULL
    )"""
else:
    LIN = "lin AS (SELECT CAST(NULL AS STRING) AS statement_id, CAST(NULL AS STRING) AS src WHERE false)"

UNKNOWN_TABLE = (
    "max(CASE WHEN t.full_name IS NULL AND l.src NOT LIKE 'system.%' AND l.src NOT LIKE 'hive_metastore.%' THEN 1 ELSE 0 END)"
    if SRC["system.information_schema.tables"] else "0"
)

# Where the query came from (built from the query_source fields present in this account).
QH = "system.query.history"
src_cases = []
if has_field(QH, "query_source.genie_space_id"):
    src_cases.append("WHEN q.query_source.genie_space_id IS NOT NULL THEN 'Genie'")
src_cases.append("WHEN q.client_application ILIKE '%genie%' THEN 'Genie'")
if has_field(QH, "query_source.job_info"):
    src_cases.append("WHEN q.query_source.job_info.job_id IS NOT NULL THEN 'Job'")
src_cases.append("WHEN q.client_application = 'Databricks Jobs' THEN 'Job'")
if has_field(QH, "query_source.pipeline_info"):
    src_cases.append("WHEN q.query_source.pipeline_info.pipeline_id IS NOT NULL THEN 'Pipeline'")
for field, label in [("dashboard_id", "AI/BI dashboard"), ("legacy_dashboard_id", "Legacy dashboard"),
                     ("alert_id", "Alert"), ("notebook_id", "Notebook"), ("sql_query_id", "SQL editor")]:
    if has_field(QH, f"query_source.{field}"):
        src_cases.append(f"WHEN q.query_source.{field} IS NOT NULL THEN '{label}'")
src_cases.append("WHEN q.client_application = 'Databricks SQL Dashboard' THEN 'AI/BI dashboard'")
src_cases.append("WHEN q.client_application ILIKE 'Databricks SQL%' THEN 'SQL editor'")
SOURCE_EXPR = "CASE " + " ".join(src_cases) + " ELSE 'External client / API' END"

META_TYPES = "('', 'SHOW', 'DESCRIBE', 'USE', 'SET', 'EXPLAIN')"
NON_WRITE_TYPES = "('SELECT', '', 'SHOW', 'DESCRIBE', 'USE', 'SET', 'EXPLAIN')"
UDF_WHEN = (
    f"WHEN b.txt RLIKE r'(?i)\\b({'|'.join(udf_names)})\\s*\\(' THEN 'Python UDF'" if udf_names else ""
)
SAMPLE_TEXT = (
    "any_value(CASE WHEN category IN ('ELIGIBLE', 'CHECK', 'TUNE') THEN substr(regexp_replace(txt, r'\\s+', ' '), 1, 400) END, true)"
    if SHOW_TEXT else "CAST(NULL AS STRING)"
)
# One full sample per query pattern, with the tables lineage saw it read, for the query plan check.
PLAN_SAMPLE = (
    f"CASE WHEN grouping(pattern_id) = 0 THEN any_value(named_struct('txt', substr(txt, 1, {PLAN_MAX_CHARS}), "
    "'len', length(txt), 'srcs', srcs), true) END"
    if PLAN_CHECK else "CAST(NULL AS STRING)"
)
# Reasons inside the candidate categories (the query plan check moves counts out of them by these names).
R_SUB1, R_1_5, R_5_PLUS = "< 1 s", "1–5 s", f"5–{ELIG_S} s"
R_CHECK, R_TUNE_RT, R_TUNE_FILES = "table not visible to you in information_schema", f"runtime {ELIG_S} s – 2 min", "files read 10K–50K"
GiB = 1024 ** 3

# COMMAND ----------

# --- Main pass: classify every warehouse statement in the window, aggregate per warehouse / reason / query pattern ---
MAIN_SQL = rf"""
WITH {TBL},
{LIN},
stmt_tables AS (
  SELECT l.statement_id,
         nullif(concat_ws(', ', collect_set(CASE
           WHEN l.src LIKE 'system.%' THEN 'system table'
           WHEN l.src LIKE 'hive_metastore.%' THEN 'Hive metastore table'
           ELSE t.table_blocker END)), '') AS table_blocker,
         {UNKNOWN_TABLE} AS has_unknown_table,
         collect_set(l.src) AS srcs
  FROM lin l LEFT JOIN tbl t ON t.full_name = l.src
  GROUP BY l.statement_id
),
base AS (
  SELECT q.statement_id, q.workspace_id, q.compute.warehouse_id AS warehouse_id,
         q.execution_status AS status,
         upper(coalesce(q.statement_type, '')) AS stype,
         -- any write / DDL attempt (also failed ones): Lakehouse//RT rejects all of them
         upper(coalesce(q.statement_type, '')) NOT IN {NON_WRITE_TYPES} AS is_write,
         q.statement_text AS txt,
         q.executed_by,
         q.start_time,
         coalesce(q.execution_duration_ms, 0) AS exec_ms,
         coalesce(q.total_task_duration_ms, 0) AS task_ms,
         coalesce(q.waiting_at_capacity_duration_ms, 0) AS queue_ms,
         coalesce(q.read_bytes, 0) AS read_bytes,
         coalesce(q.read_files, 0) AS read_files,
         coalesce(q.shuffle_read_bytes, 0) AS shuffle_bytes,
         coalesce(q.spilled_local_bytes, 0) AS spill_bytes,
         coalesce(q.written_rows, 0) AS written_rows,
         coalesce(q.from_result_cache, false) AS from_cache,
         coalesce(q.client_application, 'unknown') AS client_app,
         coalesce(q.client_driver, '') AS client_driver,
         {SOURCE_EXPR} AS source,
         -- does the same connection/session also write? Then its reads can't move without an app change
         max(CASE WHEN upper(coalesce(q.statement_type, '')) NOT IN {NON_WRITE_TYPES} THEN 1 ELSE 0 END)
           OVER (PARTITION BY coalesce(q.session_id, q.statement_id)) AS session_writes
  FROM system.query.history q
  WHERE q.start_time >= current_timestamp() - INTERVAL {DAYS} DAYS
    AND q.compute.warehouse_id IS NOT NULL {QH_FILTER}
),
cls AS (
  SELECT b.*,
         st.statement_id IS NOT NULL AS has_lineage,
         st.table_blocker,
         coalesce(st.has_unknown_table, 0) AS has_unknown_table,
         st.srcs,
         (b.status = 'FINISHED' AND b.stype = 'SELECT') AS is_read,
         CASE WHEN b.status = 'FINISHED' AND b.stype = 'SELECT' THEN CASE
           WHEN b.txt RLIKE r'(?i)\b(ai_[a-z_]+|vector_search)\s*\(' THEN 'AI function'
           {UDF_WHEN}
           WHEN b.txt RLIKE r'(?i)\b(st_[a-z0-9_]+|h3_[a-z0-9_]+)\s*\(|\b(geography|geometry)\b' THEN 'spatial function / GEO type'
           WHEN b.txt RLIKE r'(?i)\b(xpath[a-z_]*|from_xml|to_xml|schema_of_xml)\s*\(' THEN 'XML / XPath function'
           WHEN b.txt RLIKE r'(?i)\binput_file_(name|block_start|block_length)\s*\(' THEN 'input_file_* function'
           WHEN b.txt RLIKE r'(?i)\bcount\s*\(\s*[a-z_][a-z0-9_]*\s*\.\s*\*\s*\)' THEN 'count(t.*)'
           WHEN b.txt RLIKE r'(?i)\b(read_files|remote_query|read_kafka|read_kinesis|read_pubsub|read_pulsar|read_statestore|read_state_metadata)\s*\(|\b(delta|parquet|csv|json|orc|avro|text|binaryfile)\s*\.\s*`' THEN 'file path / streaming / federation function'
         END END AS sql_blocker,
         CASE WHEN b.status = 'FINISHED' AND b.stype = 'SELECT' THEN CASE
           WHEN b.txt RLIKE r'(?i)`?\bsystem`?\s*\.\s*`?[a-z_]+`?\s*\.' THEN 'system table'
           WHEN b.txt RLIKE r'(?i)\binformation_schema`?\s*\.' THEN 'information_schema view'
           WHEN b.txt RLIKE r'(?i)`?\bhive_metastore`?\s*\.' THEN 'Hive metastore table'
         END END AS text_table_blocker
  FROM base b LEFT JOIN stmt_tables st ON st.statement_id = b.statement_id
),
cat AS (
  SELECT c.*,
    CASE
      WHEN c.is_write                                           THEN array('WRITE', c.stype)
      WHEN c.status <> 'FINISHED'                               THEN array('FAILED', 'failed / canceled')
      WHEN c.stype IN {META_TYPES}                              THEN array('META', coalesce(nullif(c.stype, ''), 'untyped'))
      WHEN c.source IN ('Genie', 'Job', 'Pipeline')             THEN array('BLK_TOOL', c.source)
      WHEN c.sql_blocker IS NOT NULL                            THEN array('BLK_SQL', c.sql_blocker)
      WHEN coalesce(c.table_blocker, c.text_table_blocker) IS NOT NULL
                                                                THEN array('BLK_TABLE', coalesce(c.table_blocker, c.text_table_blocker))
      WHEN c.exec_ms > 120000                                   THEN array('BLK_GUARD', 'runtime > 2 min')
      WHEN c.shuffle_bytes > {300 * GiB}                        THEN array('BLK_GUARD', 'shuffle > 300 GB')
      WHEN c.spill_bytes > {30 * GiB}                           THEN array('BLK_GUARD', 'spill > 30 GB')
      WHEN c.read_files > 50000                                 THEN array('BLK_GUARD', 'files read > 50K')
      WHEN c.exec_ms > {ELIG_MS}                                THEN array('TUNE', '{R_TUNE_RT}')
      WHEN c.read_files > 10000                                 THEN array('TUNE', '{R_TUNE_FILES}')
      WHEN c.from_cache                                         THEN array('CACHE', 'served from result cache')
      WHEN c.has_unknown_table = 1                              THEN array('CHECK', '{R_CHECK}')
      WHEN c.exec_ms < 1000                                     THEN array('ELIGIBLE', '{R_SUB1}')
      WHEN c.exec_ms < 5000                                     THEN array('ELIGIBLE', '{R_1_5}')
      ELSE                                                           array('ELIGIBLE', '{R_5_PLUS}')
    END AS cr
  FROM cls c
),
cat2 AS (
  SELECT *, cr[0] AS category, cr[1] AS reason,
    CASE WHEN cr[0] IN ('ELIGIBLE', 'CHECK', 'TUNE') THEN CAST(xxhash64(regexp_replace(regexp_replace(regexp_replace(
      lower(substr(txt, 1, 5000)), r"'[^']*'", '?'), r'\b\d+(\.\d+)?\b', '?'), r'\s+', ' ')) AS STRING) END AS pattern_id,
    CASE
      WHEN client_driver ILIKE 'DatabricksSqlExecApi%' OR client_driver ILIKE 'ADBC%' THEN 'REST'
      WHEN client_driver ILIKE '%odbc%' OR client_app ILIKE '%odbc%' THEN 'ODBC'
      WHEN client_driver ILIKE '%jdbc%' OR client_app ILIKE '%jdbc%' THEN 'JDBC'
      WHEN client_driver ILIKE 'PyDatabricksSqlConnector%' THEN 'PYTHON'
      WHEN client_driver ILIKE '%node%' OR client_app ILIKE '%node%' THEN 'NODE'
      ELSE 'OTHER'
    END AS conn
  FROM cat
),
agg AS (
  SELECT
    warehouse_id,
    CASE WHEN grouping(pattern_id) = 0 THEN 'pattern' WHEN grouping(category) = 0 THEN 'reason' ELSE 'warehouse' END AS lvl,
    pattern_id, category, reason,
    any_value(workspace_id) AS workspace_id,
    count(*) AS n,
    count_if(is_read) AS n_reads,
    count_if(category = 'ELIGIBLE') AS n_eligible,
    count_if(category = 'TUNE') AS n_tune,
    count_if(category = 'TUNE' AND reason = '{R_TUNE_FILES}') AS n_tune_files,
    count_if(category = 'CHECK') AS n_check,
    count_if(category = 'BLK_TOOL') AS n_blk_tool,
    count_if(category = 'BLK_SQL') AS n_blk_sql,
    count_if(category = 'BLK_TABLE') AS n_blk_table,
    count_if(category = 'BLK_GUARD') AS n_blk_guard,
    count_if(category = 'WRITE') AS n_write_attempts,
    count_if(category = 'WRITE' AND status = 'FINISHED') AS n_write,
    sum(CASE WHEN category = 'WRITE' AND status = 'FINISHED' THEN written_rows ELSE 0 END) AS written_rows,
    count_if(category = 'WRITE' AND (source IN ('Job', 'Pipeline') OR client_app ILIKE '%dbt%')) AS n_writes_scheduled,
    count_if(category IN ('ELIGIBLE', 'CACHE') AND session_writes = 1) AS n_movable_in_write_sessions,
    mode(CASE WHEN category = 'WRITE' THEN CASE WHEN client_app ILIKE '%dbt%' THEN 'dbt' ELSE source END END) AS top_write_source,
    count_if(category = 'META') AS n_meta,
    count_if(category = 'FAILED') AS n_failed,
    count_if(category = 'CACHE') AS n_cache,
    -- 10-minute windows with qualifying traffic: approximates how long a Real-Time warehouse would be up
    approx_count_distinct(CASE WHEN category IN ('ELIGIBLE', 'CACHE') THEN floor(unix_timestamp(start_time) / 600) END) AS elig_10min_windows,
    count_if(category = 'ELIGIBLE' AND reason = '{R_SUB1}') AS n_elig_sub1s,
    count_if(category = 'ELIGIBLE' AND reason = '{R_1_5}') AS n_elig_1_5s,
    count_if(category = 'ELIGIBLE' AND exec_ms >= 5000) AS n_elig_5s_plus,
    count_if(category = 'ELIGIBLE' AND queue_ms > 1000) AS n_elig_queued,
    sum(CASE WHEN category = 'ELIGIBLE' THEN queue_ms ELSE 0 END) AS queue_ms_eligible,
    sum(CASE WHEN status = 'FINISHED' THEN task_ms ELSE 0 END) AS task_ms_all,
    sum(CASE WHEN category = 'ELIGIBLE' THEN task_ms ELSE 0 END) AS task_ms_eligible,
    sum(CASE WHEN category = 'TUNE' THEN task_ms ELSE 0 END) AS task_ms_tune,
    percentile_approx(CASE WHEN is_read THEN exec_ms END, array(0.5, 0.95)) AS read_pct_ms,
    percentile_approx(CASE WHEN category = 'ELIGIBLE' THEN exec_ms END, array(0.5, 0.95)) AS elig_pct_ms,
    avg(CASE WHEN category = 'ELIGIBLE' THEN read_bytes END) AS elig_avg_read_bytes,
    approx_count_distinct(executed_by) AS n_users,
    count_if(category = 'ELIGIBLE' AND source = 'AI/BI dashboard') AS e_src_dashboard,
    count_if(category = 'ELIGIBLE' AND source = 'Legacy dashboard') AS e_src_legacy_dashboard,
    count_if(category = 'ELIGIBLE' AND source = 'SQL editor') AS e_src_sql_editor,
    count_if(category = 'ELIGIBLE' AND source = 'Notebook') AS e_src_notebook,
    count_if(category = 'ELIGIBLE' AND source = 'Alert') AS e_src_alert,
    count_if(category = 'ELIGIBLE' AND source = 'External client / API') AS e_src_external,
    count_if(category = 'ELIGIBLE' AND source = 'External client / API' AND conn = 'REST') AS e_conn_rest,
    count_if(category = 'ELIGIBLE' AND source = 'External client / API' AND conn = 'JDBC') AS e_conn_jdbc,
    count_if(category = 'ELIGIBLE' AND source = 'External client / API' AND conn = 'PYTHON') AS e_conn_python,
    count_if(category = 'ELIGIBLE' AND source = 'External client / API' AND conn = 'NODE') AS e_conn_node,
    count_if(category = 'ELIGIBLE' AND source = 'External client / API' AND conn = 'ODBC') AS e_conn_odbc,
    count_if(category = 'ELIGIBLE' AND source = 'External client / API' AND conn = 'OTHER') AS e_conn_other,
    count_if(category = 'ELIGIBLE' AND client_app ILIKE '%power%bi%') AS e_powerbi,
    count_if(stype = 'SET' AND lower(txt) LIKE '%ansi_mode%false%') AS n_ansi_off,
    count_if(is_read AND has_lineage) AS n_reads_with_lineage,
    mode(source) AS top_source,
    mode(client_app) AS top_client,
    {SAMPLE_TEXT} AS sample_text,
    {PLAN_SAMPLE} AS plan_sample
  FROM cat2
  GROUP BY GROUPING SETS ((warehouse_id), (warehouse_id, pattern_id), (warehouse_id, category, reason))
)
SELECT * EXCEPT (rn) FROM (
  SELECT a.*,
         CASE WHEN lvl = 'pattern' THEN row_number() OVER (PARTITION BY warehouse_id, lvl ORDER BY pattern_id IS NULL, n DESC) END AS rn
  FROM agg a
)
WHERE lvl <> 'pattern' OR (pattern_id IS NOT NULL AND rn <= {TOP_PATTERNS})
"""

main = run(f"query classification ({DAYS} days of query history)", MAIN_SQL)
assert main is not None and len(main), "No warehouse queries found in the selected window and filters."

# COMMAND ----------

# --- Load profile: peak queries per minute / second and approximate concurrency (narrow column scan, no text) ---
peak = run("peak load per warehouse", f"""
  WITH s AS (
    SELECT q.compute.warehouse_id AS warehouse_id, date_trunc('second', q.start_time) AS sec,
           count(*) AS n, sum(coalesce(q.total_duration_ms, 0)) AS dur_ms
    FROM system.query.history q
    WHERE q.start_time >= current_timestamp() - INTERVAL {DAYS} DAYS
      AND q.compute.warehouse_id IS NOT NULL
      AND q.statement_type = 'SELECT' AND q.execution_status = 'FINISHED' {QH_FILTER}
    GROUP BY 1, 2
  ),
  m AS (
    SELECT warehouse_id, date_trunc('minute', sec) AS minute,
           sum(n) AS qpm, max(n) AS max_qps, sum(dur_ms) / 60000.0 AS conc
    FROM s GROUP BY 1, 2
  )
  SELECT warehouse_id,
         max(qpm) AS peak_qpm, percentile_approx(qpm, 0.95) AS p95_qpm,
         max(max_qps) AS peak_qps,
         max(conc) AS peak_concurrency, percentile_approx(conc, 0.95) AS p95_concurrency,
         count(*) AS active_minutes
  FROM m GROUP BY warehouse_id
""")

# --- Tables read by warehouse SELECTs and whether Lakehouse//RT can read them ---
tables = None
if SRC["system.access.table_lineage"]:
    tables = run("tables read by warehouse queries", f"""
      WITH {TBL},
      {LIN},
      q AS (
        SELECT q.statement_id, q.compute.warehouse_id AS warehouse_id
        FROM system.query.history q
        WHERE q.start_time >= current_timestamp() - INTERVAL {DAYS} DAYS
          AND q.compute.warehouse_id IS NOT NULL
          AND q.statement_type = 'SELECT' AND q.execution_status = 'FINISHED' {QH_FILTER}
      )
      SELECT l.src AS table_name,
             CASE WHEN l.src LIKE 'system.%' THEN 'system table'
                  WHEN l.src LIKE 'hive_metastore.%' THEN 'Hive metastore table'
                  WHEN t.full_name IS NULL THEN 'not visible in information_schema'
                  ELSE coalesce(t.table_blocker, 'supported') END AS rt_status,
             any_value(t.table_type) AS table_type,
             any_value(t.data_source_format) AS format,
             count(*) AS n_select_queries,
             size(collect_set(q.warehouse_id)) AS n_warehouses,
             array_join(slice(array_sort(collect_set(q.warehouse_id)), 1, 5), ', ') AS warehouses
      FROM lin l
      JOIN q ON q.statement_id = l.statement_id
      LEFT JOIN tbl t ON t.full_name = l.src
      GROUP BY 1, 2
      ORDER BY n_select_queries DESC
      LIMIT 200
    """)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Query plan check (EXPLAIN only)
# MAGIC The SQL-text check can't see inside views, and lineage doesn't cover every statement. For the most frequent query patterns
# MAGIC (up to the "top query patterns" widget per warehouse, at most 200 in total, about 10 minutes at most), one sample of each is
# MAGIC planned with `EXPLAIN EXTENDED` in this notebook session. The query is planned, **never run**, and no warehouse is used.
# MAGIC The plan lists every table and function the query really uses, including those inside views. A pattern whose plan shows a
# MAGIC blocker (AI function, Python UDF, row filter / column mask, external, foreign or Delta Sharing table, …) is moved to
# MAGIC "Blocked: found in query plan". Parameterised queries are planned with their parameters set to NULL. Queries that read
# MAGIC tables you can't access, or depend on session settings, are reported as "could not check".

# COMMAND ----------

P_OK, P_BLOCKED, P_UNVERIFIED, P_FAILED = "confirmed", "blocker found", "no blocker, some tables not visible", "could not check"
PLAN_SIGNALS = [  # (regex on the analyzed + physical plan, blocker)
    (r"AIRemoteFunction\(|\bAI Query \[|\b(ai_[a-z_]+|vector_search)\(", "AI function"),
    (r"\b(PhotonScalarUDF|BatchEvalPython\w*|ArrowEvalPython\w*|PythonUDF)\b", "Python UDF"),
    (r"\b(SecureViewUnaryNode|PhotonSecureView)\b", "row filter / column mask"),
    (r"\b(st_[a-z0-9_]+|h3_[a-z0-9_]+)\(", "spatial function / GEO type"),
    (r"\b(xpath[a-z_]*|from_xml|to_xml|schema_of_xml)\(", "XML / XPath function"),
    (r"\binput_file_(name|block_start|block_length)\(", "input_file_* function"),
    (r"\bScan JDBC\b|\bJDBCRelation\b", "foreign / federated table"),
    (r"uc-deltasharing://(?!system\.)", "Delta Sharing table"),
    (r"\bStreamingRelation", "streaming read"),
]
QUOTED = re.compile(r"('(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"|`[^`]*`|--[^\n]*|/\*.*?\*/)", re.S)  # strings, identifiers, comments
NAME = r"(?:`[^`]+`|[\w$-]+)"
TABLE_REFS = re.compile(rf"SubqueryAlias ({NAME}\.{NAME}\.{NAME})\s*$|View \(({NAME}\.{NAME}\.{NAME}),", re.M)


def outside_quotes(sql, fn):
    parts = QUOTED.split(sql)
    return "".join(fn(p) if i % 2 == 0 else p for i, p in enumerate(parts))


def explain(sql):
    try:
        p = "\n".join(r[0] or "" for r in spark.sql("EXPLAIN EXTENDED " + sql).collect())
    except Exception as e:
        return None, str(e).splitlines()[0][:200]
    if "== Analyzed Logical Plan ==" not in p:  # planning errors come back as text rows, the reason after the first
        lines = [x.strip() for x in p.splitlines() if x.strip() and not x.startswith("Error occurred during query planning")]
        return None, " ".join(lines[:2])[:200] or "planning error"
    return p, None


def plan_one(sample):
    txt = (sample["txt"] or "").strip().rstrip(";").strip()
    srcs = [s for s in (sample["srcs"] if sample["srcs"] is not None else []) if s and s.count(".") == 2]
    if not txt:
        return {"plan_status": P_FAILED, "plan_error": "no query text"}
    if (sample["len"] or 0) > PLAN_MAX_CHARS:
        return {"plan_status": P_FAILED, "plan_error": f"query text longer than {PLAN_MAX_CHARS:,} characters"}
    if ";" in "".join(QUOTED.split(txt)[0::2]):
        return {"plan_status": P_FAILED, "plan_error": "more than one statement"}
    notes = []
    p, err = explain(txt)
    if p is None and "UNBOUND_SQL_PARAMETER" in err:
        txt = outside_quotes(txt, lambda s: re.sub(r"\?", "NULL", re.sub(r"(?<![:\w]):[A-Za-z_]\w*", "NULL", s)))
        notes.append("parameters set to NULL")
        p, err = explain(txt)
    if p is None and srcs and "TABLE_OR_VIEW_NOT_FOUND" in err:  # unqualified names: retry in the schema(s) lineage saw it read
        cur = (spark.catalog.currentCatalog(), spark.catalog.currentDatabase())
        try:
            for ns in list(dict.fromkeys(s.rsplit(".", 1)[0] for s in srcs))[:3]:
                spark.catalog.setCurrentCatalog(ns.split(".")[0])
                spark.catalog.setCurrentDatabase(ns.split(".")[1])
                p, _ = explain(txt)
                if p is not None:
                    notes.append(f"planned in schema {ns}")
                    break
        except Exception:
            pass
        finally:  # restore the session's catalog and schema (the original schema may not exist, e.g. `default`)
            for set_current, value in ((spark.catalog.setCurrentCatalog, cur[0]), (spark.catalog.setCurrentDatabase, cur[1])):
                try:
                    set_current(value)
                except Exception:
                    pass
    if p is None:
        return {"plan_status": P_FAILED, "plan_error": err, "plan_note": ", ".join(notes)}

    analyzed = p.split("== Analyzed Logical Plan ==", 1)[1].split("== Optimized Logical Plan ==", 1)[0]
    physical = p.split("== Physical Plan ==", 1)[1] if "== Physical Plan ==" in p else ""
    names = {(a or b).replace("`", "").lower() for a, b in TABLE_REFS.findall(analyzed)}
    names = {"hive_metastore." + n.split(".", 1)[1] if n.startswith("spark_catalog.") else n for n in names}
    missing = re.search(r"^\s*missing = (.*)$", p, re.M)
    shape_ok = "parameters set to NULL" not in notes  # NULL filters can prune the physical plan
    return {
        "plan_status": None,  # set once table support is known
        "plan_signals": [label for rx, label in PLAN_SIGNALS if re.search(rx, analyzed + physical)],
        "names": names,
        "joins": len(re.findall(r"(SortMergeJoin|ShuffledHashJoin|BroadcastHashJoin|BroadcastNestedLoopJoin|CartesianProduct)\b", physical)) if shape_ok else None,
        "shuffles": len(re.findall(r"ShuffleExchangeSink|\bExchange (hash|range|RoundRobin|Single)", physical)) if shape_ok else None,
        "missing_stats": missing.group(1).strip() if missing else "",
        "plan_note": ", ".join(notes),
    }


plan = None
pat = main[main.lvl == "pattern"].copy()
# Existing Lakehouse//RT warehouses are out of scope: only what runs on DBSQL today is assessed.
rt_ids = set(inventory[inventory.warehouse_type.fillna("").str.upper() == "REAL_TIME"].warehouse_id) if inventory is not None else set()
if PLAN_CHECK and len(pat):
    pat = pat[~pat.warehouse_id.isin(rt_ids)]
    pat["cand"] = sum(pd.to_numeric(pat[c]) for c in ("n_eligible", "n_check", "n_tune"))
    todo = pat.sort_values("cand", ascending=False).drop_duplicates("pattern_id").head(PLAN_MAX_PATTERNS)
    rows, t0 = [], time.time()
    for pid, sample in zip(todo.pattern_id, todo.plan_sample):
        if time.time() - t0 > PLAN_BUDGET_S:
            break
        rows.append({"pattern_id": pid, **(plan_one(sample) if sample is not None else {"plan_status": P_FAILED, "plan_error": "no sample"})})

    # Support status of every table seen in the plans (same rules as the main pass).
    names = sorted(set().union(*[r.get("names") or set() for r in rows]))
    status = {}
    for n in names:
        if n.startswith("system."):
            status[n] = "system table"
        elif n.startswith("hive_metastore."):
            status[n] = "Hive metastore table"
        elif ".information_schema." in n:
            status[n] = "information_schema view"
    rest = [n for n in names if n not in status]
    for i in range(0, len(rest), 500):
        chunk = rest[i:i + 500]
        t = run(f"table status for plan check ({i + len(chunk)}/{len(rest)})", f"""
          WITH {TBL}
          SELECT full_name, table_blocker FROM tbl WHERE full_name IN ({', '.join(repr(x) for x in chunk)})
        """)
        if t is not None:
            status.update({f: b or "supported" for f, b in zip(t.full_name, t.table_blocker)})

    for r in rows:
        if r["plan_status"] is not None:
            continue
        hidden = sorted(n for n in r["names"] if n not in status)
        blockers = list(dict.fromkeys(r["plan_signals"] + sorted({status[n] for n in r["names"] if status.get(n, "supported") != "supported"})))
        r["plan_status"] = P_BLOCKED if blockers else P_UNVERIFIED if hidden else P_OK
        r["plan_blockers"] = "; ".join(blockers)
        r["tables_in_plan"] = len(r["names"])
        r["tables_not_visible"] = ", ".join(hidden[:5])
    plan = pd.DataFrame(rows).drop(columns=["names", "plan_signals"], errors="ignore")
    for c in ["plan_blockers", "tables_not_visible", "plan_note", "plan_error", "missing_stats"]:
        plan[c] = plan[c].fillna("") if c in plan else ""
    counts = plan.plan_status.value_counts().to_dict()
    skipped = len(todo) - len(plan)
    print(f"✓ query plan check: {len(plan)} patterns in {time.time() - t0:.0f}s: "
          + ", ".join(f"{k} {v}" for k, v in counts.items()) + (f" ({skipped} skipped, time budget reached)" if skipped else ""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Analyse

# COMMAND ----------

LABEL = {  # display order
    "ELIGIBLE": "Qualifies now",
    "CHECK": "Probably qualifies (table not visible to you, so not verified)",
    "TUNE": "Qualifies after tuning (runs 20 s – 2 min today)",
    "BLK_TOOL": "Blocked: tool not supported yet (Genie / Jobs)",
    "BLK_SQL": "Blocked: unsupported SQL feature",
    "BLK_TABLE": "Blocked: unsupported table type / governance",
    "BLK_GUARD": "Blocked: exceeds preview guardrails",
    "BLK_PLAN": "Blocked: found in query plan (e.g. inside a view)",
    "CACHE": "Served from result cache today (no engine work)",
    "WRITE": "Not eligible: writes / DDL / maintenance (incl. failed attempts)",
    "META": "Excluded: metadata & session commands",
    "FAILED": "Excluded: failed / canceled",
}
LABEL["TUNE"] = LABEL["TUNE"].replace("20 s", f"{ELIG_S} s")
RT_SIZE_DBU = {"Small": 12, "Medium": 24, "Large": 48, "X-Large": 96}


def num(df, cols):
    for c in cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def pct(a, b):
    return round(100.0 * a / b, 1) if b else 0.0


def first(x, i):
    return float(x[i]) if x is not None and len(x) > i and x[i] is not None else None


reasons = main[main.lvl == "reason"][["warehouse_id", "category", "reason", "n"]].copy()
patterns = main[main.lvl == "pattern"].copy()
wh = main[main.lvl == "warehouse"].drop(columns=["lvl", "pattern_id", "category", "reason"]).copy()
num(wh, [c for c in wh.columns if c.startswith(("n", "e_", "task_", "queue_", "elig_avg", "elig_10min", "written_"))])
for df in (wh, patterns):
    df["elig_p50_ms"] = df["elig_pct_ms"].apply(lambda x: first(x, 0))
    df["elig_p95_ms"] = df["elig_pct_ms"].apply(lambda x: first(x, 1))

# --- Apply the query plan check: executions of patterns whose plan shows a blocker move to "Blocked: found in query plan" ---
ADDITIVE = ["n_eligible", "n_check", "n_tune", "n_tune_files", "n_elig_sub1s", "n_elig_1_5s", "n_elig_5s_plus", "n_elig_queued",
            "queue_ms_eligible", "task_ms_eligible", "task_ms_tune", "n_movable_in_write_sessions",
            *[c for c in wh.columns if c.startswith("e_")]]
num(patterns, ADDITIVE + ["n", "elig_avg_read_bytes"])
reasons["n"] = pd.to_numeric(reasons["n"])
for c in ["n_blk_plan", "n_elig_plan_ok", "n_check_plan_ok"]:
    wh[c] = 0
if plan is not None and len(plan):
    patterns = patterns.merge(plan, on="pattern_id", how="left")
    patterns.loc[patterns.warehouse_id.isin(rt_ids), "plan_status"] = None  # same query shape on a Real-Time warehouse
    bad = patterns[patterns.plan_status == P_BLOCKED]
    if len(bad):
        sub = bad.groupby("warehouse_id")[ADDITIVE].sum().reindex(wh.warehouse_id).fillna(0)
        wh[ADDITIVE] = wh[ADDITIVE].values - sub.values
        wh["n_blk_plan"] = (sub.n_eligible + sub.n_check + sub.n_tune).values
        moves = []
        for r in bad.itertuples():
            first_blocker = r.plan_blockers.split("; ")[0]
            moves += [(r.warehouse_id, "ELIGIBLE", R_SUB1, -r.n_elig_sub1s), (r.warehouse_id, "ELIGIBLE", R_1_5, -r.n_elig_1_5s),
                      (r.warehouse_id, "ELIGIBLE", R_5_PLUS, -r.n_elig_5s_plus), (r.warehouse_id, "CHECK", R_CHECK, -r.n_check),
                      (r.warehouse_id, "TUNE", R_TUNE_RT, -(r.n_tune - r.n_tune_files)), (r.warehouse_id, "TUNE", R_TUNE_FILES, -r.n_tune_files),
                      (r.warehouse_id, "BLK_PLAN", first_blocker, r.n_eligible + r.n_check + r.n_tune)]
        reasons = (pd.concat([reasons, pd.DataFrame(moves, columns=reasons.columns)])
                   .groupby(["warehouse_id", "category", "reason"], as_index=False)["n"].sum())
        reasons = reasons[reasons.n > 0]
    ok = (patterns[patterns.plan_status == P_OK].groupby("warehouse_id")[["n_eligible", "n_check"]].sum()
          .reindex(wh.warehouse_id).fillna(0))
    wh["n_elig_plan_ok"], wh["n_check_plan_ok"] = ok.n_eligible.values, ok.n_check.values
patterns["plan_status"] = patterns.get("plan_status", pd.Series(index=patterns.index, dtype=object)).fillna("not checked")

if inventory is not None:
    wh = wh.merge(inventory.drop(columns=["workspace_id"]), on="warehouse_id", how="left")
else:
    for c in ["warehouse_name", "warehouse_type", "warehouse_size", "min_clusters", "max_clusters", "auto_stop_minutes", "deleted"]:
        wh[c] = None
if billing is not None:
    wh = wh.merge(num(billing, ["dbus", "list_cost", "peak_hour_dbu", "billed_hours", "rt_list_price", "serverless_price"]),
                  on="warehouse_id", how="left")
else:
    for c in ["dbus", "list_cost", "peak_hour_dbu", "billed_hours", "rt_list_price", "serverless_price", "sku_name"]:
        wh[c] = None
peak_cols = ["peak_qpm", "p95_qpm", "peak_qps", "peak_concurrency", "p95_concurrency", "active_minutes"]
if peak is not None:
    wh = wh.merge(num(peak, peak_cols), on="warehouse_id", how="left")
else:
    for c in peak_cols:
        wh[c] = None

wh["warehouse_name"] = wh["warehouse_name"].fillna("(not in system.compute.warehouses)")
wh["is_rt"] = wh["warehouse_type"].fillna("").str.upper().eq("REAL_TIME")
wh["n_exec"] = wh.n_reads - wh.n_cache  # SELECTs the engine actually executed
wh["eligible_pct"] = [pct(a, b) for a, b in zip(wh.n_eligible, wh.n_exec)]
wh["eligible_upper_pct"] = [pct(a + c, b) for a, c, b in zip(wh.n_eligible, wh.n_check, wh.n_exec)]
wh["tune_pct"] = [pct(a, b) for a, b in zip(wh.n_tune, wh.n_exec)]
wh["cache_pct"] = [pct(a, b) for a, b in zip(wh.n_cache, wh.n_reads)]
wh["eligible_compute_pct"] = [pct(a, b) for a, b in zip(wh.task_ms_eligible, wh.task_ms_all)]
wh["eligible_per_day"] = (wh.n_eligible / DAYS).round(0)
wh["latency_upside_pct"] = [pct(a + b, c) for a, b, c in zip(wh.n_elig_1_5s, wh.n_elig_5s_plus, wh.n_eligible)]
wh["queued_pct"] = [pct(a, b) for a, b in zip(wh.n_elig_queued, wh.n_eligible)]
wh["lineage_coverage_pct"] = [pct(a, b) for a, b in zip(wh.n_reads_with_lineage, wh.n_reads)]
wh["plan_checked_pct"] = [pct(a, b) for a, b in zip(wh.n_elig_plan_ok, wh.n_eligible)]  # qualifying queries whose plan was confirmed


def top_blockers(wid, n_exec, k=3):
    r = reasons[(reasons.warehouse_id == wid) & reasons.category.str.startswith(("BLK", "TUNE"))]
    r = r.sort_values("n", ascending=False).head(k)
    return "; ".join(f"{x.reason} ({pct(x.n, n_exec)}%)" for x in r.itertuples()) or "—"


wh["top_blockers"] = [top_blockers(w, n) for w, n in zip(wh.warehouse_id, wh.n_exec)]


def fit(r):
    if r.is_rt:
        return "0 Already Lakehouse//RT"
    if r.n_exec == 0:
        return "6 No executed reads"
    if r.eligible_pct >= 60 and r.eligible_per_day >= 500:
        return "1 Strong fit"
    if r.eligible_pct >= 40 and r.eligible_per_day >= 50:
        return "2 Good fit"
    if r.eligible_pct + r.tune_pct >= 40 and (r.n_eligible + r.n_tune) / DAYS >= 50:
        return "3 Fit after tuning"
    if r.n_eligible > 0:
        return "4 Weak fit"
    return "5 Not a fit"


wh["fit"] = wh.apply(fit, axis=1)
wh["fit_label"] = wh["fit"].str[2:]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Confidence score
# MAGIC Every input is a share between 0 and 1. *Movable* queries are qualifying queries plus result-cache hits that pass every rule
# MAGIC (they can be served from Lakehouse//RT too).
# MAGIC
# MAGIC | Factor | Meaning | Formula |
# MAGIC |---|---|---|
# MAGIC | **Q**: qualifying | Executed SELECTs that meet every Lakehouse//RT rule | qualifying ÷ executed SELECTs |
# MAGIC | **W**: write-free | Lakehouse//RT can't write. Full credit with no writes, falling to 0 at 5% writes | max(0, 1 − write share ÷ 5%) |
# MAGIC | **V**: verified | Qualifying queries whose tables could be checked (in information_schema, or in the query plan). Unverified ones count half | 0.5 + 0.5 × verified share |
# MAGIC | **S**: sample size | Enough history to trust the result: 100 queries → 0.5, 1,000 → 0.75, ≥ 10,000 → 1 | min(1, log10(1 + n) ÷ 4) |
# MAGIC | **D**: demand | Lakehouse//RT pays off on frequent low-latency queries: 10/day → 0.35, 100/day → 0.67, ≥ 1,000/day → 1 | min(1, log10(1 + qualifying per day) ÷ 3) |
# MAGIC | **X**: separable | Movable queries that run in sessions without writes. Mixed sessions need an app change, so they count half | 1 − 0.5 × mixed-session share |
# MAGIC
# MAGIC - **Full-fit confidence** = Q × W × V × S(executed SELECTs) × (0.6 + 0.4 D)
# MAGIC - **Partial-fit confidence** = (0.5 + 0.5 × min(1, Q ÷ 50%)) × X × V × S(qualifying queries) × (0.4 + 0.6 D)
# MAGIC
# MAGIC **What can move.** **Whole warehouse (full fit)**: ≥ 80% of executed SELECTs qualify and writes are ≤ 1% of statements.
# MAGIC **Subset (partial fit)**: not a full fit, but a qualifying subset with partial-fit confidence ≥ 25% could move to a separate
# MAGIC Real-Time warehouse; the rest (writes, blocked queries) stays where it is. **Not enough data**: fewer than 50 executed SELECTs.
# MAGIC Confidence bands: ≥ 75% high, 50–74% medium, < 50% low. The score is a transparent heuristic for evidence strength,
# MAGIC not a probability. Confirm with a proof of concept.

# COMMAND ----------

FULL_MIN_Q, FULL_MAX_WRITES, WRITE_ZERO_AT, PARTIAL_MIN_CONF, MIN_SAMPLE = 0.80, 0.01, 0.05, 25, 50
WHOLE, SUBSET = "Whole warehouse (full fit)", "Subset (partial fit)"


def sample(n):
    return min(1.0, math.log10(1 + n) / 4)


def demand(per_day):
    return min(1.0, math.log10(1 + per_day) / 3)


def band(c):
    return "high" if c >= 75 else "medium" if c >= 50 else "low"


def fmt(n):
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.1f}k" if n >= 1e3 else f"{n:.0f}"


def score(r):
    n_movable = r.n_eligible + r.n_cache
    n_work = r.n_reads + r.n_write                       # finished reads + writes
    write_share = r.n_write / n_work if n_work else (1.0 if r.n_write_attempts else 0.0)
    Q = r.n_eligible / r.n_exec if r.n_exec else (1.0 if r.n_cache else 0.0)
    W = max(0.0, 1 - write_share / WRITE_ZERO_AT)
    V = 0.5 + 0.5 * ((r.n_eligible + r.n_check_plan_ok) / (r.n_eligible + r.n_check)) if (r.n_eligible + r.n_check) else 0.5
    D = demand(r.n_eligible / DAYS)
    X = 1 - 0.5 * (r.n_movable_in_write_sessions / n_movable) if n_movable else 1.0
    full = 100 * Q * W * V * sample(r.n_exec) * (0.6 + 0.4 * D)
    partial = 100 * (0.5 + 0.5 * min(1.0, Q / 0.5)) * X * V * sample(r.n_eligible) * (0.4 + 0.6 * D)

    if r.is_rt:
        move, conf = "— (already Lakehouse//RT)", None
    elif n_work == 0:
        move, conf = "Nothing: no finished queries", 0.0
    elif r.n_reads == 0:
        move, conf = "Nothing: write-only", 0.0
    elif r.n_exec < MIN_SAMPLE:
        move, conf = "Not enough data", max(full, partial)
    elif Q >= FULL_MIN_Q and write_share <= FULL_MAX_WRITES:
        move, conf = WHOLE, full
    elif r.n_eligible > 0 and partial >= PARTIAL_MIN_CONF:
        move, conf = SUBSET, partial
    else:
        move, conf = "Nothing yet", max(full, partial)

    return pd.Series({
        "can_move": move,
        "confidence_pct": None if conf is None else round(conf),
        "confidence_band": None if conf is None else band(conf),
        "full_fit_confidence_pct": round(full),
        "partial_fit_confidence_pct": round(partial),
        "writes": "yes" if r.n_write_attempts else "no",
        "write_pct": round(100 * write_share, 1),
        "movable_pct_of_queries": pct(n_movable, n_work),     # how much of the warehouse could move
        "movable_per_day": round(n_movable / DAYS),
        "n_movable": n_movable,
        "Q": round(Q, 2), "W": round(W, 2), "V": round(V, 2), "S": round(sample(r.n_exec), 2), "D": round(D, 2), "X": round(X, 2),
    })


wh = pd.concat([wh, wh.apply(score, axis=1)], axis=1)
wh["confidence_pct"] = pd.to_numeric(wh["confidence_pct"], errors="coerce").astype("Int64")


def why(r):
    if r.is_rt:
        return "Already a Real-Time warehouse."
    parts = []
    if r.n_exec:
        parts.append(f"{r.eligible_pct}% of executed SELECTs qualify ({fmt(r.n_eligible / DAYS)}/day)")
    if r.n_write_attempts:
        top = reasons[(reasons.warehouse_id == r.warehouse_id) & (reasons.category == "WRITE")].sort_values("n", ascending=False).head(3)
        kinds = ", ".join(f"{x.reason} {fmt(x.n)}" for x in top.itertuples())
        src = f"; mostly from {r.top_write_source}" if isinstance(r.top_write_source, str) else ""
        parts.append(f"writes are {r.write_pct}% of statements ({kinds}{src})")
        if r.can_move == WHOLE:
            parts.append("those few writes need to run on another warehouse")
    else:
        parts.append("no writes")
    if r.can_move == SUBSET:
        parts.append(f"{r.movable_pct_of_queries}% of the warehouse's queries could move to a separate Real-Time warehouse")
    if r.n_movable and r.n_movable_in_write_sessions / r.n_movable > 0.05:
        parts.append(f"{pct(r.n_movable_in_write_sessions, r.n_movable)}% of movable queries share a session with writes (needs a separate connection)")
    if r.n_eligible + r.n_check and r.n_check / (r.n_eligible + r.n_check) > 0.05:
        parts.append(f"{pct(r.n_check, r.n_eligible + r.n_check)}% of candidate queries read tables you can't see (not verified)")
    if r.n_blk_plan:
        parts.append(f"query plans found blockers in {fmt(r.n_blk_plan)} queries that passed the text check (see 3.7)")
    if r.plan_checked_pct:
        parts.append(f"query plans confirm {r.plan_checked_pct}% of qualifying queries")
    if r.n_exec and r.top_blockers != "—":
        parts.append("held back by " + r.top_blockers.replace("; ", ", "))
    if r.S < 0.75:
        parts.append(f"small sample ({int(r.n_exec):,} executed SELECTs), so confidence is reduced")
    return "; ".join(parts) + "."


wh["why"] = wh.apply(why, axis=1)
wh = wh.sort_values(["fit", "confidence_pct", "n_eligible"], ascending=[True, False, False]).reset_index(drop=True)

# COMMAND ----------

# --- Cost model (indicative, list prices) ---
def rt_price(r):
    if RT_PRICE_OVERRIDE is not None:
        return RT_PRICE_OVERRIDE
    for p in (r["rt_list_price"], r["serverless_price"]):
        if p is not None and not pd.isna(p):
            return float(p)
    return 0.70


wh["elig_share"] = [
    (a / b) if b else (c / d if d else 0) for a, b, c, d in zip(wh.task_ms_eligible, wh.task_ms_all, wh.n_eligible, wh.n_exec)
]
wh["rt_price_per_dbu"] = wh.apply(rt_price, axis=1)
wh["current_price_per_dbu"] = [(c / d) if d else None for c, d in zip(wh.list_cost.fillna(0), wh.dbus.fillna(0))]
wh["rt_query_size"] = ["Small" if (p or 0) <= 5000 else "Medium" for p in wh.elig_p95_ms]
wh["current_cost_month"] = wh.list_cost.fillna(0) * MONTH
wh["qualifying_compute_dbus_month"] = wh.dbus.fillna(0) * wh.elig_share * MONTH
# A Real-Time warehouse bills at least its query size while up; uptime ≈ 10-minute windows that have qualifying queries.
wh["rt_uptime_hours_month"] = wh.elig_10min_windows / 6 * MONTH
floor_dbus = wh.rt_uptime_hours_month * wh.rt_query_size.map(RT_SIZE_DBU)
for f in FACTORS:
    rt_dbus = (wh.qualifying_compute_dbus_month / f).combine(floor_dbus, max)
    wh[f"rt_cost_month_{f:g}x"] = rt_dbus * wh.rt_price_per_dbu
wh["suggested_max_dbu_per_hour"] = [
    max(RT_SIZE_DBU[s], math.ceil(p / 12) * 12) if p and not pd.isna(p) else RT_SIZE_DBU[s]
    for p, s in zip(wh.peak_hour_dbu, wh.rt_query_size)
]
candidates = wh[wh.fit.str[0].isin(["1", "2", "3"])]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Results

# COMMAND ----------

non_rt = wh[~wh.is_rt]
tot = non_rt[["n", "n_reads", "n_exec", "n_cache", "n_eligible", "n_check", "n_tune", "task_ms_all", "task_ms_eligible",
              "n_elig_1_5s", "n_elig_5s_plus", "n_elig_queued", "n_blk_plan", "n_elig_plan_ok"]].sum()
whole, subset = non_rt[non_rt.can_move == WHOLE], non_rt[non_rt.can_move == SUBSET]
writers = non_rt[non_rt.writes == "yes"]
f_lo, f_hi = FACTORS[0], FACTORS[-1]
cand_now = candidates.current_cost_month.sum()
cand_rt_lo = candidates[f"rt_cost_month_{f_lo:g}x"].sum()
cand_rt_hi = candidates[f"rt_cost_month_{f_hi:g}x"].sum()
cost_note = "" if billing is not None else " (billing tables not accessible, so costs are missing)"
rt_range = (f"${cand_rt_lo:,.0f}/month" if round(cand_rt_hi) == round(cand_rt_lo)
            else f"${cand_rt_hi:,.0f} – ${cand_rt_lo:,.0f}/month ({f_hi:g}× to {f_lo:g}× scenario)")


if plan is not None and len(plan):
    _pc = plan.plan_status.value_counts()
    plan_row = (f"<b>{len(plan)}</b> most frequent patterns planned with EXPLAIN: {_pc.get(P_OK, 0)} confirmed, "
                f"{_pc.get(P_BLOCKED, 0)} with a blocker hidden from the text check, {_pc.get(P_UNVERIFIED, 0) + _pc.get(P_FAILED, 0)} not verifiable · "
                f"{pct(tot.n_blk_plan, tot.n_exec)}% of executed SELECTs moved to blocked · plans confirm {pct(tot.n_elig_plan_ok, tot.n_eligible)}% of qualifying queries")
else:
    plan_row = "not run (widget 10 is off, or no query patterns)"


def row(k, v):
    return f'<tr><td style="padding:3px 18px 3px 0;color:#444;vertical-align:top">{k}</td><td>{v}</td></tr>'


def bands(df):
    return ", ".join(f"{b}: {(df.confidence_band == b).sum()}" for b in ("high", "medium", "low"))


def top_list(df, k=3):
    return "<br>".join(f"{r.warehouse_name} <span style='color:#666'>({r.warehouse_id})</span>: <b>{r.confidence_pct}%</b> "
                       f"({r.confidence_band}), {r.movable_pct_of_queries}% of queries movable"
                       for r in df.sort_values("confidence_pct", ascending=False).head(k).itertuples()) or "—"


displayHTML(f"""
<div style="font-family:sans-serif;line-height:1.45">
<h2 style="margin-bottom:6px">Lakehouse//RT fit: last {DAYS} days</h2>
<table style="border-collapse:collapse;font-size:15px">
{row("Warehouses analysed", f"<b>{len(non_rt):,}</b> ({int(wh.is_rt.sum())} already Lakehouse//RT, excluded)")}
{row("Finished SELECTs", f"<b>{int(tot.n_reads):,}</b>, of which {pct(tot.n_cache, tot.n_reads)}% were served from the result cache")}
{row("Executed SELECTs that qualify today", f"<b>{pct(tot.n_eligible, tot.n_exec)}%</b> (up to {pct(tot.n_eligible + tot.n_check, tot.n_exec)}% counting tables you can't see) · {pct(tot.task_ms_eligible, tot.task_ms_all)}% of warehouse compute time")}
{row("Could qualify after tuning", f"<b>+{pct(tot.n_tune, tot.n_exec)}%</b> (run {ELIG_S} s – 2 min today)")}
{row("Strong / good / after-tuning fits", f"<b>{(wh.fit.str[0] == '1').sum()} / {(wh.fit.str[0] == '2').sum()} / {(wh.fit.str[0] == '3').sum()}</b> warehouses")}
{row("Whole warehouse can move (full fit)", f"<b>{len(whole)}</b> · confidence {bands(whole)}")}
{row("Only a subset can move (partial fit)", f"<b>{len(subset)}</b> · confidence {bands(subset)}")}
{row("Warehouses that write", f"<b>{len(writers)}</b> of {len(non_rt)} (these can't move as a whole)")}
{row("Top full fits", top_list(whole))}
{row("Top partial fits", top_list(subset))}
{row("Query plan check", plan_row)}
{row("Qualifying queries taking ≥ 1 s today", f"<b>{pct(tot.n_elig_1_5s + tot.n_elig_5s_plus, tot.n_eligible)}%</b> (where users would notice a speed-up) · {pct(tot.n_elig_queued, tot.n_eligible)}% waited &gt; 1 s in a queue")}
{row("Candidate warehouses, cost today", f"<b>${cand_now:,.0f}/month</b> for the whole warehouses (list price){cost_note}")}
{row("Est. Lakehouse//RT cost for their qualifying load", f"<b>{rt_range}</b> (list price)")}
</table>
<p style="color:#666;font-size:12px">List prices, not contract prices. Qualifying % is of SELECTs the engine actually ran (result-cache hits excluded).
The Lakehouse//RT estimate is the larger of (a) today's compute for those queries divided by the scenario factor and (b) the minimum charge for the
suggested query size over the 10-minute windows that have qualifying queries. Promotional pricing may apply; check with your field team for promo pricing eligibility.
Confidence % is a transparent heuristic for evidence strength (section 2), not a probability. Confirm with a proof of concept.</p>
</div>""")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Which warehouses are a good fit
# MAGIC > **Start with the `why` column (3rd column).** It explains each warehouse's verdict in one sentence: how much qualifies,
# MAGIC > whether and what it writes, how much of it could move, and what holds the rest back. Read it before quoting any score.
# MAGIC
# MAGIC **Strong fit**: ≥ 60% of executed SELECTs qualify and ≥ 500 qualifying queries/day. **Good fit**: ≥ 40% and ≥ 50/day.
# MAGIC **Fit after tuning**: the same thresholds when "after tuning" queries are counted too. **Weak fit**: some queries qualify, below those thresholds.
# MAGIC `can_move` and `confidence_%` come from the confidence score (section 2): the whole warehouse (full fit) or only a subset (partial fit).
# MAGIC `movable_%_of_queries` is the share of the warehouse's finished queries (reads and writes) that could run on Lakehouse//RT.
# MAGIC The second table shows the score factors behind each confidence.
# MAGIC `latency_upside_%` is the share of qualifying queries that take ≥ 1 s today. `queued_%` is the share that waited > 1 s for capacity.
# MAGIC `plan_confirmed_%_of_qualifying` is the share of qualifying queries whose pattern passed the query plan check, and
# MAGIC `blocked_by_query_plan` counts queries moved to blocked by it (section 3.7).

# COMMAND ----------

display(wh[[
    "fit_label", "can_move", "why", "confidence_pct", "confidence_band", "warehouse_name", "warehouse_id", "workspace_id",
    "eligible_pct", "movable_pct_of_queries", "writes", "write_pct", "warehouse_type", "warehouse_size",
    "n_reads", "cache_pct", "n_exec", "n_eligible", "eligible_upper_pct", "tune_pct", "eligible_compute_pct",
    "eligible_per_day", "elig_p50_ms", "elig_p95_ms", "latency_upside_pct", "queued_pct",
    "peak_qpm", "peak_qps", "peak_concurrency", "n_users", "top_source", "top_blockers", "plan_checked_pct", "n_blk_plan",
]].rename(columns={
    "fit_label": "fit", "confidence_pct": "confidence_%", "movable_pct_of_queries": "movable_%_of_queries",
    "write_pct": "write_%_of_statements", "n_reads": "selects", "cache_pct": "result_cache_%", "n_exec": "executed_selects",
    "n_eligible": "qualifying", "eligible_pct": "qualifying_%", "eligible_upper_pct": "qualifying_%_incl_unverified",
    "tune_pct": "after_tuning_%", "eligible_compute_pct": "qualifying_%_of_compute", "eligible_per_day": "qualifying_per_day",
    "elig_p50_ms": "qualifying_p50_ms", "elig_p95_ms": "qualifying_p95_ms", "latency_upside_pct": "latency_upside_%",
    "queued_pct": "queued_%", "peak_concurrency": "peak_concurrency_est", "n_users": "users",
    "plan_checked_pct": "plan_confirmed_%_of_qualifying", "n_blk_plan": "blocked_by_query_plan",
}))

display(non_rt[[
    "fit_label", "can_move", "confidence_pct", "warehouse_name", "warehouse_id", "full_fit_confidence_pct", "partial_fit_confidence_pct",
    "Q", "W", "V", "S", "D", "X", "n_exec", "n_eligible", "n_cache", "n_check", "n_tune", "n_write_attempts", "n_write",
    "written_rows", "top_write_source", "n_movable_in_write_sessions", "n_blk_plan", "n_check_plan_ok",
]].rename(columns={
    "fit_label": "fit", "confidence_pct": "confidence_%", "n_exec": "executed_selects", "n_eligible": "qualifying",
    "n_cache": "result_cache_hits", "n_check": "unverified_tables", "n_tune": "needs_tuning", "n_write_attempts": "write_attempts",
    "n_write": "finished_writes", "n_movable_in_write_sessions": "movable_in_write_sessions",
    "n_blk_plan": "blocked_by_query_plan", "n_check_plan_ok": "unverified_confirmed_by_plan",
}))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 What percentage of queries qualify, and why the rest don't
# MAGIC Covers all warehouses except existing Lakehouse//RT ones. Each statement is counted once, under the first rule it fails.
# MAGIC Percentages are of executed SELECTs.

# COMMAND ----------

r_all = reasons[reasons.warehouse_id.isin(non_rt.warehouse_id)].groupby(["category", "reason"], as_index=False)["n"].sum()
READ_CATS = ["ELIGIBLE", "CHECK", "TUNE", "BLK_TOOL", "BLK_SQL", "BLK_TABLE", "BLK_GUARD", "BLK_PLAN"]
cat_sum = r_all.groupby("category", as_index=False)["n"].sum()
cat_sum["%_of_executed_selects"] = [pct(n, tot.n_exec) if c in READ_CATS else None for c, n in zip(cat_sum.category, cat_sum.n)]
cat_sum["%_of_all_statements"] = [pct(n, tot.n) for n in cat_sum.n]
cat_sum["order"] = cat_sum.category.map({k: i for i, k in enumerate(LABEL)})
cat_sum["category"] = cat_sum.category.map(LABEL)
display(cat_sum.sort_values("order").drop(columns="order").rename(columns={"n": "statements"}))

detail = r_all[r_all.category.isin(READ_CATS)].copy()
detail["%_of_executed_selects"] = [pct(n, tot.n_exec) for n in detail.n]
detail["order"] = detail.category.map({k: i for i, k in enumerate(LABEL)})
detail["category"] = detail.category.map(LABEL)
display(detail.sort_values(["order", "n"], ascending=[True, False]).drop(columns="order").rename(columns={"n": "statements"}))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 Benefits for candidate warehouses
# MAGIC **Latency**: Lakehouse//RT targets sub-second answers at high concurrency. Databricks reports response times as low as 10 ms
# MAGIC and sub-100 ms latency at 12,000 QPS. The biggest visible gain is for qualifying queries that take ≥ 1 s today.
# MAGIC
# MAGIC **Concurrency**: Lakehouse//RT autoscaling adds compute in small steps (measured in DBUs, not whole clusters),
# MAGIC so queries that queue today are the second signal to look at.
# MAGIC
# MAGIC **Cost (indicative)**: the whole warehouse's list cost today, next to the estimated Lakehouse//RT cost for its qualifying load.
# MAGIC Each scenario divides today's compute for those queries by its factor (1× = no speed-up assumed). The estimate never goes below
# MAGIC the minimum charge for the suggested query size during the hours with qualifying traffic.
# MAGIC Writes and non-qualifying queries keep running on the existing warehouse. All costs are at list price.
# MAGIC Promotional pricing may apply; check with your field team for promo pricing eligibility. Validate with a proof of concept.

# COMMAND ----------

cost_cols = [f"rt_cost_month_{f:g}x" for f in FACTORS]
benefits = candidates if len(candidates) else non_rt.head(10)
display(benefits[[
    "fit_label", "warehouse_name", "warehouse_id", "warehouse_type", "warehouse_size", "sku_name",
    "n_eligible", "elig_p50_ms", "elig_p95_ms", "n_elig_1_5s", "n_elig_5s_plus", "latency_upside_pct",
    "n_elig_queued", "queued_pct", "p95_qpm", "peak_qpm", "p95_concurrency", "peak_concurrency",
    "current_price_per_dbu", "current_cost_month", "eligible_compute_pct", "qualifying_compute_dbus_month",
    "rt_price_per_dbu", "rt_query_size", "rt_uptime_hours_month", *cost_cols, "suggested_max_dbu_per_hour",
]].round(2).rename(columns={
    "fit_label": "fit", "n_eligible": "qualifying", "elig_p50_ms": "p50_ms_today", "elig_p95_ms": "p95_ms_today",
    "n_elig_1_5s": "qualifying_1_5s", "n_elig_5s_plus": f"qualifying_5_{ELIG_S}s", "n_elig_queued": "queued_gt_1s",
    "current_cost_month": "warehouse_cost_month_today", "eligible_compute_pct": "qualifying_%_of_compute",
}))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4 Readiness checks for candidate warehouses
# MAGIC Lakehouse//RT only accepts the Statement Execution API (REST). Clients that already use it are ready. Other drivers need a setting:
# MAGIC Python `use_kernel=True`, Node.js `useKernel: true`, JDBC `UseThriftClient=0`, and ODBC needs the Databricks OSS ODBC driver
# MAGIC (the Simba ODBC driver isn't supported). For Power BI, use the ADBC driver with REST (SEA).
# MAGIC Lakehouse//RT always runs in ANSI mode, so test sessions that set `ANSI_MODE = false` first.

# COMMAND ----------

display(benefits[[
    "warehouse_name", "warehouse_id", "n_eligible",
    "e_src_dashboard", "e_src_legacy_dashboard", "e_src_sql_editor", "e_src_notebook", "e_src_alert", "e_src_external",
    "e_conn_rest", "e_conn_python", "e_conn_jdbc", "e_conn_node", "e_conn_odbc", "e_conn_other", "e_powerbi",
    "n_ansi_off", "n_blk_tool", "lineage_coverage_pct", "top_client",
]].rename(columns={
    "n_eligible": "qualifying", "e_src_dashboard": "from_aibi_dashboards", "e_src_legacy_dashboard": "from_legacy_dashboards",
    "e_src_sql_editor": "from_sql_editor", "e_src_notebook": "from_notebooks", "e_src_alert": "from_alerts",
    "e_src_external": "from_external_clients", "e_conn_rest": "ext_rest_ready", "e_conn_python": "ext_python_set_use_kernel",
    "e_conn_jdbc": "ext_jdbc_set_UseThriftClient_0", "e_conn_node": "ext_node_set_useKernel", "e_conn_odbc": "ext_odbc_switch_driver",
    "e_conn_other": "ext_other_check", "e_powerbi": "power_bi", "n_ansi_off": "sessions_setting_ansi_off",
    "n_blk_tool": "genie_or_job_selects", "lineage_coverage_pct": "lineage_coverage_%",
}))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.5 Tables
# MAGIC First, the tables read by warehouse queries that Lakehouse//RT can't read yet. Convert them to UC managed tables, or keep those
# MAGIC queries on DBSQL. "Not visible" means the user running this notebook has no access to the table's metadata.
# MAGIC Second, a layout check of the most-read supported tables: liquid clustering, file sizes, predictive optimization.

# COMMAND ----------

if tables is not None and len(tables):
    tables = num(tables, ["n_select_queries", "n_warehouses"])
    display(tables[tables.rt_status != "supported"].head(50))
else:
    print("Table lineage isn't available, so table-level checks are skipped. The SQL-text check still flags system and Hive metastore tables.")

# COMMAND ----------

layout = []
if INSPECT_TABLES and tables is not None and len(tables):
    top_ok = tables[(tables.rt_status == "supported") & tables.table_type.fillna("").str.upper().str.startswith("MANAGED")]
    for name, n_q in zip(top_ok.table_name.head(10), top_ok.n_select_queries.head(10)):
        ident = ".".join(f"`{p}`" for p in name.split("."))
        row_ = {"table_name": name, "n_select_queries": int(n_q)}
        try:
            d = spark.sql(f"DESCRIBE DETAIL {ident}").collect()[0].asDict()
            ext = {r.col_name: r.data_type for r in spark.sql(f"DESCRIBE TABLE EXTENDED {ident}").collect()}
        except Exception as e:
            row_["recommendation"] = "could not inspect: " + str(e).splitlines()[0][:160]
            layout.append(row_)
            continue
        row_.update(size_gb=round((d.get("sizeInBytes") or 0) / GiB, 2), num_files=d.get("numFiles"),
                    clustering=", ".join(d.get("clusteringColumns") or []) or None,
                    partitioning=", ".join(d.get("partitionColumns") or []) or None,
                    cluster_by_auto=d.get("clusterByAuto"),
                    predictive_optimization=ext.get("Predictive Optimization"))
        tips = []
        if not row_["clustering"] and not row_["partitioning"] and row_["size_gb"] >= 1:
            tips.append("add liquid clustering on filter/join columns (or CLUSTER BY AUTO)")
        if row_["num_files"] and row_["num_files"] > 1000 and row_["size_gb"] * 1024 / row_["num_files"] < 32:
            tips.append("many small files: run OPTIMIZE")
        if not (row_["predictive_optimization"] or "").upper().startswith("ENABLE"):
            tips.append("enable predictive optimization")
        row_["recommendation"] = "; ".join(tips) or "layout looks fine"
        layout.append(row_)
    print(f"Inspected {len(layout)} tables with DESCRIBE (metadata only).")
    if layout:
        display(pd.DataFrame(layout))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.6 Top query patterns (POC candidates)
# MAGIC Queries that qualify, probably qualify, or qualify after tuning, grouped by shape with literals replaced.
# MAGIC `plan_status` is the result of the query plan check (3.7). A POC should replay a representative production mix at real concurrency,
# MAGIC not one or two queries.

# COMMAND ----------

if len(patterns):
    patterns["warehouse_name"] = patterns.warehouse_id.map(wh.set_index("warehouse_id")["warehouse_name"])
    patterns["fit"] = patterns.warehouse_id.map(wh.set_index("warehouse_id")["fit"])
    patterns["avg_read_mb"] = (patterns.elig_avg_read_bytes / 1024 ** 2).round(1)
    p = patterns[patterns.warehouse_id.isin(benefits.warehouse_id)].sort_values(["fit", "warehouse_id", "n"], ascending=[True, True, False])
    p["fit"] = p.fit.str[2:]
    display(p[["fit", "warehouse_name", "warehouse_id", "pattern_id", "n", "n_eligible", "n_check", "n_tune", "plan_status",
               "elig_p50_ms", "elig_p95_ms", "avg_read_mb", "n_elig_queued", "top_source", "top_client", "sample_text"]].rename(columns={
        "n": "executions", "n_eligible": "qualifying", "n_check": "probably_qualifying", "n_tune": "after_tuning",
        "elig_p50_ms": "qualifying_p50_ms", "elig_p95_ms": "qualifying_p95_ms", "n_elig_queued": "queued_gt_1s"}))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.7 Query plan check
# MAGIC One row per planned query pattern (all warehouses except Lakehouse//RT ones), blockers first. `plan_blockers` lists what the plan
# MAGIC shows that Lakehouse//RT can't run, often hidden inside a view. `joins` and `shuffles` describe the shape: simple filters and lookups
# MAGIC (0–1 shuffles) are the best fit; many joins and shuffles mean the query may need tuning to get low latency.
# MAGIC `missing_stats` lists tables without optimizer statistics (run `ANALYZE TABLE … COMPUTE STATISTICS FOR ALL COLUMNS`).
# MAGIC "Could not check" usually means the user running this notebook can't read a table (`INSUFFICIENT_PERMISSIONS`; run the notebook
# MAGIC as a metastore admin to cover them), or the query needs session settings (`TABLE_OR_VIEW_NOT_FOUND` for unqualified names).

# COMMAND ----------

if plan is not None and len(plan):
    order = {P_BLOCKED: 0, P_UNVERIFIED: 1, P_FAILED: 2, P_OK: 3}
    pv = patterns[patterns.plan_status != "not checked"].copy()
    pv["order"] = pv.plan_status.map(order)
    cols = ["plan_status", "plan_blockers", "warehouse_name", "warehouse_id", "pattern_id", "n", "n_eligible", "n_check", "n_tune",
            "tables_in_plan", "tables_not_visible", "joins", "shuffles", "missing_stats", "plan_note", "plan_error", "sample_text"]
    display(pv.sort_values(["order", "n"], ascending=[True, False])[[c for c in cols if c in pv]].rename(columns={
        "n": "executions", "n_eligible": "qualifying", "n_check": "probably_qualifying", "n_tune": "after_tuning"}))
else:
    print("The query plan check did not run (widget 10 is off, or there are no query patterns).")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.8 Summary (copy/paste)

# COMMAND ----------

lines = [
    f"Lakehouse//RT fit assessment: last {DAYS} days, {len(non_rt)} SQL warehouses, {int(tot.n):,} statements "
    f"({int(tot.n_exec):,} executed SELECTs, plus {int(tot.n_cache):,} served from the result cache).",
    f"- {pct(tot.n_eligible, tot.n_exec)}% of executed SELECTs qualify today "
    f"(up to {pct(tot.n_eligible + tot.n_check, tot.n_exec)}% counting tables that couldn't be verified). "
    f"That is {pct(tot.task_ms_eligible, tot.task_ms_all)}% of warehouse compute time. "
    f"Another {pct(tot.n_tune, tot.n_exec)}% could qualify after tuning (they run {ELIG_S} s – 2 min today).",
    f"- {pct(tot.n_elig_1_5s + tot.n_elig_5s_plus, tot.n_eligible)}% of qualifying queries take ≥ 1 s today, and "
    f"{pct(tot.n_elig_queued, tot.n_eligible)}% waited > 1 s in a queue.",
]
blk = detail[detail.category.isin([LABEL[c] for c in ("TUNE", "BLK_TOOL", "BLK_SQL", "BLK_TABLE", "BLK_GUARD", "BLK_PLAN")])]
if len(blk):
    lines.append("- Main reasons queries don't qualify: " + "; ".join(
        f"{r.reason} ({pct(r.n, tot.n_exec)}%)" for r in blk.sort_values("n", ascending=False).head(4).itertuples()) + ".")
if plan is not None and len(plan):
    _pc = plan.plan_status.value_counts()
    lines.append(f"- Query plans of the {len(plan)} most frequent query patterns were checked with EXPLAIN (planned, not run): "
                 f"{_pc.get(P_OK, 0)} confirmed, {_pc.get(P_BLOCKED, 0)} showed a blocker the text check missed "
                 f"({pct(tot.n_blk_plan, tot.n_exec)}% of executed SELECTs, now counted as blocked). "
                 f"Plans confirm {pct(tot.n_elig_plan_ok, tot.n_eligible)}% of qualifying queries.")
lines.append(f"- {len(whole)} warehouses can move as a whole (full fit), {len(subset)} can move a subset of their queries "
             f"(partial fit); {len(writers)} of {len(non_rt)} warehouses write.")
if len(candidates):
    lines.append(f"- Strong/good/after-tuning candidates cost ${cand_now:,.0f}/month today (list price, whole warehouses). Their qualifying "
                 f"load is estimated at {rt_range} on Lakehouse//RT at list price. Promotional pricing may apply; check with your field "
                 f"team for promo pricing eligibility.")


def summary_line(r):
    p95 = 0 if pd.isna(r.elig_p95_ms) else int(r.elig_p95_ms)
    qpm = 0 if pd.isna(r.peak_qpm) else int(r.peak_qpm)
    return (f"  • {r.warehouse_name} ({r.warehouse_id}): {r.fit_label}, {r.confidence_pct}% confidence ({r.confidence_band}). {r.why} "
            f"Qualifying p95 {p95:,} ms, peak {qpm:,} queries/min. Today ${r.current_cost_month:,.0f}/month for the whole warehouse; "
            f"qualifying load est. ${getattr(r, f'rt_cost_month_{f_lo:g}x'):,.0f}/month on Lakehouse//RT ({f_lo:g}×, list price). "
            f"Start with query size {r.rt_query_size} and max {r.suggested_max_dbu_per_hour} DBU/h.")


for title, df in (("Full fits: the whole warehouse can move", whole),
                  ("Partial fits: a subset can move to a separate Real-Time warehouse", subset)):
    if len(df):
        lines.append(f"- {title}:")
        lines += [summary_line(r) for r in df.sort_values("confidence_pct", ascending=False).head(8).itertuples()]
if not len(whole) and not len(subset):
    lines.append("- No warehouse is a full or partial fit yet. See the 'why' column in 3.1 for what holds each one back.")
lines.append("- Next step: enable the Lakehouse//RT preview, create a Real-Time warehouse, and replay the top query patterns (3.6) at production concurrency.")
print("\n".join(lines))
