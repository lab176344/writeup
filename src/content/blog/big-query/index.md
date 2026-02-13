---
title: "A Guide to BigQuery Optimisation: Performance, Cost, and Architecture"
description: "An overview of BigQuery architecture, storage strategies, and query optimisation techniques based on real-world case studies."
pubDate: "2026-02-13"
tags: ["big-query", "gcp", "data-engineering", "sql-optimisation"]
---

BigQuery is a tool for processing large datasets, but its performance and
cost depend on how it is used. Without an understanding of its architecture,
queries can be slow and expensive.
 Unlike traditional relational databases where performance is often
measured by execution time, BigQuery demands a shift in focus: **you are billed
for the data you scan, not the time you spend scanning it.**

We are focusing only on the on-demand pay-as-you-go model, as it is the most common and relevant for optimisation.

## 1. Understanding the Architecture: Storage vs. Compute

To optimise BigQuery, you must first understand how it separates storage from
compute. Your data lives in **Colossus**, Google’s distributed file system,
stored in a proprietary columnar format called **Capacitor**. When you execute a
query, the **Dremel** engine allocates compute slots to process that data.

The connection between these two layers is a high-bandwidth network.
 While this network ensures fast data movement from storage to compute,
data movement *between* compute slots (shuffling) during complex
operations like joins or aggregations is a common performance bottleneck.
The initial load bottleneck is usually the sheer volume of data being
pulled from storage.

### The Billing Model

* **On-Demand:** Costs depend on the amount of data scanned.
* **Storage:** Costs depend on the volume of data stored, with lower rates for
  data that has not been modified for 90 days.

**Key Takeaway:** A 5-second query isn't necessarily a cheap query. Always watch
the "bytes processed" metric rather than the clock.

---

## 2. Data Modelling: Denormalisation and Data Types

In traditional SQL (Postgres, MySQL), normalisation is the standard approach. In
BigQuery, **denormalisation** is often the path to efficiency.

### Denormalisation vs. Joins

Joins in BigQuery require "shuffling"—moving data between slots to find matching
keys. This is compute-intensive and expensive. By pre-joining data into flat or
nested structures, you reduce the need for shuffles.

### Choosing Appropriate Data Types

Every byte counts in a columnar store. Using `INT64` (8 bytes) instead of a
`STRING` representation can save significant costs over billions of rows.

* **STRING:** 2 bytes + UTF-8 length.
* **NUMERIC:** 16 bytes (use only for high-precision financial data).
* **DATE/TIMESTAMP:** 8 bytes.

```sql
-- Efficient Table Definition
CREATE TABLE analytics.events (
  user_id INT64,             -- 8 bytes
  event_type STRING,         -- 2 bytes + length
  event_date DATE,           -- 8 bytes
  metadata JSON              -- Use sparsely for performance
)
CLUSTER BY user_id;
```

### Primary and Foreign Keys

In traditional databases, primary keys are essential for performance (indexing)
and data integrity. BigQuery's approach is different:

* **Optional and Non-Enforced:** Primary and foreign keys are optional. If
  defined, they are **not enforced** by BigQuery during data ingestion. You are
  responsible for ensuring data uniqueness and referential integrity.
* **Performance Optimisation:** While not enforced, defining these keys can
  improve query performance. The query optimiser uses the constraint information
  to apply better join strategies (such as eliminating unnecessary shuffles) and
  improving join ordering.
* **Cost of Enforcement:** Because BigQuery is a distributed system, enforcing
  uniqueness at scale would require significant cross-slot coordination, which
  would slow down data loading and increase costs.

---

## 3. Storage Optimisation: Partitioning and Clustering

These are effective methods for reducing query costs.

### Partitioning

Partitioning divides your table into segments (usually by date). When you filter
by the partition column, BigQuery "prunes" the partitions, ignoring the data it
doesn't need.

```sql
-- Partitioning by day with a mandatory filter
CREATE TABLE analytics.logs (
  timestamp TIMESTAMP,
  message STRING
)
PARTITION BY TIMESTAMP_TRUNC(timestamp, DAY)
OPTIONS (
  require_partition_filter = true
);
```

### Clustering

Clustering sorts the data within partitions based on up to four columns. This is
particularly effective for columns with high cardinality (like `user_id` or
`country_code`).

**Note:** The order of clustering columns matters.
column, then the second, and so on. Your `WHERE` clauses should follow this
order for maximum effect.

---

## 4. Advanced Data Structures: Arrays and Structs

BigQuery allows you to store related data in a single row using `ARRAY` and
`STRUCT`. This significantly reduces row counts and avoids expensive joins.

### The "Global IoT Sensor" Case Study

Analysis of a high-frequency IoT sensor dataset illustrates this. By
restructuring a flat table of **1.2 billion sensor readings (450 GB)** into a
nested structure where readings were grouped into hourly arrays per device, the
dataset was reduced to **60 million rows (280 GB)**—a 95% reduction in row count
and a 38% reduction in storage size.

#### Example: Normalised vs. Nested

```sql
-- Normalised (Flat): Requires a JOIN and shuffles data for every reading.
SELECT 
  d.device_name, 
  s.reading 
FROM `project.dataset.sensors` s
JOIN `project.dataset.devices` d ON s.device_id = d.device_id;

-- Nested (BigQuery): Data is co-located. No JOIN or shuffle required.
SELECT 
  device_name, 
  r.reading 
FROM `project.dataset.device_readings`,
UNNEST(readings) AS r;
```

### Querying Nested Data

Use the `UNNEST` keyword to flatten arrays back into a relational format for
analysis.

```sql
-- Querying a nested user activity table
SELECT
  user_id,
  session.page_path,
  session.duration
FROM `project.dataset.user_activity`,
UNNEST(sessions) AS session
WHERE session.date = '2024-01-01';
```

---

## 5. Modern Search Capabilities: Search Indexes

For logs or text-heavy data, standard `LIKE '%term%'` filters are slow and
expensive. BigQuery's **Search Indexes** allow for efficient pattern
matching.

```sql
-- Creating a Search Index
CREATE SEARCH INDEX idx_logs
ON analytics.server_logs(ALL COLUMNS);

-- Using the SEARCH function
SELECT *
FROM analytics.server_logs
WHERE SEARCH(server_logs, 'error "critical failure"');
```

---

## 6. Advanced SQL Techniques: Window Functions

Window functions allow you to perform calculations across a set of rows related
to the current row, without collapsing them into a single output row.

### Key Window Functions

* **Ranking:** `ROW_NUMBER()`, `RANK()`, `DENSE_RANK()`.
* **Analytical:** `LAG()`, `LEAD()` (comparing to previous/next rows).
* **Aggregation:** `SUM() OVER()`, `AVG() OVER()`.

```sql
-- Calculating a 7-day moving average of revenue
SELECT
  date,
  revenue,
  AVG(revenue) OVER (
    ORDER BY date
    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
  ) AS moving_avg_7d
FROM sales_table;
```

### Optimisation: `EXISTS` and `IN` for Semi-Joins

In BigQuery, using `EXISTS` or `IN` with a subquery is an optimisation that
the engine translates into a **Semi-Join** or **Anti-Join**.

* **Efficiency:** A standard `JOIN` can produce duplicate rows if the join key is
  not unique, requiring a `DISTINCT` operation. `EXISTS` and `IN` stop processing
  a match as soon as the first one is found, reducing **Slot Time** and
  **Bytes Shuffled**.
* **Performance:** Use `EXISTS` for correlated subqueries and `IN` for
  uncorrelated subqueries or static lists.
* **Anti-Joins:** `NOT EXISTS` is more reliable for anti-joins because it handles
  `NULL` values differently than `NOT IN`. If a `NOT IN` subquery returns a
  `NULL`, the query will return zero results.

#### Example: Semi-Join vs. Inner Join

```sql
-- Standard Join: Might produce duplicate rows if a user has multiple orders,
-- requiring an expensive DISTINCT.
SELECT DISTINCT u.user_id, u.name
FROM `project.dataset.users` u
JOIN `project.dataset.orders` o ON u.user_id = o.user_id;

-- Optimised with EXISTS (Semi-Join): Stops as soon as the first order is found.
-- More efficient for large datasets.
SELECT u.user_id, u.name
FROM `project.dataset.users` u
WHERE EXISTS (
  SELECT 1 FROM `project.dataset.orders` o WHERE o.user_id = u.user_id
);

-- Optimised with IN (Uncorrelated): Best for static lists or independent subqueries.
SELECT user_id, name
FROM `project.dataset.users`
WHERE user_id IN (SELECT user_id FROM `project.dataset.orders`);
```

---

## 7. Practical Performance Tuning: The Developer's Workflow

1. **Filter Early:** Apply `WHERE` clauses as close to the source as possible.
2. **Select Only Needed Columns:** Avoid `SELECT *`. Every extra column adds to
   the bytes scanned.
3. **Aggregate Late:** Join small tables first, then aggregate.
4. **Use CTEs for Clarity:** Common Table Expressions (CTEs) make your code more
   readable, but remember that BigQuery does not "materialise" them—using the
   same CTE multiple times will result in multiple scans. Use temporary tables
   for heavy intermediate results.

---

## 8. Analysing the Execution Graph: Identifying Bottlenecks

The execution graph provides a detailed view of how your query is processed. It
shows how the **Dremel** engine breaks your SQL into multiple stages.

### Comparison of Execution Metrics

* **Slot Time:** The total compute resources consumed. High slot time relative to
  execution time indicates high parallelism. Extremely high slot time with slow
  performance can indicate inefficient operations like cross-joins.
* **Wait Time (ms):** The time a stage waited for an available slot. High wait
  times indicate that the project has reached its slot limit.
* **Bytes Shuffled:** The amount of data moved between stages. Large shuffle
  volumes are a bottleneck and can often be reduced through clustering or better
  data modelling.
* **Compute Time (Avg vs Max):** A large difference between Average and Maximum
  compute time in a stage indicates **Data Skew**, where a small number of slots
  process more data than others.

### Join Patterns in the Graph

BigQuery uses two strategies for joins:

1. **Broadcast Join:** For joins with small tables (usually < 10MB), BigQuery
   sends the small table to every slot processing the large table. This avoids
   a shuffle.
2. **Hash Join:** For joins between large tables, BigQuery shuffles both tables
   based on the join key.

### Common Bottlenecks and Solutions

<!-- markdownlint-disable MD013 -->
| Symptom | Probable Cause | Solution |
| :--- | :--- | :--- |
| **High Slot Time** | Complex `REGEXP`, UDFs, or large cross-joins. | Simplify logic; use `SEARCH` for text; ensure join keys are efficient. |
| **High Wait Time** | Slot contention or hitting project limits. | Switch to Capacity pricing (Reservations) or optimise query to use fewer slots. |
| **High Bytes Shuffled** | Joining large tables on non-clustered columns. | Cluster both tables on the join keys; use nested/repeated fields to avoid joins. |
| **Data Skew** | Join keys are unevenly distributed (e.g., many NULLs). | Filter out NULLs before joining; distribute frequently occurring keys using a "salting" technique. |
| **Exploding Joins** | **Low Cardinality** on join keys (Many-to-Many). | Check join logic; ensure you aren't creating a Cartesian product inadvertently. |
<!-- markdownlint-enable MD013 -->

---

## Final Thoughts

Efficiency in BigQuery involves understanding the separation of storage and
compute. By using denormalisation, partitioning, and clustering, you can
reduce the amount of data processed and the resources consumed.

Always keep an eye on the execution graph. It tells you more about your query's
health than the execution timer ever will.

### Citations and Further Reading

* [Google Cloud: BigQuery Explained - Storage Overview](https://cloud.google.com/blog/products/data-analytics/bigquery-explained-storage-overview)
* [Official BigQuery Pricing Guide](https://cloud.google.com/bigquery/pricing)
* [A Guide to Using Window Functions](https://medium.com/data-science/a-guide-to-using-window-functions-4b2768f589d9)
* [Mastering Arrays in BigQuery (2024)](https://medium.com/@thomas.ellyatt/mastering-arrays-in-bigquery-2024-e62612b15c30)
* [A Guide to Search Indexes in BigQuery](https://medium.com/@thomas.ellyatt/a-guide-to-search-indexes-in-bigquery-3d2b586e10fb)
* [BigQuery Efficiency: How I Reduced Table Size by 35%](https://towardsdatascience.com/bigquery-efficiency-how-i-reduced-my-table-size-by-35-5-and-rows-by-93-1-dc8b9b7276ff/)
* [Clustering for Improved Performance (2024)](https://medium.com/@thomas.ellyatt/clustering-for-improved-performance-in-bigquery-2024-55c9285828dd)
* [Why Partitioning Tables is Essential (2024)](https://medium.com/@thomas.ellyatt/why-partitioning-tables-is-essential-in-bigquery-2024-6e771c9fc288)
* [Ultimate Guide to Saving Time and Money with BigQuery](https://medium.com/@thomas.ellyatt/ultimate-guide-to-saving-time-and-money-with-bigquery-f2cddc1af1ad)
* [Burn Data Rather Than Money with BigQuery: The Definitive Guide](https://towardsdatascience.com/burn-data-rather-than-money-with-bigquery-the-definitive-guide-1b50a9fdf096/)
* [Query Optimisation Best Practices for Data Analysts](https://medium.com/@sheerwolff/query-optimization-in-bigquery-best-practices-for-data-analysts-9aba6cf68fea)
* [Why Your 5-Second BigQuery Query Isn’t Cheap](https://luminousmen.substack.com/p/why-your-5-second-bigquery-query)
