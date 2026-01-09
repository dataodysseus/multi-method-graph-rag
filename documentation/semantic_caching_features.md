# Semantic SQL Caching

## Semantic Caching Strategy 

**Similar Queries**
> "Show me top 5 high-value customers"  
> "List top 5 customers that spent the most"  
> "Who are my top 5 most valuable customers"  

All generate the same SQL but call expensive LLM endpoint each time. A solution to this is: 
**Semantic SQL Cache = Smart caching using embeddings**
---
### **Traditional Caching (Exact Match):**
```
Query: "Show me top 5 high-value customers"
Cache Key: MD5("Show me top 5 high-value customers")
Result: Cache miss for slightly different wording
```
### **Semantic Caching (Similarity Match):**
```
Query: "Show me top 5 high-value customers"
1. Embed question → [0.23, -0.15, 0.87, ...]
2. Compare with cached embeddings (cosine similarity)
3. If similarity > 0.85 → Cache hit!
4. Reuse cached SQL

Query: "List top 5 customers that spent the most"
1. Embed question → [0.25, -0.14, 0.89, ...]  (very similar)
2. Similarity = 0.92 > 0.85 → Cache HIT
3. Reuse same SQL (no LLM call needed)
```
**Key Insight:** Semantic meaning, not exact text
---
## Impact Analysis
### **Cost Savings:**

| Scenario | Without Cache | With Cache (50% hit rate) | Savings |
|----------|--------------|---------------------------|---------|
| 100 queries/month | $5.00 | $2.50 | $2.50/mo |
| 1,000 queries/month | $50.00 | $25.00 | $25/mo |
| 10,000 queries/month | $500.00 | $250.00 | $250/mo |
| 100,000 queries/month | $5,000.00 | $2,500.00 | $2,500/mo |

**At scale:** $30,000/year saved for 100K queries/month.
---
### **Performance Improvement:**

| Operation | Time | Speedup |
|-----------|------|---------|
| **LLM SQL Generation** | 3.9s | Baseline |
| **Cache Lookup** | 0.1s | 39x faster |

**User Experience:** Near-instant response for cached queries
---
### **Consistency:**

| Without Cache | With Cache |
|--------------|------------|
| LLM might generate slightly different SQL each time |  Same question = same SQL always |
| Results might vary |  Consistent results |
| Hard to debug |  Predictable behavior |

---

## Architecture

### **Cache Table Schema:**

```sql
CREATE TABLE accenture.sales_analysis.llm_sql_cache (
    cache_id STRING,                    -- MD5 hash of question
    question_text STRING,               -- Original question
    question_embedding ARRAY<FLOAT>,    -- 384-dim vector
    generated_sql STRING,               -- LLM-generated SQL
    query_metadata STRING,              -- JSON with intent, params
    created_at TIMESTAMP,               -- First cached
    last_used_at TIMESTAMP,             -- Last reused
    usage_count INT,                    -- How many times reused
    avg_execution_time_ms DOUBLE,       -- SQL performance
    success_rate DOUBLE                 -- % of successful executions
)
```
### **Cache Lookup Flow:**
<img width="1024" height="1024" alt="Gemini_Generated_Image_nh4fdwnh4fdwnh4f" src="https://github.com/user-attachments/assets/d510be34-039e-4ec0-805d-d45d4a88f4cc" />

---

## Implementation Process

### **Create Cache**

```python
from semantic_sql_cache import SemanticSQLCache
from sentence_transformers import SentenceTransformer

# Initialize embedding model (same as main system)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Create cache
cache = SemanticSQLCache(
    spark=spark,
    embedding_model=embedding_model,
    catalog="accenture",
    schema="sales_analysis",
    cache_table="llm_sql_cache",
    similarity_threshold=0.85,  # 85% similarity for cache hit
    ttl_days=90  # Keep cache entries for 90 days
)
```
---
### **Enable for LLM Generator**

```python
from semantic_sql_cache import enhance_llm_with_cache

# Enable caching
enhance_llm_with_cache(rag_system.sql_builder.llm_generator, cache)
```
---

### **Use Normally**

```python
# First query (cache miss)
answer1 = rag_system.query("Show me top 5 high-value customers")
# Output: CACHE MISS -> Calls LLM (3.9s)

# Similar query (cache hit)
answer2 = rag_system.query("List top 5 customers that spent the most")
# Output: CACHE HIT (similarity: 0.89) -> Reuses SQL (0.1s)

# Another similar query (cache hit)
answer3 = rag_system.query("Who are my top 5 most valuable customers")
# Output: CACHE HIT (similarity: 0.87) -> Reuses SQL (0.1s)

# Result: 2 LLM calls saved ($0.10 saved, 7.6s saved)
```
---
## Similarity Threshold Tuning

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| 0.95 (Very strict) | Only nearly identical questions hit cache | High precision, lower hit rate |
| 0.85 (Recommended) | Semantically similar questions hit | Balanced precision/recall |
| 0.75 (Lenient) | Loosely related questions hit | Higher hit rate, lower precision |

### **Examples:**
```python
Q1: "Show top 5 customers by revenue"
Q2: "List top 5 customers by sales"
Similarity: 0.92 → Hit at all thresholds

Q1: "Show top 5 customers"
Q2: "Show top 10 customers"
Similarity: 0.88 → Hit at 0.85, 0.75

Q1: "Show top customers"
Q2: "Show top items"
Similarity: 0.76 → Only hit at 0.75

Q1: "High-value customers"
Q2: "Top locations"
Similarity: 0.42 → No hit at any threshold
```
### **Recommendation:**

Start with **0.85** and adjust based on:
- **Too many false positives?** → Increase to 0.90
- **Too many cache misses?** → Decrease to 0.80
---
## Monitoring Cache Performance
```python
stats = cache.get_statistics()

print(f"Total Queries: {stats['total_queries']}")
print(f"Cache Hits: {stats['cache_hits']}")
print(f"Hit Rate: {stats['hit_rate_percent']:.1f}%")
print(f"Cost Savings: ${stats['estimated_cost_savings_usd']:.2f}")
print(f"Time Saved: {stats['time_saved_seconds']:.1f}s")
```
### **Target Metrics:**
| Metric | Target | Status |
|--------|--------|--------|
| Hit Rate | 50-70% |  Good after warmup |
| Cache Size | 100-1000 entries |  Scales with usage |
| Avg Similarity | 0.85-0.95 | ✅ High-quality matches |
### **Cache Warmup:**
```
Queries 1-50: Hit rate ~10% (building cache)
Queries 51-200: Hit rate ~30% (warming up)
Queries 200+: Hit rate ~50-70% (mature cache)
```
---
### **Example 1: Customer Value Queries**
```python
# All these map to same SQL:
queries = [
    "Show me top 5 high-value customers",
    "List top 5 customers that spent the most",
    "Who are my top 5 most valuable customers",
    "Top 5 customers by purchase amount",
    "5 customers with highest spending",
    "Best customers by revenue"
]
```
---
### **Example 2: Top Items Queries**
```python
queries = [
    "What are my best-selling products?",
    "Show me top items by revenue",
    "Which products generate most sales?",
    "Top-performing items",
    "Best products by sales volume"
]
```
---
### **Example 3: Location Analysis**
```python
queries = [
    "Which store locations perform best?",
    "Show me top locations by sales",
    "What are my most profitable stores?",
    "Best-performing store locations",
    "Top stores by revenue"
]
```
---
## ROI Calculator
```python
# Input your usage
QUERIES_PER_MONTH = 1000  # Your monthly query volume
LLM_COST_PER_QUERY = 0.05  # Claude Sonnet 4.5 cost
EXPECTED_HIT_RATE = 0.50  # 50% after warmup

# Calculate
monthly_cost_without_cache = QUERIES_PER_MONTH * LLM_COST_PER_QUERY
monthly_cost_with_cache = QUERIES_PER_MONTH * (1 - EXPECTED_HIT_RATE) * LLM_COST_PER_QUERY
monthly_savings = monthly_cost_without_cache - monthly_cost_with_cache
annual_savings = monthly_savings * 12

print(f"Monthly Savings: ${monthly_savings:.2f}")
print(f"Annual Savings: ${annual_savings:.2f}")
```
### **Typical Results:**
| Queries/Month | Cost Without | Cost With | Savings/Year |
|---------------|-------------|-----------|--------------|
| 1,000 | $50 | $25 | $300 |
| 5,000 | $250 | $125 | $1,500 |
| 10,000 | $500 | $250 | $3,000 |
| 50,000 | $2,500 | $1,250 | $15,000 |
---
### **1. Cache Analytics**

```python
# View top queries
top_queries = cache.get_top_queries(n=10)
display(top_queries)

# Output:
# question_text                          usage_count  last_used_at
# "Show top 5 customers by revenue"      47          2026-01-04 15:23:11
# "Best-selling products"                32          2026-01-04 15:22:45
# ...
```
---
### **2. Cache Cleanup**
```python
# Remove old entries
cache.cleanup_old_entries(days=90)
```
---
### **3. Manual Cache Population - Recommended**

```python
# Pre-populate with common queries
common_queries = [
    "Show top 5 customers by revenue",
    "Top 10 items by sales",
    "Best locations by performance"
]
for query in common_queries:
    sql = llm_generator.generate_sql(query)
    cache.save_to_cache(query, sql)
```
---
### **4. Cache Inspection**

```sql
-- View cache contents
SELECT 
    question_text,
    substring(generated_sql, 1, 100) as sql_preview,
    usage_count,
    created_at,
    last_used_at
FROM accenture.sales_analysis.llm_sql_cache
ORDER BY usage_count DESC
LIMIT 10;
```
---
