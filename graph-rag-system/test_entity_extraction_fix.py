# Test Entity Extraction Fix

from graph_rag_core import GraphRAGConfig
from graph_rag_intelligence import QueryIntentClassifier

# Configure
config = GraphRAGConfig(
    catalog="accenture",
    schema="sales_analysis",
    fact_table="items_sales",
    dimension_tables=["item_details", "store_location", "customer_details"],
    fk_mappings={
        "items_sales": {
            "item_id": "item_details",
            "location_id": "store_location",
            "customer_id": "customer_details"
        }
    }
)

# Create classifier
classifier = QueryIntentClassifier(config)

print("="*80)
print("TESTING ENTITY EXTRACTION FIX")
print("="*80)

# Test cases that were failing
test_cases = [
    {
        'query': 'List top 5 customers by number of items purchased',
        'expected_entity': 'customer_details',
        'reason': 'Primary subject is "customers", not "items"'
    },
    {
        'query': 'Name my top 5 customers by number of items purchased',
        'expected_entity': 'customer_details',
        'reason': 'Query starts with action on "customers"'
    },
    {
        'query': 'Show customers who bought the most items',
        'expected_entity': 'customer_details',
        'reason': 'Subject is "customers", "items" is just context'
    },
    {
        'query': 'Which customers purchased over 100 items?',
        'expected_entity': 'customer_details',
        'reason': '"customers" is the question subject'
    },
    {
        'query': 'List items purchased by top customers',
        'expected_entity': 'item_details',
        'reason': 'Primary subject is "items", customers is context'
    },
    {
        'query': 'Show me top items by sales',
        'expected_entity': 'item_details',
        'reason': 'Clearly about items'
    },
    {
        'query': 'Top locations by customer count',
        'expected_entity': 'store_location',
        'reason': 'Primary subject is "locations"'
    },
    {
        'query': 'Which stores have the most customers?',
        'expected_entity': 'store_location',
        'reason': '"stores" is the question subject'
    }
]

passed = 0
failed = 0

for i, test in enumerate(test_cases, 1):
    query = test['query']
    expected = test['expected_entity']
    reason = test['reason']
    
    # Classify
    intent = classifier.classify(query)
    actual = intent.get('entity_type')
    
    # Check result
    success = actual == expected
    
    if success:
        status = " PASS"
        passed += 1
    else:
        status = " FAIL"
        failed += 1
    
    print(f"\n{'-'*80}")
    print(f"Test {i}: {status}")
    print(f"{'-'*80}")
    print(f"Query: {query}")
    print(f"Expected: {expected}")
    print(f"Actual: {actual}")
    print(f"Reason: {reason}")
    
    if not success:
        print(f" MISMATCH - Entity extraction needs further tuning")

print(f"\n{'='*80}")
print(f"RESULTS: {passed} passed, {failed} failed")
print(f"{'='*80}")

if failed == 0:
    print(" All tests passed! Entity extraction is working correctly.")
else:
    print(f" {failed} test(s) failed. Review entity extraction logic.")
