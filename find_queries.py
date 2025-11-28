"""
Script to find all SQL queries with ? placeholders in persistence.py
This will help us update them to use the correct placeholder based on database type
"""

import re
from pathlib import Path

persistence_file = Path(__file__).parent / "rct_field_flow" / "persistence.py"

content = persistence_file.read_text()

# Find all SQL queries with ? placeholder
pattern = r'(execute|executemany)\s*\(\s*["\']([^"\']*\?[^"\']*)["\']'
matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)

print(f"Found {len(matches)} queries with ? placeholders:\n")

for i, (method, query) in enumerate(matches, 1):
    # Clean up query for display
    clean_query = ' '.join(query.split())[:100]
    print(f"{i}. {method}(...)")
    print(f"   {clean_query}...")
    print()

print("\nRecommendation: Create a helper function to replace ? with %s for PostgreSQL")
