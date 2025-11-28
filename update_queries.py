"""
Script to update all SQL execute statements in persistence.py to use _query() wrapper
"""

import re
from pathlib import Path

persistence_file = Path(__file__).parent / "rct_field_flow" / "persistence.py"

# Read the file
content = persistence_file.read_text(encoding='utf-8')

# Pattern to match SQL execute statements with string literals (both single and double quotes)
# We need to wrap the SQL string with _query()

patterns = [
    # Pattern 1: cur.execute("SELECT...", ...)
    (r'cur\.execute\(\s*"([^"]+)"\s*,', r'cur.execute(_query("\1"),'),
    # Pattern 2: cur.execute('SELECT...'  , ...)
    (r"cur\.execute\(\s*'([^']+)'\s*,", r"cur.execute(_query('\1'),"),
    # Pattern 3: conn.execute("SELECT...", ...)
    (r'conn\.execute\(\s*"([^"]+)"\s*,', r'conn.execute(_query("\1"),'),
    # Pattern 4: conn.execute('SELECT...', ...)
    (r"conn\.execute\(\s*'([^']+)'\s*,", r"conn.execute(_query('\1'),"),
    # Pattern 5: cur.execute("SELECT...") without params
    (r'cur\.execute\(\s*"([^"]+)"\s*\)', r'cur.execute(_query("\1"))'),
    # Pattern 6: cur.execute('SELECT...') without params
    (r"cur\.execute\(\s*'([^']+)'\s*\)", r"cur.execute(_query('\1'))"),
    # Pattern 7: conn.execute("SELECT...") without params
    (r'conn\.execute\(\s*"([^"]+)"\s*\)', r'conn.execute(_query("\1"))'),
    # Pattern 8: conn.execute('SELECT...') without params
    (r"conn\.execute\(\s*'([^']+)'\s*\)", r"conn.execute(_query('\1'))"),
]

# Track changes
changes_made = 0
original_content = content

for pattern, replacement in patterns:
    new_content, count = re.subn(pattern, replacement, content)
    if count > 0:
        print(f"Pattern {pattern[:30]}... matched {count} times")
        changes_made += count
        content = new_content

if changes_made > 0:
    # Write back
    persistence_file.write_text(content, encoding='utf-8')
    print(f"\n✅ Updated {changes_made} SQL execute statements")
    print(f"✅ File saved: {persistence_file}")
else:
    print("ℹ️  No changes needed")
