#!/usr/bin/env python3
"""Remove extra blank lines from app.py"""

with open('rct_field_flow/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Remove sequences of 3+ blank lines, replace with single blank line
import re
content = re.sub(r'\n\n\n+', '\n\n', content)

with open('rct_field_flow/app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ Removed extra blank lines from app.py")
