#!/usr/bin/env python3
"""Remove BOM from app.py"""

with open('rct_field_flow/app.py', 'rb') as f:
    content = f.read()

# Remove BOM if present
if content.startswith(b'\xef\xbb\xbf'):
    content = content[3:]
    print("Removed UTF-8 BOM")
else:
    print("No BOM found")

with open('rct_field_flow/app.py', 'wb') as f:
    f.write(content)

print("✓ File cleaned")
