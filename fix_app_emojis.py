#!/usr/bin/env python3
"""Fix emoji glitches in app.py"""

# Read as binary to avoid encoding issues
with open('rct_field_flow/app.py', 'rb') as f:
    content = f.read()

# Replace malformed UTF-8 sequences with correct ones
replacements = [
    # microscope 🔬
    (b'\xc3\xb0\xc5\xb8\xc2\x94\xc2\xac', b'\xf0\x9f\x94\xac'),
    # dice 🎲
    (b'\xc3\xb0\xc5\xb8\xc28\xc2\xb2', b'\xf0\x9f\x8e\xb2'),
    # clipboard 📋
    (b'\xc3\xb0\xc5\xb8\xc2\x94\xc2\x8b', b'\xf0\x9f\x93\x8b'),
    # checkmark ✅
    (b'\xc3\xa2\xc5\x93\xc2\x85', b'\xe2\x9c\x85'),
    # chart 📊
    (b'\xc3\xb0\xc5\xb8\xc2\x94\xc2\x8a', b'\xf0\x9f\x93\x8a'),
    # folder 📁
    (b'\xc3\xb0\xc5\xb8\xc2\x94\xc2\x81', b'\xf0\x9f\x93\x81'),
    # check ✓
    (b'\xc3\xa2\xc5\x93\xc2\x94', b'\xe2\x9c\x94'),
    # multiply ×
    (b'\xc3\x83\xc2\x97', b'\xc3\x97'),
    # gear ⚙️
    (b'\xc3\xa2\xc5\xa1\xc2\x99\xc3\xaf\xc2\xb8\xc2\x8f', b'\xe2\x9a\x99\xef\xb8\x8f'),
    # rocket 🚀
    (b'\xc3\xb0\xc5\xb8\xc2\x9a\xc2\x80', b'\xf0\x9f\x9a\x80'),
    # keycap 1️⃣
    (b'1\xc3\xaf\xc2\xb8\xc2\x8f\xc3\xa2\xc6\x92\xc2\xa3', b'1\xef\xb8\x8f\xe2\x83\xa3'),
    # keycap 2️⃣
    (b'2\xc3\xaf\xc2\xb8\xc2\x8f\xc3\xa2\xc6\x92\xc2\xa3', b'2\xef\xb8\x8f\xe2\x83\xa3'),
    # keycap 3️⃣
    (b'3\xc3\xaf\xc2\xb8\xc2\x8f\xc3\xa2\xc6\x92\xc2\xa3', b'3\xef\xb8\x8f\xe2\x83\xa3'),
    # keycap 4️⃣
    (b'4\xc3\xaf\xc2\xb8\xc2\x8f\xc3\xa2\xc6\x92\xc2\xa3', b'4\xef\xb8\x8f\xe2\x83\xa3'),
    # pointing up 👆
    (b'\xc3\xb0\xc5\xb8\xc2\x91\xc2\x86', b'\xf0\x9f\x91\x86'),
    # warning ⚠️
    (b'\xc3\xa2\xc5\xa1\xc2\xa0\xc3\xaf\xc2\xb8\xc2\x8f', b'\xe2\x9a\xa0\xef\xb8\x8f'),
    # floppy disk 💾
    (b'\xc3\xb0\xc5\xb8\xc2\x91\xc2\xbe', b'\xf0\x9f\x92\xbe'),
    # cross mark ❌
    (b'\xc3\xa2\xc5\x93\xc2\x8c', b'\xe2\x9c\x8c'),
    # magnifying glass 🔍
    (b'\xc3\xb0\xc5\xb8\xc2\x94\xc2\x8d', b'\xf0\x9f\x94\x8d'),
    # stopwatch ⏱️
    (b'\xc3\xa2\xc2\xb1\xc2\xb1\xc3\xaf\xc2\xb8\xc2\x8f', b'\xe2\x8f\xb1\xef\xb8\x8f'),
    # arrows 🔄
    (b'\xc3\xb0\xc5\xb8\xc2\x94\xc2\x84', b'\xf0\x9f\x94\x84'),
    # people 👥
    (b'\xc3\xb0\xc5\xb8\xc2\x91\xc2\xa5', b'\xf0\x9f\x91\xa5'),
    # inbox tray 📥
    (b'\xc3\xb0\xc5\xb8\xc2\x94\xc2\xa5', b'\xf0\x9f\x93\xa5'),
    # chart increasing 📈
    (b'\xc3\xb0\xc5\xb8\xc2\x94\xcb\x86', b'\xf0\x9f\x93\x88'),
    # target 🎯
    (b'\xc3\xb0\xc5\xb8\xc28\xc2\xaf', b'\xf0\x9f\x8e\xaf'),
    # calendar 📅
    (b'\xc3\xb0\xc5\xb8\xc2\x94\xc2\x85', b'\xf0\x9f\x93\x85'),
    # books 📚
    (b'\xc3\xb0\xc5\xb8\xc2\x94\xc2\x9a', b'\xf0\x9f\x93\x9a'),
]

for old, new in replacements:
    content = content.replace(old, new)

# Also fix the duplicate footer/else block
content_str = content.decode('utf-8')

# Remove the duplicate else block and footer section
lines = content_str.split('\n')
fixed_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    # Skip duplicate "else: st.info..." at end after footer
    if i > 1130 and 'else:' in line and i < len(lines) - 5:
        # Check if next line has st.info about loading data
        if i + 1 < len(lines) and 'Please load data to view dashboard' in lines[i + 1]:
            # Skip this else block (duplicate)
            i += 2  # Skip else and st.info
            continue
    # Skip extra indented footer line
    if 'st.sidebar.markdown("[GitHub Repository]' in line and line.startswith('        '):
        # Fix indentation
        line = line[8:]  # Remove extra indent
    fixed_lines.append(line)
    i += 1

content_str = '\n'.join(fixed_lines)

# Write back
with open('rct_field_flow/app.py', 'w', encoding='utf-8') as f:
    f.write(content_str)

print("✓ Fixed all emoji glitches in app.py")
print("✓ Fixed duplicate footer section")
print("✓ Fixed indentation errors")
