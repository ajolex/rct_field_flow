"""Fix malformed emoji glitches in app.py"""

# Read the file
with open('rct_field_flow/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Define emoji replacements (malformed -> correct)
replacements = {
    '\uf04c': '🔬',  # microscope
    '\uf072': '🎲',  # dice
    '\uf04b': '📋',  # clipboard
    '\u2713': '✅',  # checkmark
    '\uf04a': '📊',  # chart
    '\uf0"': '📁',  # folder
    '\u0094': '✓',   # check
    '\u00d7': '×',   # multiply
    '\u2699\ufe0f': '⚙️',  # gear
    '\uf05a': '🚀',  # rocket
    '1\ufe0f\u20e3': '1️⃣',  # keycap 1
    '2\ufe0f\u20e3': '2️⃣',  # keycap 2
    '3\ufe0f\u20e3': '3️⃣',  # keycap 3
    '4\ufe0f\u20e3': '4️⃣',  # keycap 4
    '\uf076': '👆',  # pointing up
    '\u26a0\ufe0f': '⚠️',  # warning
    '\uf076': '💾',  # floppy disk
    '\u274c': '❌',  # cross mark
    '\uf05': '🔍',  # magnifying glass
    '\u23f1\ufe0f': '⏱️',  # stopwatch
    '\uf04': '🔄',  # arrows
    '\uf076': '👥',  # people
    '\uf065': '📥',  # inbox tray
    '\uf048': '📈',  # chart increasing
    '\uf06f': '🎯',  # target
    '\uf046': '📅',  # calendar
    '\uf05a': '📚',  # books
}

# Apply specific pattern replacements for malformed UTF-8
import re

# Replace specific malformed patterns we see in the file
patterns = [
    (r'ðŸ"¬', '🔬'),
    (r'ðŸŽ²', '🎲'),
    (r'ðŸ"‹', '📋'),
    (r'âœ…', '✅'),
    (r'ðŸ"Š', '📊'),
    (r'ðŸ"', '📁'),
    (r'âœ"', '✓'),
    (r'Ã—', '×'),
    (r'âš™ï¸', '⚙️'),
    (r'ðŸš€', '🚀'),
    (r'1ï¸âƒ£', '1️⃣'),
    (r'2ï¸âƒ£', '2️⃣'),
    (r'3ï¸âƒ£', '3️⃣'),
    (r'4ï¸âƒ£', '4️⃣'),
    (r'ðŸ'†', '👆'),
    (r'âš ï¸', '⚠️'),
    (r'ðŸ'¾', '💾'),
    (r'âŒ', '❌'),
    (r'ðŸ"', '🔍'),
    (r'â±ï¸', '⏱️'),
    (r'ðŸ"„', '🔄'),
    (r'ðŸ'¥', '👥'),
    (r'ðŸ"¥', '📥'),
    (r'ðŸ"ˆ', '📈'),
    (r'ðŸŽ¯', '🎯'),
    (r'ðŸ"…', '📅'),
    (r'ðŸ"š', '📚'),
]

for pattern, replacement in patterns:
    content = content.replace(pattern, replacement)

# Fix specific syntax errors we saw
# Remove duplicate else clause at the end
lines = content.split('\n')
fixed_lines = []
footer_started = False
skip_next_else = False

for i, line in enumerate(lines):
    # Fix the duplicate footer section
    if '# Footer' in line and not footer_started:
        footer_started = True
        fixed_lines.append(line)
    elif footer_started and line.strip().startswith('else:'):
        # Skip duplicate else block at end
        skip_next_else = True
        continue
    elif skip_next_else and line.strip().startswith('st.info'):
        continue
    elif skip_next_else and not line.strip():
        skip_next_else = False
        continue
    else:
        fixed_lines.append(line)

content = '\n'.join(fixed_lines)

# Write back
with open('rct_field_flow/app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('✓ Fixed emoji glitches in app.py')
print('✓ Fixed syntax errors')
