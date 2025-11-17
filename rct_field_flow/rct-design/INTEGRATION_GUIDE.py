"""
Integration Guide for RCT Design Wizard

This file shows how to integrate the wizard into the main app.py navigation.
"""

# Option 1: Add as a page in existing Streamlit app
# ================================================

# In your main app.py, add this import at the top:

import sys
from pathlib import Path

# Add rct-design directory to Python path
rct_design_path = Path(__file__).parent / "rct-design"
if str(rct_design_path) not in sys.path:
    sys.path.insert(0, str(rct_design_path))

# Import the wizard main function
try:
    from wizard import main as wizard_main
except ImportError:
    print("Warning: Could not import RCT Design Wizard")
    wizard_main = None


# Then in your page navigation/menu:

pages = {
    "Home": home_page,
    "Randomization": randomization_page,
    "Power Analysis": power_page,
    "RCT Design Wizard": wizard_main,  # Add this line
    # ... other pages
}

# Option 2: Standalone Streamlit app
# ==================================

# Run directly from command line:
# streamlit run rct_field_flow/rct-design/wizard.py

# Option 3: Add to existing multipage app structure
# =================================================

# If using Streamlit's built-in multipage apps:
# 1. Create a file: pages/6_RCT_Design_Wizard.py
# 2. Add this content:

"""
# In pages/6_RCT_Design_Wizard.py
import sys
from pathlib import Path

# Add rct-design to path
rct_design_path = Path(__file__).parent.parent / "rct-design"
sys.path.insert(0, str(rct_design_path))

from wizard import main

if __name__ == "__main__":
    main()
"""

# Integration Checklist
# =====================

print("""
RCT Design Wizard Integration Checklist:

□ 1. Ensure jinja2 is installed: pip install jinja2
□ 2. Verify rct-design folder exists in rct_field_flow/
□ 3. Check that schema.json exists
□ 4. Create data/ directory (auto-created on first run)
□ 5. Add wizard to app navigation
□ 6. Test integration with existing Randomization page
□ 7. Test integration with existing Power page
□ 8. Test save/load functionality
□ 9. Test export to Markdown
□ 10. Verify all 15 sections render correctly

Integration Contracts Required:
- Randomization page should set st.session_state["rand"]
- Power page should set st.session_state["power"]
- See README.md for detailed contract specifications

For standalone testing:
    cd rct_field_flow
    streamlit run rct-design/wizard.py
""")
