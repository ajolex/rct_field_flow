# Integration Proposal: RCT Design Activity into RCT Field Flow

## Executive Summary

The `rct-design` folder contains a **complete, standalone Streamlit app** designed to guide teams through a 6-step RCT design sprint (20-30 minutes) before conducting randomization. It's a perfect **pre-randomization component** that helps teams make critical design decisions before using the RCT Field Flow randomization tool.

**Proposal:** Integrate RCT Design as the **second page/module** in the RCT Field Flow app (after Home), positioned **before Randomization** in the navigation menu.

---

## Current State

### RCT Design Activity (Standalone)

- **Purpose:** Interactive learning activity for RCT design
- **Workflow:** 6-step design sprint (Challenge → Theory of Change → Measurement → Randomization → Implementation → Decision Trigger)
- **Features:**
  - Realistic program scenarios (Education, Health, Agriculture)
  - Sample baseline data for randomization practice
  - Team progress tracking
  - Professional HTML report generation
  - Facilitator dashboard (password-protected)
  - Embedded link to RCT Field Flow randomization tool

### RCT Field Flow (Current)

- **Purpose:** Execute RCT operations (randomization, case assignment, monitoring, analysis, etc.)
- **Current Navigation:** Home → Randomization → Case Assignment → Quality Checks → Analysis → Backcheck → Reports → Monitoring

---

## Integration Approach - SELECTED: Option 1 Full Integration ⭐

**Goal:** Seamlessly integrate RCT Design as the second navigation item in RCT Field Flow. Users complete the RCT Design sprint, then navigate directly to Randomization to execute their design. Both pages share the same app instance with unified navigation and session state.

**Key Principle:** Pages can operate standalone (RCT Design can run independently) OR fully integrated (smooth flow from Design → Randomization).

---

### **Architecture Overview**

```
RCT Field Flow Main App (app.py)
    ├── Navigation Menu
    │   ├── 🏠 Home
    │   ├── 🎯 RCT Design         ← NEW (integrated)
    │   ├── 🎲 Randomization      ← Modified (receives design data)
    │   ├── 📋 Case Assignment
    │   ├── ✅ Quality Checks
    │   ├── 📊 Analysis
    │   ├── 🔍 Backcheck
    │   ├── 📄 Reports
    │   └── 📈 Monitoring
    │
    └── Shared Session State
        ├── design_data (from RCT Design)
        ├── baseline_data (from Randomization)
        ├── randomization_result (shared)
        └── team_info (team name, progress)
```

---

### **Implementation Strategy**

#### **Step 1: Update `app.py` Navigation**

```python
nav = {
    "home": "🏠 Home",
    "design": "🎯 RCT Design",         # NEW: Second item
    "random": "🎲 Randomization",      # Modified to accept design data
    "cases": "📋 Case Assignment",
    "quality": "✅ Quality Checks",
    "analysis": "📊 Analysis",
    "backcheck": "🔍 Backcheck",
    "reports": "📄 Reports",
    "monitor": "📈 Monitoring Dashboard",
}
```

#### **Step 2: Integrate RCT Design Module**

Instead of `render_rct_design()` calling an external app, **import and adapt** the RCT Design code:

```python
def render_rct_design():
    """Render the RCT Design module (integrated)"""
    # Import the main logic from rct-design/app/main.py
    # Adapt it to use the shared session state
    
    # Key modifications:
    # 1. Remove st.set_page_config() (already set in app.py)
    # 2. Use shared session state keys
    # 3. Add "Next: Randomization" button
    # 4. Can still run standalone if needed
    
    from rct_field_flow.rct_design.app.main import (
        render_header,
        render_design_workbook,
        render_program_card_display,
    )
    
    # Render RCT Design interface
    render_header()
    
    # ... design logic ...
    
    # After design complete, offer next step
    if design_complete:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📋 Generate Report"):
                # Generate report
                pass
        with col2:
            if st.button("🎲 Continue to Randomization →", type="primary"):
                # Pass design_data to randomization
                st.session_state.design_data = design_data
                st.session_state.page = "random"
                st.rerun()
```

#### **Step 3: Modify Randomization to Accept Design Data**

```python
def render_randomization():
    """Render randomization module (enhanced with design data)"""
    
    # Check if coming from RCT Design
    if "design_data" in st.session_state and st.session_state.design_data:
        st.info("✓ Design loaded from RCT Design sprint")
        st.markdown(f"**Team:** {st.session_state.design_data.get('team_name')}")
        st.markdown(f"**Program:** {st.session_state.design_data.get('program_card')}")
        
        # Pre-populate randomization settings from design decisions
        randomization_unit = st.session_state.design_data.get('randomization_unit')
        # ... continue with existing randomization logic
    else:
        st.info("Upload baseline data to proceed with randomization")
    
    # ... existing randomization code ...
```

#### **Step 4: Update Session State Initialization**

```python
# In app.py initialization (around line 65):
if "design_data" not in st.session_state:
    st.session_state.design_data = None  # NEW: RCT Design decisions
if "team_name" not in st.session_state:
    st.session_state.team_name = None   # NEW: Team tracking
```

---

### **Workflow After Integration**

```
User opens RCT Field Flow
        ↓
🏠 Home page (default)
        ↓
User clicks: 🎯 RCT Design (from sidebar menu)
        ↓
RCT Design Interface (same Streamlit app)
   - Team name input
   - Program card selection
   - 6-step design sprint
        ↓
Design Complete ✓
        ↓
[Button] 🎲 Continue to Randomization →
        ↓
        (st.session_state.design_data stored)
        (Navigation changed to "random")
        (st.rerun() refreshes page)
        ↓
🎲 Randomization page (within same app)
   - Design data pre-loaded
   - Upload baseline data
   - Configure randomization per design
        ↓
Randomization Complete ✓
        ↓
User can navigate to next steps:
   - 📋 Case Assignment
   - ✅ Quality Checks
   - 📊 Analysis
   - etc.
```

---

### **File Structure Changes**

```
rct_field_flow/
├── app.py                          # Main app (UPDATED)
│   ├── Navigation: added "design" option
│   ├── render_rct_design() function (NEW)
│   ├── render_randomization() modified (accepts design_data)
│   └── Session state: added design_data, team_name
│
├── rct-design/                     # Submodule (MINIMALLY MODIFIED)
│   ├── app/
│   │   ├── main.py                (extract reusable functions)
│   │   ├── config.py              (shared)
│   │   ├── pages/
│   │   │   ├── randomization.py   (REMOVED - use main app randomization)
│   │   │   ├── report_generation.py
│   │   │   └── facilitator_dashboard.py
│   │   └── components/
│   ├── data/
│   └── docs/
│
├── randomize.py                    (existing)
├── monitor.py                      (existing)
├── analyze.py                      (existing)
└── ... other modules
```

---

### **Advantages of This Approach**

- ✅ **Seamless user experience:** One-click flow from Design → Randomization
- ✅ **Unified app:** Single navigation, shared session state
- ✅ **Standalone capability:** RCT Design can still be run independently if needed
- ✅ **Data continuity:** Design decisions automatically available in Randomization
- ✅ **Single deployment:** One Streamlit Cloud instance
- ✅ **Shared infrastructure:** Same watermark, same configuration
- ✅ **Future-proof:** Easy to add more modules later

---

### **Implementation Considerations**

1. **Session State Keys (Unified):**
   ```python
   # RCT Design outputs
   st.session_state.team_name              # Team identifier
   st.session_state.program_card_selected  # Education/Health/Agriculture
   st.session_state.design_data            # All 6-step decisions
   st.session_state.workbook_responses     # Detailed responses
   
   # RCT Field Flow (existing)
   st.session_state.baseline_data          # Upload data for randomization
   st.session_state.randomization_result   # Randomization output
   ```

2. **Navigation State:**
   ```python
   # Replace sidebar radio with conditional page selection
   current_page = st.session_state.get("current_page", "home")
   
   # Allow both:
   # - Navigation menu selection
   # - Programmatic navigation (e.g., after design complete)
   ```

3. **Standalone Flexibility:**
   - RCT Design can detect if running within RCT Field Flow or standalone
   - If standalone: Use rct-design's own session state
   - If integrated: Use shared session state from app.py

---

### **Dependencies Merge**

Add to main `requirements.txt`:
```
pydantic>=2.0.0
python-docx>=0.8.11
openpyxl>=3.1.0
streamlit-authenticator>=0.2.3  # Optional: for facilitator dashboard
```

Most other dependencies already present (pandas, streamlit, jinja2, weasyprint, plotly, etc.)

---

## Recommended Implementation

### **Phase 1: Core Integration (Week 1)**

- [ ] Extract reusable functions from rct-design/app/main.py
- [ ] Add `render_rct_design()` to app.py
- [ ] Update navigation menu in app.py
- [ ] Initialize design_data in session state
- [ ] Add "Continue to Randomization" button
- [ ] Modify `render_randomization()` to accept design_data
- [ ] Test end-to-end: Design → Randomization

### **Phase 2: Refinement (Week 2)**

- [ ] User testing and feedback
- [ ] Optimize session state passing
- [ ] Add report generation integration
- [ ] Test facilitator dashboard (if needed)
- [ ] Performance optimization

### **Phase 3: Polish (Week 3)**

- [ ] Update documentation
- [ ] Deploy to Streamlit Cloud
- [ ] Monitor for issues
- [ ] Gather user feedback

---

## User Experience & Data Flow

### **Complete Integrated Workflow:**

```
User opens RCT Field Flow
        ↓
🏠 Home page (landing page)
        ↓
User selects: 🎯 RCT Design (from sidebar)
        ↓
RCT Design Interface (integrated, same app)
   • Team name: "RCT Team A"
   • Program: "Education - Bridge to Basics"
   • 6-step sprint completed
   • Design decisions stored in session state
        ↓
Design Complete ✓
        ↓
[Button] 🎲 Continue to Randomization →
        ↓
🎲 Randomization page (same Streamlit instance)
   • Design data pre-populated (team, program, decisions)
   • Upload baseline data (CSV)
   • Randomization settings from design pre-filled
   • Run randomization
   • Export results
        ↓
Randomization Complete ✓
        ↓
User continues to: 📋 Case Assignment → ✅ Quality Checks → 📊 Analysis
```

---

## File Organization After Integration

```
rct_field_flow/
├── app.py                          # Main app (UPDATED)
│   ├── Added: "design" to nav menu
│   ├── Added: render_rct_design()
│   ├── Modified: render_randomization() for design_data
│   ├── Added: session state for design_data, team_name
│   └── Seamless page navigation
│
├── rct-design/                     # Integrated submodule
│   ├── app/
│   │   ├── main.py               (extract reusable functions)
│   │   ├── config.py
│   │   ├── pages/
│   │   │   ├── randomization.py  (REMOVED - not needed)
│   │   │   ├── report_generation.py
│   │   │   └── facilitator_dashboard.py
│   │   ├── utils/
│   │   ├── components/
│   │   └── assets/
│   ├── data/
│   ├── docs/
│   └── requirements.txt
│
├── randomize.py                    (existing)
├── analyze.py                      (existing)
└── ... other modules
```

---

## Advantages of Full Integration

✅ **Seamless UX:** One click from Design to Randomization  
✅ **Unified Navigation:** All tools in one menu  
✅ **Shared Session State:** Design decisions automatically available in Randomization  
✅ **Single Deployment:** One Streamlit Cloud instance  
✅ **Standalone Option:** Can still run rct-design independently if needed  
✅ **Natural Workflow:** Design → Execute → Analyze  
✅ **No Page Embedding:** Clean navigation, not iframes or external embeds  

---

## Implementation Checklist

### **Phase 1: Core Integration (Week 1)**

- [ ] Extract reusable functions from rct-design/app/main.py
- [ ] Add `render_rct_design()` to app.py
- [ ] Update navigation menu in app.py
- [ ] Initialize design_data in session state
- [ ] Add "Continue to Randomization" button
- [ ] Modify `render_randomization()` to accept design_data
- [ ] Test end-to-end: Design → Randomization

### **Phase 2: Refinement (Week 2)**

- [ ] User testing and feedback
- [ ] Optimize session state passing
- [ ] Add report generation integration
- [ ] Test facilitator dashboard (if needed)
- [ ] Performance optimization

### **Phase 3: Polish (Week 3)**

- [ ] Update documentation
- [ ] Deploy to Streamlit Cloud
- [ ] Monitor for issues
- [ ] Gather user feedback

---

## Key Technical Notes

### 1. **Removing Embedded RCT Field Flow**

The current rct-design app embeds the RCT Field Flow randomization tool via iframe. With full integration:

```python
# OLD (rct-design/app/pages/randomization.py):
st.components.v1.iframe(
    src="https://aj-rctfieldflow.streamlit.app/",
    height=1000
)

# NEW (unified approach):
# Simply navigate to randomization page within same app
if st.button("🎲 Continue to Randomization →"):
    st.session_state.current_page = "random"
    st.rerun()
```

### 2. **Session State Continuity**

```python
# When user clicks "Continue to Randomization":
st.session_state.design_data = {
    "team_name": "RCT Team A",
    "program_card": "education_bridge_to_basics",
    "randomization_unit": "schools",
    "assignment_method": "simple",
    "sample_size": 2400,
    # ... all other design decisions
}
st.session_state.current_page = "random"
st.rerun()  # Reload with new page
```

### 3. **Standalone Flexibility**

RCT Design can still detect if it's running standalone:

```python
def render_rct_design():
    # Check if part of integrated app
    if "IS_INTEGRATED_APP" in st.session_state:
        # Use shared session state
        # Show "Continue to Randomization" button
    else:
        # Running standalone (rct-design/app/main.py)
        # Use local session state
        # Show embedded iframe to RCT Field Flow
```

---

## Summary & Approval

### **What's Being Proposed:**

✅ **Option 1: Full Integration** (your choice)

- RCT Design becomes page 2 in RCT Field Flow (after Home)
- Seamless navigation from Design → Randomization via button click
- No embedded iframes (clean navigation instead)
- Design decisions automatically available in Randomization
- Pages can still operate standalone if needed
- Single Streamlit Cloud deployment

### **Key Benefits:**

✅ Users complete design sprint, then randomize in same app  
✅ Clean page navigation (no iframes, no context switching)  
✅ Design data flows to Randomization automatically  
✅ One-click experience: Design → Randomization → Execute  
✅ Standalone flexibility maintained  
✅ Single deployment, unified navigation  

### **Timeline:**

- **Week 1:** Extract and integrate core functions, update app.py navigation
- **Week 2:** Testing, feedback, refinement
- **Week 3:** Polish and deploy to Streamlit Cloud

### **Next Action:**

Ready to implement! Proceed with Phase 1 or make modifications as needed.

---

## Resources

**RCT Design Activity:**
- `rct_field_flow/rct-design/README.md` – Overview
- `rct_field_flow/rct-design/docs/INTEGRATION_SUMMARY.md` – Integration details

**RCT Field Flow:**
- `rct_field_flow/README.md` – Main app overview

---

**Status:** ✅ Approved for Option 1 - Full Integration  
**Last Updated:** November 14, 2025  
**Integration Approach:** Seamless page navigation without embedded iframes
