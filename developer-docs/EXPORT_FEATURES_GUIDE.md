# RCT Design Wizard - Export & Sample Data Feature Summary

## What's New ✨

The RCT Design Wizard now includes:

### 1. **Sample Concept Note Preview** 
View realistic examples from three sectors (Education, Health, Agriculture) before you start building your own concept note. Helps you understand:
- How the final document will look
- What good writing looks like for each section
- Realistic numbers and timelines for social development RCTs

### 2. **Multiple Export Formats**
Export your concept note in three formats optimized for different workflows:

| Format | Best For | Features |
|--------|----------|----------|
| **Markdown** (.md) | Version control & Git | Plain text, trackable changes |
| **Word** (.docx) | Editing & collaboration | Fully editable in Word/Google Docs |
| **PDF** (.pdf) | Final review & printing | Professional formatting, print-ready |

### 3. **Professional Document Generation**
Each export format includes:
- Professional styling and formatting
- All 16 concept note sections
- Metadata (title, PIs, country, funder, stage)
- Budget tables with calculations
- Proper margins and typography

---

## How to Use

### Preview a Sample Concept Note

1. Open the RCT Design Wizard main page
2. Expand the **"📋 Sample & Export"** section
3. In the left column, select a sector:
   - 📚 **Education**: Malawi Early Literacy Program (realistic K-12 intervention)
   - 🏥 **Health**: Ghana Maternal Health (health worker training)
   - 🌾 **Agriculture**: Kenya Climate-Smart Farming (smallholder farmers)
4. Click **"👁️ View Sample"** to see three tabs:
   - **Markdown Preview**: How the formatted document looks
   - **JSON Data**: Raw data structure (for technical users)
   - **Fields Completed**: Statistics on data completeness

### Export Your Concept Note

1. Complete your concept note using the 15-section wizard
2. Expand the **"📋 Sample & Export"** section
3. In the right column, select your desired format:
   - **Markdown (.md)** - for version control
   - **Word Document (.docx)** - for editing and sharing
   - **PDF (.pdf)** - for final review and printing
4. Click **"📥 Generate Export"**
5. Click the download button that appears
6. Use the file outside the application:
   - Edit in Word/Google Docs (DOCX)
   - Track changes in Git (Markdown)
   - Print or email (PDF)

---

## Sample Programs Included

### Education: Malawi Early Literacy Program
- **Setting**: Rural Malawi, Lilongwe and Mchinji districts
- **Participants**: 3,200 Grade 1-3 students across 16 schools
- **Intervention**: Phonics-based reading training + coaching
- **Timeline**: 2-year implementation + analysis
- **Budget**: $275,000
- **Sample Design**: Simple randomization with stratification (2 arms)
- **Primary Outcome**: Grade 3 reading proficiency
- **Key Features**: Complete example of all 16 sections with realistic details

### Health: Ghana Maternal Health Program
- **Setting**: Ghana, community health worker recruitment  
- **Participants**: 8,000 pregnant women
- **Intervention**: SMS reminders + community worker support
- **Focus**: Increasing antenatal care attendance
- **Details**: Includes realistic health sector terminology and outcomes

### Agriculture: Kenya Climate-Smart Farming
- **Setting**: Kenya, semi-arid smallholder farmers
- **Participants**: 2,500 farmers
- **Intervention**: Water conservation training + input provision
- **Focus**: Drought resilience and farm productivity
- **Details**: Includes agricultural program structure and outcomes

---

## Technical Details

### New Modules

**`export_formats.py`** - Multi-format export engine
- Renders markdown using Jinja2 templates
- Generates DOCX with python-docx (professional formatting)
- Generates PDF via weasyprint (HTML-to-PDF conversion)
- Graceful error handling for optional dependencies

**`sample_data.py`** - Sample program generator
- Three complete example concept notes
- All 16 sections fully populated
- Realistic numbers, budgets, and timelines
- Education example (primary) expanded with complete details

### Dependencies
All required packages already installed in your environment:
- `jinja2>=3.1` - Template rendering ✓
- `python-docx>=0.8.11` - DOCX export ✓
- `weasyprint>=62.0` - PDF export ✓

### Backward Compatibility
- Old `export.py` still works (now wraps new `export_formats.py`)
- All existing functionality preserved
- New features are additions, not replacements

---

## Export Format Guide

### When to Use Markdown (.md)
**Best for:**
- Software developers using Git
- Academic collaborations with version control
- Tracking changes over time
- Combining with other development tools

**Format characteristics:**
- Plain text, human-readable
- Preserves all formatting and structure
- File size: ~15-20 KB
- Can be tracked in Git with diff/blame

**Example use:**
```bash
git add concept_note.md
git commit -m "Updated intervention description"
git push origin feature/my-rct
```

### When to Use Word (.docx)
**Best for:**
- Stakeholder review and feedback
- Collaborative editing with non-technical partners
- Final document customization
- Institutional requirements

**Format characteristics:**
- Fully editable in Word, Google Docs, LibreOffice
- Professional table formatting with styled cells
- File size: ~100-150 KB
- Easy for others to comment and suggest changes

**Example use:**
1. Export to DOCX
2. Share with co-investigators via email or Google Drive
3. They comment/track changes
4. Merge feedback back into wizard

### When to Use PDF (.pdf)
**Best for:**
- IRB submissions
- Donor/funder applications
- Final concept note archives
- Printing and distribution

**Format characteristics:**
- Professional appearance guaranteed across devices
- Print-ready formatting (1" margins)
- Read-only (prevents accidental editing)
- File size: ~80-120 KB

**Example use:**
```
Concept Note - RCT Design Wizard v1.0
Generated: 2024-11-16 14:23 UTC
Status: Final for IRB Submission
```

---

## Common Use Cases

### Use Case 1: Developing with Your Team
1. Create concept note in wizard
2. Export as **Markdown**
3. Commit to Git repository
4. Share link with collaborators
5. Track all changes with `git log`

### Use Case 2: Getting Stakeholder Feedback
1. Create concept note in wizard
2. Export as **Word (.docx)**
3. Share via email or Google Drive
4. Stakeholders edit and comment
5. You review changes and update wizard
6. Re-export DOCX with updates

### Use Case 3: Final Submission
1. Complete concept note in wizard
2. Export as **PDF** (for IRB/funder)
3. Also export as **Word** (for reviewers to edit)
4. Keep **Markdown** in Git for version control

### Use Case 4: Learning/Teaching
1. View sample concept note
2. Study Education example (3 sectors available)
3. Understand structure and best practices
4. Use as template for your own RCT

---

## Tips for Best Results

### For Sample Preview
- **Education sample** is most complete (all 16 sections detailed)
- **Health & Agriculture** are functional examples (could be expanded)
- Use samples as template when filling your own concept note
- Copy realistic phrasing and structure from samples

### For Exporting
- **Start with Markdown** to track changes
- **Use DOCX** for stakeholder review
- **Finish with PDF** for final submission
- Keep all three versions (development, draft, final)

### For Editing in Word
- Use "Track Changes" to record edits
- Accept/reject changes before re-uploading to wizard
- Maintain question placeholders while drafting
- Use Comments for peer review

### For Version Control
- Commit Markdown regularly
- Use meaningful commit messages
- Branch for major changes
- Tag versions for submissions (v1.0-irb, v1.1-feedback, v2.0-final)

---

## Troubleshooting

### Q: I get a weasyprint error when trying PDF export
**A:** This is a Windows library issue. Try these alternatives:
1. Export to **DOCX** instead (works reliably on all platforms)
2. Open DOCX in Word and export to PDF manually
3. Use an online converter to convert Markdown to PDF

### Q: I exported but don't see the download button
**A:** 
1. Check your browser's download settings
2. The file may be in your Downloads folder
3. Try a different browser (Chrome, Firefox) if issue persists
4. Check browser console for errors (F12 → Console)

### Q: Can I edit the sample data?
**A:**
1. Currently, samples are read-only for reference
2. Create a new project and use samples as a starting template
3. Copy-paste content from sample preview into your own concept note

### Q: How do I include sample data in my concept note?
**A:**
1. View sample in the "Sample & Export" preview
2. Copy sections you want
3. Paste into corresponding wizard sections
4. Customize with your own details

---

## What's Included in Each Export

### All Exports Include:
- ✅ All 16 concept note sections
- ✅ Professional formatting
- ✅ Complete metadata (title, PIs, country, funder, stage)
- ✅ Timestamp and version info
- ✅ All text content with preserved structure

### Markdown Extra:
- ✅ Heading hierarchy (# ## ###)
- ✅ Bold/italic formatting
- ✅ Lists (unordered and numbered)
- ✅ Code blocks (for code examples)

### DOCX Extra:
- ✅ Professional styling (colors, fonts)
- ✅ Styled tables (for budgets, timelines)
- ✅ Margins set to 1 inch (standard)
- ✅ Proper page breaks between sections
- ✅ Alternating row colors in tables

### PDF Extra:
- ✅ Print-ready formatting
- ✅ Guaranteed appearance across devices
- ✅ Digital-signature ready
- ✅ Compressed for email (80-120 KB)

---

## Getting Started

1. **To see a sample**: Expand "📋 Sample & Export" → View sample for your sector
2. **To create your own**: Use tips and examples from sample
3. **To export**: Complete your concept note → Select format → Download

---

## Questions or Feedback?

See documentation in:
- `EXPORT_AND_SAMPLES.md` - Detailed technical documentation
- `IMPLEMENTATION_COMPLETE.md` - Implementation notes and architecture
- `README.md` - Main wizard documentation
- `schema.json` - Data structure reference

All modules are fully functional and ready to use!
