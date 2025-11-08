#!/bin/bash
# Launch RCT Field Flow Integrated UI
# This script starts the Streamlit application

echo "========================================"
echo "  RCT Field Flow - Integrated UI"
echo "========================================"
echo ""
echo "Starting application..."
echo ""
echo "The app will open in your default browser at:"
echo "http://localhost:8501"
echo ""
echo "To stop the server, press Ctrl+C"
echo ""
echo "========================================"
echo ""

python -m streamlit run rct_field_flow/app.py
