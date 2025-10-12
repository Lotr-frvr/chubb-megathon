#!/bin/bash

echo "=========================================="
echo "  Starting Churn Prediction Server"
echo "=========================================="
echo ""

# Use the virtual environment Python
PYTHON_CMD="/Users/akshat/Downloads/html/venv/bin/python"

# Check if Python is available
if ! command -v "$PYTHON_CMD" &> /dev/null; then
    echo "❌ Error: Virtual environment Python not found"
    echo "   Expected at: $PYTHON_CMD"
    exit 1
fi

echo "✅ Python found at: $PYTHON_CMD"
echo ""
echo "🚀 Starting backend server..."
echo "   Server will be available at: http://localhost:5000"
echo "   Press Ctrl+C to stop"
echo ""
echo "=========================================="
echo ""

# Start the backend server
"$PYTHON_CMD" backend_prediction.py
