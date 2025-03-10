#!/bin/bash

# Exit on error
set -e

echo "Rebuilding and linking langgraph package..."

# Navigate to langgraph package
cd ../langgraph

# Build the package
npm run build

# Run link:local
npm run link:local

echo "Rebuild complete! You can now start Jupyter Lab." 