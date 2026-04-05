#!/bin/bash

# TechSLM Improved API - Curl Testing Script
# Run this after starting the API server with: python task4_fastapi_deployment_improved.py

echo ""
echo "========================================================================"
echo "IMPROVED TechSLM API - CURL TESTING"
echo "========================================================================"
echo ""

BASE_URL="http://localhost:8000"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if server is running
echo "Checking server status..."
if ! curl -s "$BASE_URL/health" > /dev/null 2>&1; then
    echo "❌ Server not running! Start it with:"
    echo "   python task4_fastapi_deployment_improved.py"
    exit 1
fi
echo "✓ Server is running"
echo ""

# Function to make a request and display results
test_generation() {
    local test_num=$1
    local prompt=$2
    local max_tokens=$3
    local temperature=$4
    
    echo -e "${YELLOW}Test $test_num:${NC} $prompt"
    echo "  Max tokens: $max_tokens | Temperature: $temperature"
    
    response=$(curl -s -X POST "$BASE_URL/generate" \
        -H "Content-Type: application/json" \
        -d "{
            \"prompt\": \"$prompt\",
            \"max_tokens\": $max_tokens,
            \"temperature\": $temperature,
            \"top_k\": 50
        }")
    
    # Extract fields from response
    generated=$(echo "$response" | grep -o '"generated_text":"[^"]*' | cut -d'"' -f4)
    num_tokens=$(echo "$response" | grep -o '"num_tokens":[0-9]*' | cut -d':' -f2)
    
    if [ -z "$num_tokens" ]; then
        echo "  ❌ Error: $(echo "$response" | grep -o '"detail":"[^"]*' | cut -d'"' -f4)"
    else
        word_count=$(echo "$generated" | wc -w)
        echo "  ✓ Generated $num_tokens tokens ($word_count words)"
        echo "  Output: ${generated:0:120}..."
    fi
    echo ""
}

# Test cases
test_generation 1 "Artificial intelligence" 100 0.7
test_generation 2 "Machine learning models" 150 0.7
test_generation 3 "Deep learning requires" 120 0.6
test_generation 4 "The transformer architecture" 200 0.8
test_generation 5 "Neural networks consist of" 100 0.5
test_generation 6 "Data preprocessing is" 80 0.3
test_generation 7 "Computer vision applications" 100 1.2
test_generation 8 "Training neural networks" 150 0.7
test_generation 9 "Transfer learning enables" 120 0.8
test_generation 10 "Natural language processing" 100 0.7

echo "========================================================================"
echo "Testing Model Info Endpoint"
echo "========================================================================"
echo ""

curl -s -X GET "$BASE_URL/info" | python3 -m json.tool

echo ""
echo "========================================================================"
echo "✓ All curl tests completed!"
echo "========================================================================"
echo ""
