#!/bin/bash
# Test script for CLI interface

echo " Testing annot-utils CLI"
echo "=========================="

# Test 1: Help command
echo -e "\n1. Testing help command..."
annot-utils --help

# Test 2: Convert help
echo -e "\n2. Testing convert help..."
annot-utils convert --help

# Test 3: Validate help
echo -e "\n3. Testing validate help..."
annot-utils validate --help

# Test 4: Agreement help
echo -e "\n4. Testing agreement help..."
annot-utils agreement --help

# Test 5: Version
echo -e "\n5. Testing version..."
annot-utils --version

echo -e "\n✅ CLI tests completed!"
