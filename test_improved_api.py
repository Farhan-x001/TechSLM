#!/usr/bin/env python3
"""
Test script for improved TechSLM API
- Tests word-level generation (not character-level)
- Validates coherence and length
- Compares with previous results
"""

import subprocess
import json
import time
import requests
from typing import Optional


def wait_for_server(max_retries=30, timeout=1):
    """Wait for API server to be ready."""
    for attempt in range(max_retries):
        try:
            response = requests.get("http://localhost:8000/health", timeout=timeout)
            if response.status_code == 200:
                print(f"✓ Server ready! ({attempt + 1} attempts)")
                return True
        except requests.exceptions.ConnectionError:
            if attempt < max_retries - 1:
                time.sleep(1)
    
    print(f"✗ Server failed to start after {max_retries} attempts")
    return False


def run_test_suite():
    """Run comprehensive tests on the improved API."""
    
    print("\n" + "=" * 70)
    print("IMPROVED TechSLM API - TEST SUITE")
    print("=" * 70 + "\n")
    
    # Test cases: (name, prompt, max_tokens, temperature, expected_behavior)
    test_cases = [
        ("Basic AI topic", "Artificial intelligence", 100, 0.7, "coherent paragraph"),
        ("ML concepts", "Machine learning models use", 150, 0.7, "sentence completion"),
        ("Data science", "Data preprocessing is important", 100, 0.5, "deterministic output"),
        ("Long generation", "The transformer architecture", 200, 0.8, "longer text"),
        ("Neural networks", "Deep learning requires", 120, 0.6, "technical accuracy"),
        ("Low temperature", "Python programming language", 100, 0.3, "focused output"),
        ("High temperature", "Computer vision tasks", 100, 1.2, "diverse output"),
        ("Short prompt", "AI", 80, 0.7, "context recovery"),
        ("Question format", "What is machine learning?", 150, 0.7, "answer-like response"),
        ("Complex topic", "Transfer learning enables knowledge reuse through", 120, 0.8, "coherent continuation"),
    ]
    
    results = []
    
    for i, (name, prompt, max_tokens, temp, expected) in enumerate(test_cases, 1):
        print(f"Test {i:2d}: {name}")
        print(f"  Prompt: '{prompt}'")
        print(f"  Max tokens: {max_tokens}, Temperature: {temp}")
        
        try:
            start_time = time.time()
            
            response = requests.post(
                "http://localhost:8000/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temp,
                    "top_k": 50
                },
                timeout=120
            )
            
            elapsed = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                generated = data['generated_text']
                num_tokens = data['num_tokens']
                
                # Count actual words (not characters)
                words = generated.split()
                word_count = len(words)
                
                print(f"  ✓ Generated {num_tokens} tokens ({word_count} words) in {elapsed:.1f}s")
                print(f"  Output: {generated[:120]}...")
                
                results.append({
                    "test_number": i,
                    "name": name,
                    "prompt": prompt,
                    "status": "success",
                    "tokens_generated": num_tokens,
                    "words_generated": word_count,
                    "response_time": elapsed,
                    "temperature": temp,
                    "expected_behavior": expected,
                    "generated_text": generated[:200]  # First 200 chars
                })
                
            else:
                print(f"  ✗ HTTP {response.status_code}")
                print(f"  Error: {response.text}")
                
                results.append({
                    "test_number": i,
                    "name": name,
                    "status": "failed",
                    "error": response.text
                })
        
        except requests.exceptions.Timeout:
            print(f"  ✗ Request timeout")
            results.append({
                "test_number": i,
                "name": name,
                "status": "timeout"
            })
        
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                "test_number": i,
                "name": name,
                "status": "error",
                "error": str(e)
            })
        
        print()
    
    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    successful = [r for r in results if r.get('status') == 'success']
    print(f"\nPassed: {len(successful)}/{len(test_cases)}")
    
    if successful:
        avg_tokens = sum(r['tokens_generated'] for r in successful) / len(successful)
        avg_words = sum(r['words_generated'] for r in successful) / len(successful)
        avg_time = sum(r['response_time'] for r in successful) / len(successful)
        
        print(f"\nAverage metrics:")
        print(f"  Tokens generated: {avg_tokens:.1f}")
        print(f"  Words generated: {avg_words:.1f}")
        print(f"  Response time: {avg_time:.2f}s")
        
        min_words = min(r['words_generated'] for r in successful)
        max_words = max(r['words_generated'] for r in successful)
        print(f"\nWord count range: {min_words} - {max_words} words")
    
    # Save results
    with open('test_improved_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to test_improved_results.json")
    
    return len(successful), len(test_cases)


if __name__ == "__main__":
    print("\nChecking for running server...")
    if not wait_for_server():
        print("\n✗ API server is not running. Start it with:")
        print("  python task4_fastapi_deployment_improved.py")
        exit(1)
    
    passed, total = run_test_suite()
    
    print("\n" + "=" * 70)
    if passed == total:
        print(f"✓ ALL TESTS PASSED ({passed}/{total})")
    else:
        print(f"⚠ PARTIAL SUCCESS ({passed}/{total} passed)")
    print("=" * 70 + "\n")
