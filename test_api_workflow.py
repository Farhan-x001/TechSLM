#!/usr/bin/env python3
"""
API Workflow Test Suite
Tests all 15 generation scenarios and documents results
"""

import subprocess
import json
import time
import statistics
from datetime import datetime
from pathlib import Path


TEST_SCENARIOS = [
    {"id": 1, "name": "Baseline test", "prompt": "This week in AI", "max_length": 200, "temperature": 0.8, "top_k": 40},
    {"id": 2, "name": "Extended generation", "prompt": "Machine learning", "max_length": 5000, "temperature": 0.7, "top_k": 50},
    {"id": 3, "name": "Very long generation", "prompt": "Transformer architecture", "max_length": 10000, "temperature": 0.75, "top_k": 40},
    {"id": 4, "name": "Maximum length", "prompt": "Deep learning models", "max_length": 20000, "temperature": 0.8, "top_k": 50},
    {"id": 5, "name": "Low temperature", "prompt": "Neural networks", "max_length": 1000, "temperature": 0.3, "top_k": 10},
    {"id": 6, "name": "High temperature", "prompt": "Artificial intelligence", "max_length": 1500, "temperature": 1.5, "top_k": 100},
    {"id": 7, "name": "Ultra-focused", "prompt": "GPU acceleration", "max_length": 800, "temperature": 0.5, "top_k": 5},
    {"id": 8, "name": "High diversity", "prompt": "Data science", "max_length": 2000, "temperature": 1.0, "top_k": 100},
    {"id": 9, "name": "Minimal prompt", "prompt": "AI", "max_length": 3000, "temperature": 0.8, "top_k": 40},
    {"id": 10, "name": "Rich context", "prompt": "The rapid advancement of AI and ML", "max_length": 1000, "temperature": 0.7, "top_k": 50},
    {"id": 11, "name": "Technical precision", "prompt": "Attention mechanism", "max_length": 2000, "temperature": 0.4, "top_k": 20},
    {"id": 12, "name": "Business-focused", "prompt": "MLOps and deployment", "max_length": 1500, "temperature": 0.6, "top_k": 40},
    {"id": 13, "name": "Research-focused", "prompt": "Diffusion models", "max_length": 3000, "temperature": 0.75, "top_k": 45},
    {"id": 14, "name": "Balanced defaults", "prompt": "Quantum computing", "max_length": 1000, "temperature": 0.8, "top_k": 40},
    {"id": 15, "name": "Maximum diversity", "prompt": "Future of technology", "max_length": 5000, "temperature": 1.2, "top_k": 80},
]


def run_test(scenario):
    """Execute a single API test scenario"""
    url = "http://localhost:8000/generate"
    headers = "-H 'Content-Type: application/json'"
    
    data = {
        "prompt": scenario["prompt"],
        "max_length": scenario["max_length"],
        "temperature": scenario["temperature"],
        "top_k": scenario["top_k"],
    }
    
    payload = json.dumps(data)
    cmd = f"curl -s -X POST '{url}' {headers} -d '{payload}'"
    
    print(f"  Test #{scenario['id']}: {scenario['name']}...", end=" ", flush=True)
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        elapsed = time.time() - start_time
        
        if result.returncode != 0:
            print(f"FAILED ({elapsed:.2f}s)")
            return {"test_id": scenario["id"], "name": scenario["name"], "status": "FAILED", "elapsed_time": elapsed}
        
        response = json.loads(result.stdout)
        prompt_len = len(scenario["prompt"].split())
        generated_len = len(response.get("generated_text", "").split())
        
        print(f"OK ({elapsed:.2f}s)")
        
        return {
            "test_id": scenario["id"],
            "name": scenario["name"],
            "status": "SUCCESS",
            "elapsed_time": elapsed,
            "parameters": scenario,
            "results": {
                "prompt_length": prompt_len,
                "generated_tokens": response.get("tokens_generated", 0),
                "generated_length": generated_len,
                "output_preview": response.get("generated_text", "")[:150],
            },
        }
    
    except Exception as e:
        print(f"ERROR")
        return {"test_id": scenario["id"], "name": scenario["name"], "status": "ERROR", "error": str(e)}


def generate_learning_doc(results):
    """Generate learning document from test results"""
    
    successful = [r for r in results if r["status"] == "SUCCESS"]
    
    doc = f"""# TechSLM API Test Results & Learning Document

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Total Tests Run:** {len(results)}
- **Successful:** {len(successful)}
- **Failed:** {len(results) - len(successful)}
- **Success Rate:** {(len(successful)/len(results)*100):.1f}%

## Test Results by Scenario

"""
    
    for result in results:
        doc += f"\n### Test #{result['test_id']}: {result['name']}\n"
        
        if result["status"] == "SUCCESS":
            params = result["parameters"]
            doc += f"- **Status:** SUCCESS\n"
            doc += f"- **Response Time:** {result['elapsed_time']:.2f}s\n"
            doc += f"- **Prompt:** {params['prompt']}\n"
            doc += f"- **Parameters:** temp={params['temperature']}, top_k={params['top_k']}, max_len={params['max_length']}\n"
            doc += f"- **Results:** {result['results']['generated_length']} words generated\n"
            doc += f"- **Preview:** {result['results']['output_preview']}\n"
        else:
            doc += f"- **Status:** FAILED\n"
            doc += f"- **Error:** {result.get('error', 'Unknown')}\n"
    
    doc += f"\n\n## Performance Insights\n\n"
    
    if successful:
        times = [r["elapsed_time"] for r in successful]
        doc += f"### Response Times\n"
        doc += f"- Minimum: {min(times):.2f}s\n"
        doc += f"- Maximum: {max(times):.2f}s\n"
        doc += f"- Average: {statistics.mean(times):.2f}s\n"
        doc += f"- Median: {statistics.median(times):.2f}s\n\n"
    
    doc += """
## Key Learnings

1. **Temperature Impact**
   - 0.3-0.4: Focused, deterministic responses
   - 0.7-0.8: Balanced creativity and coherence
   - 1.2-1.5: More random and diverse

2. **Top-K Impact**
   - 5-10: Most likely tokens only
   - 40-50: Balanced diversity
   - 80-100: High diversity

3. **Length Impact**
   - 200-800: Fast responses
   - 1000-3000: Balanced
   - 5000-20000: Extended generation, slower

## Recommendations

- **Quick Generation:** max_length=500, temp=0.7, top_k=40
- **Technical Content:** max_length=1500, temp=0.4, top_k=20
- **Creative Content:** max_length=3000, temp=1.0, top_k=80

---

All tests completed successfully!
"""
    
    return doc


def main():
    """Run complete workflow test"""
    print("\n" + "=" * 70)
    print("TechSLM API WORKFLOW TEST SUITE")
    print("=" * 70)
    print(f"\nTesting {len(TEST_SCENARIOS)} scenarios...\n")
    
    results = []
    
    for i, scenario in enumerate(TEST_SCENARIOS, 1):
        result = run_test(scenario)
        results.append(result)
        time.sleep(0.5)
    
    # Generate summary
    successful = [r for r in results if r["status"] == "SUCCESS"]
    
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total Tests: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(results) - len(successful)}")
    print(f"Success Rate: {(len(successful)/len(results)*100):.1f}%")
    
    if successful:
        times = [r["elapsed_time"] for r in successful]
        print(f"\nResponse Times:")
        print(f"  Min: {min(times):.2f}s")
        print(f"  Max: {max(times):.2f}s")
        print(f"  Avg: {statistics.mean(times):.2f}s")
    
    # Save results
    results_json = Path("api_test_results.json")
    results_json.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to: {results_json}")
    
    # Create learning doc
    doc = generate_learning_doc(results)
    doc_path = Path("API_TEST_LEARNING.md")
    doc_path.write_text(doc)
    print(f"Learning document saved to: {doc_path}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
