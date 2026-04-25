import json
import time
import asyncio
from typing import List, Dict
from app.services.multi_agent import multi_agent_orchestrator
from app.services.evaluation import eval_framework

class BenchmarkingService:
    def __init__(self, data_path: str = "tests/benchmark_data.json"):
        with open(data_path, "r") as f:
            self.test_cases = json.load(f)

    async def run_benchmark(self) -> Dict:
        results = []
        total_score = 0
        
        print(f"Starting Benchmark for {len(self.test_cases)} cases...")
        
        for case in self.test_cases:
            start_time = time.time()
            
            # 1. Execute Agent
            response = await multi_agent_orchestrator.run(case["query"], "bench_session")
            latency = time.time() - start_time
            
            # 2. Evaluate Accuracy (Semantic Similarity)
            accuracy = await eval_framework.evaluate_accuracy(case["golden_answer"], response)
            
            # 3. Final Result for Case
            result = {
                "id": case["id"],
                "query": case["query"],
                "accuracy": accuracy,
                "latency": f"{latency:.2f}s",
                "status": "PASS" if accuracy > 75 else "FAIL"
            }
            results.append(result)
            total_score += accuracy

        avg_score = total_score / len(self.test_cases)
        
        report = {
            "timestamp": time.time(),
            "average_accuracy": f"{avg_score:.2f}%",
            "total_cases": len(self.test_cases),
            "results": results
        }
        
        return report

benchmarking_service = BenchmarkingService()
