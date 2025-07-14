#!/usr/bin/env python3
"""
BitMar Benchmark Runner
Runs all text benchmarks and saves results
"""

import subprocess
import sys
import json
import logging
from pathlib import Path
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_benchmark_script(script_name: str) -> dict:
    """Run a benchmark script and return results"""
    logger.info(f"🏃 Running {script_name}")
    
    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode == 0:
            logger.info(f"✅ {script_name} completed successfully")
            return {
                'status': 'success',
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
        else:
            logger.error(f"❌ {script_name} failed with code {result.returncode}")
            return {
                'status': 'failed',
                'stdout': result.stdout,
                'stderr': result.stderr,
                'returncode': result.returncode
            }
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ {script_name} timed out")
        return {
            'status': 'timeout',
            'stdout': '',
            'stderr': 'Script timed out after 5 minutes',
            'returncode': -1
        }
    except Exception as e:
        logger.error(f"💥 {script_name} crashed: {e}")
        return {
            'status': 'error',
            'stdout': '',
            'stderr': str(e),
            'returncode': -1
        }


def main():
    """Run all benchmark tests"""
    logger.info("🚀 BitMar Benchmark Suite")
    logger.info("=" * 50)
    
    start_time = time.time()
    
    # List of benchmark scripts to run
    benchmark_scripts = [
        'test_nlp_capabilities.py',
        'test_bitnet_cpu.py',
        'test_quick_benchmarks.py',
        'test_text_benchmarks.py'
    ]
    
    results = {}
    
    for script in benchmark_scripts:
        if Path(script).exists():
            results[script] = run_benchmark_script(script)
        else:
            logger.warning(f"⚠️  {script} not found, skipping")
            results[script] = {
                'status': 'not_found',
                'stdout': '',
                'stderr': f'Script {script} not found',
                'returncode': -1
            }
    
    total_time = time.time() - start_time
    
    # Summary
    logger.info("=" * 50)
    logger.info("📊 Benchmark Results Summary")
    logger.info("=" * 50)
    
    success_count = 0
    for script, result in results.items():
        status = result['status']
        if status == 'success':
            logger.info(f"✅ {script}: PASSED")
            success_count += 1
        elif status == 'failed':
            logger.info(f"❌ {script}: FAILED (code {result['returncode']})")
        elif status == 'timeout':
            logger.info(f"⏰ {script}: TIMEOUT")
        elif status == 'not_found':
            logger.info(f"⚠️  {script}: NOT FOUND")
        else:
            logger.info(f"💥 {script}: ERROR")
    
    success_rate = success_count / len(benchmark_scripts)
    
    logger.info(f"\n📈 Success Rate: {success_rate:.1%} ({success_count}/{len(benchmark_scripts)})")
    logger.info(f"⏱️  Total Time: {total_time:.2f}s")
    
    # Save detailed results
    detailed_results = {
        'summary': {
            'total_scripts': len(benchmark_scripts),
            'successful': success_count,
            'success_rate': success_rate,
            'total_time': total_time,
            'timestamp': time.time()
        },
        'detailed_results': results
    }
    
    # Save results
    results_file = Path("all_benchmark_results.json")
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    logger.info(f"\n💾 Detailed results saved to: {results_file}")
    
    # Overall assessment
    if success_rate >= 0.75:
        logger.info("🎉 BitMar shows excellent benchmark performance!")
        return 0
    elif success_rate >= 0.5:
        logger.info("👍 BitMar shows good benchmark performance.")
        return 0
    else:
        logger.warning("⚠️  BitMar needs improvement on benchmarks.")
        return 1


if __name__ == "__main__":
    exit(main())
