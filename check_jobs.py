#!/usr/bin/env python3
"""
Temporary script to check active jobs and extract data
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from job_manager import get_active_jobs, get_job
    
    print("🔍 Checking active jobs...")
    active_jobs = get_active_jobs()
    
    if not active_jobs:
        print("❌ No active jobs found")
        sys.exit(0)
    
    print(f"📊 Found {len(active_jobs)} active job(s):")
    
    for job in active_jobs:
        print(f"\n📋 Job Details:")
        print(f"  ID: {job['id']}")
        print(f"  Name: {job['name']}")
        print(f"  Status: {job['status']}")
        print(f"  Processed: {job.get('processed_docs', 0)}")
        print(f"  Started: {job.get('start_time', 'unknown')}")
        
        # Try to get full job details
        full_job = get_job(job['id'])
        if full_job and 'results' in full_job:
            results = full_job['results']
            print(f"  Results available: {len(results)} documents")
            
            # Ask if user wants to save these results
            print(f"\n💾 Would you like to save the current results to a file?")
            print(f"   This will create: temp_results_{job['name']}.json")
            
            # Save the results
            output_file = f"temp_results_{job['name']}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "job_id": job['id'],
                    "name": job['name'],
                    "status": job['status'],
                    "processed_docs": job.get('processed_docs', 0),
                    "results": results,
                    "exported_at": "2025-01-24 17:59:00 UTC"
                }, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Results saved to: {output_file}")
            print(f"📁 File size: {Path(output_file).stat().st_size / 1024:.1f} KB")
        else:
            print("  ⚠️ No results data available yet")

except Exception as e:
    print(f"❌ Error accessing job data: {e}")
    import traceback
    traceback.print_exc() 