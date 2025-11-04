#!/usr/bin/env python3
"""
Recovery script for temporary database data.

This script allows you to recover data from jobs that failed or were interrupted
before completion. The temporary database stores results after each document
is processed, so you can salvage partial work.

Usage:
    python scripts/recover_temp_data.py [options]

Options:
    --list              List all recoverable jobs
    --recover JOB_ID    Recover a specific job by ID
    --recover-all       Recover all recoverable jobs
    --clean-old         Clean up old temporary data (7+ days)
    --show-progress JOB_ID  Show progress for a specific job
"""

import argparse
import sys
from pathlib import Path

# Add the project root and src directory to the path so we can import our modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.temp_results_db import (
    list_temp_runs, get_temp_run_progress, get_temp_run_results,
    recover_from_temp_run, cleanup_old_temp_runs
)
from src.job_manager import get_recoverable_jobs, recover_job_from_temp


def print_banner():
    """Print a nice banner for the recovery script."""
    print("=" * 80)
    print("🔧 TEMPORARY DATABASE RECOVERY TOOL")
    print("=" * 80)


def list_recoverable_jobs():
    """List all jobs that can be recovered from temporary database."""
    print("🔍 Scanning for recoverable jobs...")
    print("=" * 80)
    
    try:
        # Get jobs from job manager that have temp data
        recoverable_jobs = get_recoverable_jobs()
        
        # Also get raw temp runs
        temp_runs = list_temp_runs()
        
        if not recoverable_jobs and not temp_runs:
            print("✅ No recoverable jobs found.")
            print()
            print("This means either:")
            print("  • All jobs completed successfully")
            print("  • No temporary data exists")
            print("  • Temporary data has been cleaned up")
            return
        
        # Display recoverable jobs
        if recoverable_jobs:
            print(f"📋 Found {len(recoverable_jobs)} recoverable jobs:")
            print()
            
            for job in recoverable_jobs:
                progress = get_temp_run_progress(job['job_id'])
                if progress:
                    print(f"🔄 {job['job_id']}")
                    print(f"   📝 Run: {progress['run_name']}")
                    print(f"   📊 Progress: {progress['processed_docs']}/{progress['total_docs']} docs ({progress['progress_percent']:.1f}%)")
                    print(f"   📅 Created: {_format_timestamp(progress['created_at'])}")
                    print(f"   🔧 Recovery: python scripts/recover_temp_data.py --recover {job['job_id']}")
                    print()
        
        # Display orphaned temp runs (not in job manager)
        orphaned_runs = [run for run in temp_runs if run['job_id'] not in [job['job_id'] for job in recoverable_jobs]]
        if orphaned_runs:
            print(f"⚠️  Found {len(orphaned_runs)} orphaned temporary runs:")
            print("   (These have temp data but no active job record)")
            print()
            
            for run in orphaned_runs:
                print(f"📦 {run['job_id']}")
                print(f"   📝 Run: {run['run_name']}")
                print(f"   📊 Progress: {run['processed_docs']}/{run['total_docs']} docs ({run['progress_percent']:.1f}%)")
                print(f"   📅 Created: {_format_timestamp(run['created_at'])}")
                print(f"   🔧 Recovery: python scripts/recover_temp_data.py --recover {run['job_id']}")
                print()
                
    except Exception as e:
        print(f"❌ Error scanning for recoverable jobs: {e}")


def show_job_progress(job_id: str):
    """Show detailed progress for a specific job."""
    print(f"🔍 Showing progress for job: {job_id}")
    print("=" * 80)
    
    try:
        progress = get_temp_run_progress(job_id)
        if not progress:
            print(f"❌ No temporary data found for job '{job_id}'")
            return
        
        print(f"📝 **Run Name:** {progress['run_name']}")
        print(f"📁 **Folders:** {', '.join(progress['folders'])}")
        print(f"🤖 **Models:** {', '.join(progress['models'])}")
        print(f"🔄 **Iterations:** {progress['iterations']}")
        print(f"📊 **Progress:** {progress['processed_docs']}/{progress['total_docs']} documents ({progress['progress_percent']:.1f}%)")
        print(f"📅 **Created:** {_format_timestamp(progress['created_at'])}")
        print(f"🕐 **Updated:** {_format_timestamp(progress['updated_at'])}")
        print()
        
        # Show individual document results
        results = get_temp_run_results(job_id)
        if results and results.get('docs'):
            print(f"📄 **Processed Documents ({len(results['docs'])}):**")
            for i, doc in enumerate(results['docs'], 1):
                print(f"   {i:2d}. {doc.get('document', 'Unknown')}")
                if 'runs' in doc:
                    print(f"       └─ {len(doc['runs'])} drafts generated")
        
        print()
        print(f"🔧 **To recover this job:**")
        print(f"   python scripts/recover_temp_data.py --recover {job_id}")
        
    except Exception as e:
        print(f"❌ Error showing progress for job '{job_id}': {e}")


def recover_single_job(job_id: str):
    """Recover a single job from temporary database."""
    print(f"🔧 Recovering job: {job_id}")
    print("=" * 80)
    
    try:
        # First try using job manager recovery (preferred)
        recovered_run_name = recover_job_from_temp(job_id)
        
        if recovered_run_name:
            print(f"✅ Successfully recovered job '{job_id}' as run '{recovered_run_name}'")
            print(f"📊 Run is now available in the main results database")
            return True
        else:
            # Fallback to direct temp database recovery
            print("⚠️  Job manager recovery failed, trying direct recovery...")
            recovered_run_name = recover_from_temp_run(job_id)
            
            if recovered_run_name:
                print(f"✅ Successfully recovered job '{job_id}' as run '{recovered_run_name}'")
                print(f"📊 Run is now available in the main results database")
                return True
            else:
                print(f"❌ Failed to recover job '{job_id}'")
                return False
                
    except Exception as e:
        print(f"❌ Error recovering job '{job_id}': {e}")
        return False


def recover_all_jobs():
    """Recover all recoverable jobs."""
    print("🔧 Recovering all jobs...")
    print("=" * 80)
    
    try:
        # Get all recoverable jobs
        recoverable_jobs = get_recoverable_jobs()
        temp_runs = list_temp_runs()
        
        # Combine and deduplicate by job_id
        all_job_ids = set()
        for job in recoverable_jobs:
            all_job_ids.add(job['job_id'])
        for run in temp_runs:
            all_job_ids.add(run['job_id'])
        
        if not all_job_ids:
            print("✅ No recoverable jobs found.")
            return
        
        print(f"📋 Found {len(all_job_ids)} jobs to recover")
        print()
        
        succeeded = []
        failed = []
        
        for job_id in all_job_ids:
            print(f"🔧 Recovering {job_id}...")
            if recover_single_job(job_id):
                succeeded.append(job_id)
                print(f"   ✅ Success")
            else:
                failed.append(job_id)
                print(f"   ❌ Failed")
            print()
        
        print("=" * 80)
        print("📊 **RECOVERY SUMMARY**")
        print(f"✅ Succeeded: {len(succeeded)}")
        print(f"❌ Failed: {len(failed)}")
        
        if succeeded:
            print("\n✅ **Successfully recovered:**")
            for job_id in succeeded:
                print(f"   • {job_id}")
        
        if failed:
            print("\n❌ **Failed to recover:**")
            for job_id in failed:
                print(f"   • {job_id}")
                
    except Exception as e:
        print(f"❌ Error during bulk recovery: {e}")


def clean_old_data():
    """Clean up old temporary data."""
    print("🧹 Cleaning up old temporary data (7+ days)...")
    print("=" * 80)
    
    try:
        cleanup_old_temp_runs(days=7)
        print("✅ Old temporary data cleaned up successfully")
        
    except Exception as e:
        print(f"❌ Error cleaning up old data: {e}")


def _format_timestamp(timestamp: float) -> str:
    """Format a unix timestamp for display."""
    try:
        from datetime import datetime
        return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    except:
        return f"<timestamp: {timestamp}>"


def main():
    """Main entry point for the recovery script."""
    parser = argparse.ArgumentParser(
        description="Recovery script for temporary database data",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--list", action="store_true", help="List all recoverable jobs")
    parser.add_argument("--recover", metavar="JOB_ID", help="Recover a specific job by ID")
    parser.add_argument("--recover-all", action="store_true", help="Recover all recoverable jobs")
    parser.add_argument("--clean-old", action="store_true", help="Clean up old temporary data (7+ days)")
    parser.add_argument("--show-progress", metavar="JOB_ID", help="Show progress for a specific job")
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    print_banner()
    
    try:
        if args.list:
            list_recoverable_jobs()
        elif args.show_progress:
            show_job_progress(args.show_progress)
        elif args.recover:
            recover_single_job(args.recover)
        elif args.recover_all:
            recover_all_jobs()
        elif args.clean_old:
            clean_old_data()
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 