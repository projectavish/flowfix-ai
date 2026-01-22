"""
FlowFix AI - Master Application Runner
Run this single file to execute entire pipeline and launch dashboard
"""
import subprocess
import sys
import os
from datetime import datetime
import time

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")

def run_script(script_name, description, args=None):
    """Run a Python script and return success status"""
    print(f"\n[RUNNING] {description}...")
    print(f"Script: {script_name}")
    
    cmd = [sys.executable, "-m", f"src.{script_name.replace('.py','')}"]
    if args:
        cmd.extend(args)
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            check=True
        )
        duration = time.time() - start_time
        print(f"[SUCCESS] {description} completed in {duration:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"[ERROR] {description} failed after {duration:.1f}s")
        print(f"Error: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Failed to run {script_name}: {e}")
        return False

def main():
    """Run entire FlowFix AI pipeline"""
    
    print_header("FLOWFIX AI - AUTOMATED PIPELINE")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Working Directory: {os.getcwd()}")
    
    pipeline_start = time.time()
    results = {}
    
    # Step 1: Bottleneck Detection
    print_header("STEP 1: BOTTLENECK DETECTION")
    results['bottleneck_detection'] = run_script(
        'bottleneck_detector.py',
        'Analyzing workflow bottlenecks'
    )
    
    # Step 2: ML Model Training & Predictions
    print_header("STEP 2: MACHINE LEARNING")
    results['ml_training'] = run_script(
        'ml_predictor.py',
        'Training ML models and generating predictions'
    )
    
    # Step 3: GPT AI Suggestions
    print_header("STEP 3: AI RECOMMENDATIONS")

    print("[INFO] GPT suggestions are optional and currently disabled (no API key).")
    print("[INFO] Skipping gpt_suggester.py")

    results['gpt_suggestions'] = True
   
    
    # Step 4: PDF Report Generation
    print_header("STEP 4: REPORT GENERATION")
    results['pdf_report'] = run_script(
        'pdf_generator.py',
        'Generating comprehensive PDF report'
    )
    
    # Step 5: PowerBI Export
    print_header("STEP 5: POWER BI EXPORT")
    results['powerbi_export'] = run_script(
        'export_for_powerbi.py',
        'Exporting data for Power BI'
    )
    
    # Step 6: Improvement Tracking
    print_header("STEP 6: IMPROVEMENT METRICS")
    results['improvement_tracking'] = run_script(
        'improvement_tracker.py',
        'Calculating improvement metrics'
    )
    
    # Summary
    print_header("PIPELINE SUMMARY")
    
    total_duration = time.time() - pipeline_start
    successful = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"Total Duration: {total_duration:.1f}s")
    print(f"Success Rate: {successful}/{total} steps completed\n")
    
    print("Step Results:")
    for step, success in results.items():
        status = "[SUCCESS]" if success else "[FAILED]"
        print(f"  {status} {step.replace('_', ' ').title()}")
    
    # Launch Dashboard
    if successful >= 3:  # At least 3 steps succeeded
        print_header("LAUNCHING DASHBOARD")
        print("\n[INFO] Starting Streamlit Dashboard...")
        print("[INFO] Dashboard will open in your browser")
        print("[INFO] Press CTRL+C to stop the dashboard\n")
        
        time.sleep(2)
        
        try:
            subprocess.run([
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "dashboard/streamlit_app.py",
                "--server.headless=true"
            ])
        except KeyboardInterrupt:
            print("\n\n[INFO] Dashboard stopped by user")
        except Exception as e:
            print(f"\n[ERROR] Failed to launch dashboard: {e}")
            print("\nManual launch: streamlit run dashboard/streamlit_app.py")
    else:
        print("\n[WARNING] Too many pipeline failures. Fix errors before launching dashboard.")
        print(f"\nFailed steps: {total - successful}")
    
    print_header("FLOWFIX AI - PIPELINE COMPLETE")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n[ERROR] Pipeline failed: {e}")
        sys.exit(1)
