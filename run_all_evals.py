import os
import subprocess
import time

MINERS = [
    "first_train.py",
    "second_train.py",
    "third_train.py",
    "fourth_train.py",
    "fifth_train.py",
    "sixth_train.py",
    "seventh_train.py",
    "eight_train.py",
    "train_highest_mfu.py"
]

def run_simulation(miner_file):
    print(f"\n{'='*60}")
    print(f"🚀 STARTING EVALUATION FOR: {miner_file}")
    print(f"{'='*60}\n")
    
    # Copy the specific miner to the expected location for the docker image
    # Assuming the Dockerfile relies on /test/train.py
    os.system(f"cp 'miners_score_and their code/{miner_file}' /test/train.py")
    
    start_time = time.time()
    
    try:
        # Run the official simulator script
        result = subprocess.run(
            ["python3", "/test/simulate.py"], 
            capture_output=True, 
            text=True,
            check=False # Don't throw exception on non-zero exit, capture output
        )
        
        print(result.stdout)
        
        if result.stderr:
            print("--- STDERR ---")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Failed to run simulation for {miner_file}: {e}")
        
    duration = time.time() - start_time
    print(f"\n⏱️  Evaluation completed in {duration:.2f} seconds.")

if __name__ == "__main__":
    print("🏆 STARTING THE CRUSADES Relative Benchmark 🏆")
    print("Testing all 8 ranke miners against the optimized version.\n")
    
    for miner in MINERS:
        run_simulation(miner)
        
    print(f"\n{'='*60}")
    print("🏁 ALL EVALUATIONS COMPLETE 🏁")
    print(f"{'='*60}\n")
