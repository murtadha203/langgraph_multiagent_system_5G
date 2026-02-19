import pandas as pd
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def analyze_benchmarks():
    csv_path = os.path.join(project_root, "data", "scenarios", "scenario_benchmark_results.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Group by Scenario and Orchestrator
    # Metrics: Dropped (mean * 100), Energy (mean), Latency (mean), Handovers (sum/mean?), Connected (mean)
    
    # We aggregate by Seed first to get per-episode stats?
    # Actually step-wise metrics:
    # Dropped is boolean per step (did a task drop deadline in this step?)
    # Energy is J per step
    # Latency is ms per step (only if task present)
    
    # Filter for rows where a task existed (latency > 0 or dropped is True/False)
    # Actually dropped is False by default?
    # Let's check: dropped column is boolean.
    
    print("\n=== BENCHMARK ANALYSIS ===\n")
    
    scenarios = df['scenario'].unique()
    
    for sc in scenarios:
        print(f"--- Scenario: {sc} ---")
        df_sc = df[df['scenario'] == sc]
        
        # Split by Setup (Tier 1, 2, 3)
        # Assuming setups are present in 'setup' column
        # If not present (old csv), fallback to orch logic?
        if 'setup' in df_sc.columns:
            setups = sorted(df_sc['setup'].unique())
        else:
            setups = [False, True] # Fallback
            
        for setup_val in setups:
            if 'setup' in df_sc.columns:
                subset = df_sc[df_sc['setup'] == setup_val]
                label = f"{setup_val:15s}" 
            else:
                subset = df_sc[df_sc['orchestrator'] == setup_val]
                label = "Orchestrator   " if setup_val else "Baseline       "
            
            drop_rate = subset['dropped'].mean() * 100
            avg_energy = subset['energy'].mean()
            # Filter latency > 0 for valid latency stats
            valid_lat = subset[subset['latency'] > 0]['latency']
            avg_lat = valid_lat.mean() if not valid_lat.empty else 0.0
            
            avg_ho = subset.groupby('seed')['handovers'].sum().mean()
            if pd.isna(avg_ho): avg_ho = 0.0
            avg_conn = subset['connected'].mean() * 100
            
            min_rsrp = subset['rsrp'].min()
            
            # Convert J to mJ for readability
            avg_energy_mj = avg_energy * 1000.0
            
            print(f"{label} | Drop: {drop_rate:6.2f}% | Eng: {avg_energy_mj:8.4f}mJ | Lat: {avg_lat:6.2f}ms | HO: {avg_ho:4.1f} | Conn: {avg_conn:5.1f}% | Min RSRP: {min_rsrp:.2f}")
    
    # Also write to file for viewing
    with open("summary_clean.txt", "w", encoding="utf-8") as f:
        f.write("=== BENCHMARK ANALYSIS ===\n\n")
        scenarios = df['scenario'].unique()
        for sc in scenarios:
            f.write(f"--- Scenario: {sc} ---\n")
            df_sc = df[df['scenario'] == sc]
            
            if 'setup' in df_sc.columns:
                setups = sorted(df_sc['setup'].unique())
            else:
                setups = [False, True]

            for setup_val in setups:
                if 'setup' in df_sc.columns:
                    subset = df_sc[df_sc['setup'] == setup_val]
                    label = f"{setup_val:15s}" 
                else:
                    subset = df_sc[df_sc['orchestrator'] == setup_val]
                    label = "Orchestrator   " if setup_val else "Baseline       "

                drop_rate = subset['dropped'].mean() * 100
                avg_energy = subset['energy'].mean()
                valid_lat = subset[subset['latency'] > 0]['latency']
                avg_lat = valid_lat.mean() if not valid_lat.empty else 0.0
                avg_ho = subset.groupby('seed')['handovers'].sum().mean()
                if pd.isna(avg_ho): avg_ho = 0.0
                avg_conn = subset['connected'].mean() * 100
                min_rsrp = subset['rsrp'].min()
                avg_energy_mj = avg_energy * 1000.0
                f.write(f"{label} | Drop: {drop_rate:6.2f}% | Eng: {avg_energy_mj:8.4f}mJ | Lat: {avg_lat:6.2f}ms | HO: {avg_ho:4.1f} | Conn: {avg_conn:5.1f}% | Min RSRP: {min_rsrp:.2f}\n")
            f.write("\n")

if __name__ == "__main__":
    analyze_benchmarks()
