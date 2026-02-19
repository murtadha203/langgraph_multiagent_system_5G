
import pandas as pd
import matplotlib.pyplot as pd_plt
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Set style
sns.set_theme(style="whitegrid")

def plot_ablation_study():
    # Load Data
    csv_path = os.path.join("data", "scenarios", "scenario_benchmark_results.csv")
    if not os.path.exists(csv_path):
        print("Data not found!")
        return
        
    df = pd.read_csv(csv_path)
    
    # Preprocessing
    # Ensure 'setup' column exists (Tier1, Tier2, Tier3)
    if 'setup' not in df.columns:
        print("Old data format (no 'setup' column). Cannot plot ablation.")
        return
        
    # Metrics to Plot
    # 1. Drop Rate (%)
    # 2. Energy (mJ)
    # 3. Latency (ms)
    # 4. Handovers (Count)
    
    # Aggregate Step-wise to Episode-wise
    # Group by: Scenario, Setup, Seed
    # Metrics: 
    #   Dropped: max(sum(dropped)) > 0 ?? No, Dropped Rate % of steps
    #   Let's use the same logic as summary: mean of boolean 'dropped' * 100
    
    # We need to Aggregate properly first
    # For bar plot, we need 1 value per (Scenario, Setup). Error bars from Seeds.
    
    # Step 1: Calculate Per-Episode Metrics (1 row per Seed)
    # Group by [Scenario, Setup, Seed]
    
    # Helper to aggregate
    def agg_logic(x):
        d = {}
        d['Drop Rate (%)'] = x['dropped'].mean() * 100
        d['Energy (mJ)'] = x['energy'].mean() * 1000
        d['Latency (ms)'] = x[x['latency'] > 0]['latency'].mean()
        d['Handovers'] = x['handovers'].sum()
        return pd.Series(d)
        
    df_ep = df.groupby(['scenario', 'setup', 'seed']).apply(agg_logic).reset_index()
    
    # Define Order
    setup_order = sorted(df['setup'].unique()) 
    # Expected: Tier1_Legacy, Tier2_Static, Tier3_Orch
    
    # Plotting
    metrics = ['Drop Rate (%)', 'Energy (mJ)', 'Latency (ms)', 'Handovers']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        sns.barplot(data=df_ep, x='scenario', y=metric, hue='setup', hue_order=setup_order, ax=ax, palette="viridis")
        ax.set_title(metric)
        ax.set_xlabel("")
        ax.legend(title='Tier')
        
    plt.tight_layout()
    out_path = os.path.join("data", "scenarios", "ablation_study_results.png")
    plt.savefig(out_path)
    print(f"Comparison Plot saved to {out_path}")
    
if __name__ == "__main__":
    plot_ablation_study()
