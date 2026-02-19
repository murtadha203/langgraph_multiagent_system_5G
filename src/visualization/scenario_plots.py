"""
Visualization Script for Advanced Benchmark Scenarios.
Generates:
1. Comparative Bar Charts (Baseline vs Orchestrator).
2. Time-Series for "Day in the Life" with Mode-Colored Backgrounds.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import sys

# Add project root to path
# But this script is usually run from root or run/ folder.
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(project_root, "data", "scenarios")
PLOTS_DIR = os.path.join(project_root, "plots", "scenarios")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Color Palette
COLORS = {
    "Baseline": "#7f7f7f",
    "Orchestrator": "#1f77b4",
    "Balanced": "#e6f3ff", # Light Blue
    "Green": "#e6ffe6",    # Light Green
    "Survival": "#ffe6e6", # Light Red
}

def load_data():
    path = os.path.join(RESULTS_DIR, "scenario_benchmark_results.csv")
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return None
    return pd.read_csv(path)

def plot_comparative_metrics(df):
    """
    Bar charts for Safety, Green, Congestion scenarios.
    Metrics: Drop Rate, Energy, Latency.
    """
    scenarios = ["Safety_Critical", "Green_IoT", "Congestion_Collapse"]
    metrics = [
        ("dropped", "Drop Rate (%)", lambda x: x*100), # Mean dropped is 0/1, so *100 is %
        ("energy", "Avg Energy (J)", lambda x: x),
        ("latency", "Avg Latency (ms)", lambda x: x)
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Pre-process data
    summary_list = []
    
    for sc in scenarios:
        df_sc = df[df['scenario'] == sc]
        if df_sc.empty: continue
        
        for orch_bool in [False, True]:
            label = "Orchestrator" if orch_bool else "Baseline"
            subset = df_sc[df_sc['orchestrator'] == orch_bool]
            
            # Aggregate by seed first
            start_metrics = subset.groupby('seed')[['dropped', 'energy', 'latency']].mean().reset_index()
            
            # Then mean/std over seeds
            res = {
                "Scenario": sc,
                "System": label,
                "Drop Rate (%)": start_metrics['dropped'].mean() * 100,
                "Drop_Std": start_metrics['dropped'].std() * 100,
                "Avg Energy (mJ)": start_metrics['energy'].mean() * 1000.0, # Convert to mJ
                "Energy_Std": start_metrics['energy'].std() * 1000.0,
                "Avg Latency (ms)": start_metrics['latency'].mean(),
                "Latency_Std": start_metrics['latency'].std()
            }
            summary_list.append(res)
            
    df_summ = pd.DataFrame(summary_list)
    
    if df_summ.empty:
        print("No data for comparative plots.")
        return

    # Plot
    # 1. Drop Rate (Safety & Congestion)
    sns.barplot(data=df_summ, x="Scenario", y="Drop Rate (%)", hue="System", ax=axes[0], palette=[COLORS["Baseline"], COLORS["Orchestrator"]])
    axes[0].set_title("Reliability: Task Drop Rate")
    axes[0].set_ylabel("Drop Rate (%)")
    
    # 2. Energy (Green & Others)
    sns.barplot(data=df_summ, x="Scenario", y="Avg Energy (mJ)", hue="System", ax=axes[1], palette=[COLORS["Baseline"], COLORS["Orchestrator"]])
    axes[1].set_title("Efficiency: Energy Consumption")
    axes[1].set_ylabel("Avg Energy (mJ)")
    
    # 3. Latency (Congestion & Safety)
    sns.barplot(data=df_summ, x="Scenario", y="Avg Latency (ms)", hue="System", ax=axes[2], palette=[COLORS["Baseline"], COLORS["Orchestrator"]])
    axes[2].set_title("Performance: Average Latency")
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "comparative_metrics.png"))
    print("Saved comparative_metrics.png")

def plot_day_in_life(df):
    """
    Time-series plot for Day in the Life scenario.
    """
    df_day = df[df['scenario'] == "Day_in_the_Life"]
    if df_day.empty: return
    
    # Focus on Orchestrator run (to show mode switching)
    # We can also plot Baseline for comparison
    
    df_orch = df_day[df_day['orchestrator'] == True]
    df_base = df_day[df_day['orchestrator'] == False]
    
    # Aggregate over seeds (mean) for line plots
    # Group by step - Select numeric columns only
    numeric_cols = ['battery', 'rsrp', 'latency', 'load', 'energy', 'dropped', 'handovers']
    orch_mean = df_orch.groupby('step')[numeric_cols].mean().reset_index()
    base_mean = df_base.groupby('step')[numeric_cols].mean().reset_index()
    
    # Identify Mode Zones (from Orchestrator run)
    # We use the MAJORITY mode across seeds if they differ, or just one seed representative
    # Better: Use the mode from seed 907 as representative for background
    df_rep = df_orch[df_orch['seed'] == 907].sort_values('step')
    
    # Fallback if 907 not found (e.g. if partial run)
    if df_rep.empty and not df_orch.empty:
         first_seed = df_orch['seed'].unique()[0]
         df_rep = df_orch[df_orch['seed'] == first_seed].sort_values('step')
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
    
    # 1. RSRP & Handover
    ax = axes[0]
    ax.plot(base_mean['step'], base_mean['rsrp'], label='Baseline RSRP', color='grey', alpha=0.6)
    ax.plot(orch_mean['step'], orch_mean['rsrp'], label='Orchestrator RSRP', color='blue')
    ax.set_ylabel("RSRP (dBm)")
    ax.set_title("Signal Quality & Connectivity")
    ax.legend(loc='lower left')
    
    # 2. Battery
    ax = axes[1]
    ax.plot(base_mean['step'], base_mean['battery'], label='Baseline Battery %', color='grey', linestyle='--')
    ax.plot(orch_mean['step'], orch_mean['battery'], label='Orchestrator Battery %', color='green')
    ax.set_ylabel("Battery (%)")
    ax.set_title("Energy Consumption")
    ax.legend()
    
    # 3. Latency / Load
    ax = axes[2]
    ax.plot(base_mean['step'], base_mean['latency'], label='Baseline Latency', color='grey', alpha=0.5)
    ax.plot(orch_mean['step'], orch_mean['latency'], label='Orchestrator Latency', color='orange')
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Performance")
    
    # Add Mode Backgrounds to ALL axes
    # We iterate through the representative run and find state changes
    modes = df_rep['mode'].values
    steps = df_rep['step'].values
    
    changes = []
    current = modes[0]
    start_step = steps[0]
    
    for i, m in enumerate(modes):
        if m != current:
            changes.append((current, start_step, steps[i]))
            current = m
            start_step = steps[i]
    changes.append((current, start_step, steps[-1]))
    
    for ax in axes:
        for mode, start, end in changes:
            color = COLORS.get(mode, "white")
            # Clean string if needed (some CSV might have "ControlMode.BALANCED")
            if "BALANCED" in str(mode): color = COLORS["Balanced"]
            elif "GREEN" in str(mode): color = COLORS["Green"]
            elif "SURVIVAL" in str(mode): color = COLORS["Survival"]
            
            ax.axvspan(start, end, facecolor=color, alpha=0.3)
            
    # Add text labels for phases
    axes[0].text(500, -50, "Phase 1: Congestion", ha='center', fontsize=12)
    axes[0].text(1500, -50, "Phase 2: Green Mode", ha='center', fontsize=12)
    axes[0].text(2500, -50, "Phase 3: Survival", ha='center', fontsize=12)
    
    # Add Legend for Background Colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS["Balanced"], alpha=0.3, label='balanced Intent'),
        Patch(facecolor=COLORS["Green"], alpha=0.3, label='green Intent'),
        Patch(facecolor=COLORS["Survival"], alpha=0.3, label='survival Intent')
    ]
    # Place legend outside or in a corner
    axes[0].legend(handles=legend_elements, loc='upper right', title="Orchestrator Mode")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "day_in_life_timeline.png"))
    print("Saved day_in_life_timeline.png")

def plot_congestion(df):
    """Scenario 3: Connection Stability under Load"""
    print("Plotting Scenario 3: Congestion...")
    subset = df[df["scenario"] == "Congestion_Collapse"]
    if subset.empty:
        print("No data for Congestion_Collapse")
        return
    
    # Use 'connected' metric if available, else infer from drops or similar
    if 'connected' not in subset.columns:
        print("Warning: 'connected' metric not found. Skipping congestion plot.")
        return

    # Aggregate
    # We want to see stability across the run.
    # Group by orchestrator/seed
    # 0 = Orchestrator=False (Baseline), 1 = Orchestrator=True
    
    summary = subset.groupby(["orchestrator", "seed"])["connected"].mean().reset_index()
    # Map boolean to string
    summary["System"] = summary["orchestrator"].map({False: "Baseline", True: "Orchestrator"})
    
    plt.figure(figsize=(8, 6))
    sns.barplot(data=summary, x="System", y="connected", palette=[COLORS["Baseline"], COLORS["Orchestrator"]])
    plt.title("Scenario 3: Connection Stability under High Load")
    plt.ylabel("Connectivity Ratio (1.0 = Always Connected)")
    plt.ylim(0.95, 1.005) # Zoom in as we expect high reliability usually
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "congestion_stability.png"))
    print("Saved congestion_stability.png")

def main():
    print("Loading data...")
    df = load_data()
    if df is not None:
        plot_comparative_metrics(df)
        plot_day_in_life(df)
        plot_congestion(df)
        print("Done.")

if __name__ == "__main__":
    main()
