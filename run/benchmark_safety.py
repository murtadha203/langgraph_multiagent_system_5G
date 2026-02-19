
import sys
import os

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can import from run (if run is package) OR just import from sibling
# Since we are in run/, we can't import run.benchmark_scenarios easily without init
# simpler: import the file by path or assume run is package.
# Let's try direct import if in path, or use exec?
# Proper way:
from run.benchmark_scenarios import *
from src.agents.legacy_agents import LegacyHOAgent, LegacyMECAgent

def main():
    print("Loading Agents for Safety Verification...")
    ho_ppo = HOAgentPPO()
    mec_ppo = MECAgentPPO()
    
    if os.path.exists(HO_CHECKPOINT):
        ho_ppo.load(HO_CHECKPOINT)
        print(f"Loaded Obedience Policy: {HO_CHECKPOINT}")
    else:
        print("[WARNING] Obedience Policy not found!")
    mec_ppo.load(MEC_CHECKPOINT)
    
    ho_legacy = LegacyHOAgent()
    mec_legacy = LegacyMECAgent()
    
    all_results = []
    
    # ONLY SAFETY SCENARIO
    scenarios = [
        ("Safety_Critical", setup_safety_scenario, 1000)
    ]
    
    # 3-Tier Configuration
    configs = [
        {"name": "Tier1_Legacy", "orch": False, "ho": ho_legacy, "mec": mec_legacy},
        {"name": "Tier2_Static", "orch": False, "ho": ho_ppo,    "mec": mec_ppo},
        {"name": "Tier3_Orch",   "orch": True,  "ho": ho_ppo,    "mec": mec_ppo}
    ]
    
    for sc_name, sc_setup, steps in scenarios:
        print(f"\n=== SCENARIO: {sc_name} ===")
        
        for config in configs:
            run_name = config["name"]
            orch = config["orch"]
            ho = config["ho"]
            mec = config["mec"]
            
            for seed in SEEDS:
                sim = NetworkSimulation(seed=seed)
                sim.reset(seed=seed)
                sc_setup(sim)
                
                df = run_scenario_episode(sc_name, sim, ho, mec, orch, steps, seed)
                df['setup'] = run_name
                df['agent_type'] = "Legacy" if "Legacy" in run_name else "PPO"
                
                all_results.append(df)
            
    final_df = pd.concat(all_results, ignore_index=True)
    out_path = os.path.join(RESULTS_DIR, "safety_verification_results.csv")
    final_df.to_csv(out_path, index=False)
    print(f"\n>> Results saved to {out_path}")

if __name__ == "__main__":
    main()
