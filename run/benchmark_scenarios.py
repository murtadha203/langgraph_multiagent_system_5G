"""
Advanced Benchmark Scenarios for Neuro-Symbolic 6G Orchestrator.
Includes:
1. Safety Critical (URLLC/High Mobility)
2. Green IoT (Battery Constraints)
3. Congestion Collapse (Load Balancing)
4. Day in the Life (Multi-Phase Integrated Test)
"""

import numpy as np
import pandas as pd
import random
import os
import sys
import math
from typing import Dict, List, Any, Optional

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.simulation import NetworkSimulation, SERVICE_PROFILES
from src.agents.mec_agent_ppo import MECAgentPPO
from src.agents.ho_agent_ppo import HOAgentPPO
from src.orchestrator.orchestrator import StrategicOrchestrator
from src.orchestrator.schemas import ControlMode

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------
MEC_CHECKPOINT = os.path.join(project_root, "models", "mec_policy.pth")
HO_CHECKPOINT = os.path.join(project_root, "models", "ho_obedience_policy.pth") # Use the new trained agent
RESULTS_DIR = os.path.join(project_root, "data", "scenarios")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS = [907, 97, 79, 709, 101]
# SEEDS = [907] # Debug

# -------------------------------------------------------------------------
# ORCHESTRATOR HELPERS
# -------------------------------------------------------------------------
class OrchestratorWrapper:
    def __init__(self):
        self.brain = StrategicOrchestrator(checkpoint_path=HO_CHECKPOINT) # Re-uses checkpoint internally
        # But wait, StrategicOrchestrator inits TacticalExecutor which loads HO agent.
        # We need to ensure we use the same agent instances or load strictly.
        # The Orchestrator loads its own tactical agent. ensuring strict baseline comparison:
        # The Baseline run will use the raw agents loaded here.
        # The Orchestrator run will use the internal tactical agent of the orchestrator.
        # PRO: Real world test (Orchestrator wraps the agent).
        # CON: Need to ensure they are the same weights. They load from same path, so yes.

    def step(self, sim_metrics: Dict, obs: Any, context: Optional[Dict] = None, force_run: bool = False):
        # Translate Sim Metrics to Orchestrator Metrics
        # Orchestrator expects: avg_load, arrival_rate, rsrp, rlf_rate, avg_latency_ms, ue_battery_percent, avg_velocity_kmh
        return self.brain.step(metrics=sim_metrics, observation=obs, context=context, force_run=force_run)

# -------------------------------------------------------------------------
# SCENARIO ENGINE
# -------------------------------------------------------------------------
def run_scenario_episode(
    scenario_name: str,
    sim: NetworkSimulation,
    ho_agent: HOAgentPPO,
    mec_agent: MECAgentPPO,
    orchestrator_enabled: bool,
    steps: int,
    seed: int
):
    """
    Run a single episode with specific conditions.
    """
    
    # Initialize Orchestrator if enabled
    orchestrator = None
    if orchestrator_enabled:
        orchestrator = OrchestratorWrapper()
        # Ensure orchestrator starts in Balanced
        orchestrator.brain.state["current_mode"] = ControlMode.BALANCED.value
    
    # Reset Context
    # Note: Scenario setup (UE speed, etc) happens BEFORE calling this or passed in sim
    # But sim.reset() is usually called before.
    # We assume sim is already reset and configured by the caller.

    context = sim.get_context()
    prev_cell = context['serving_cell_id']
    
    # Metrics Storage
    history = []
    
    # Rolling windows for metrics smoothing (for Orchestrator inputs)
    latency_window = []
    
    print(f"  > Running {scenario_name} | Orch={orchestrator_enabled} | Seed={seed}...")

    # MEC Callback
    def mec_callback(task, ctx):
        obs = mec_agent.get_observation(ctx)
        action = mec_agent.select_action(obs, context=ctx, training=False)
        return mec_agent.action_map.get(action, "local")

    step_data = []

    for i in range(steps):
        # 1. Update Dynamic Scenario Conditions (Day in the Life phases)
        if scenario_name == "Day_in_the_Life":
            apply_day_in_life_phase(sim, i)
            
        # 2. Gather Metrics for Orchestrator
        # We need cumulative stats, but simulation gives instantaneous.
        # We'll use simple proxies from context
        
        current_load = sim.base_stations[sim.serving_cell_id].load_factor
        current_rsrp = context['serving_rsrp_dbm']
        current_batt = sim.ue.battery_joules / 1000.0 * 100 # %
        current_vel = sim.ue.speed_mps * 3.6 # km/h
        
        # Latency proxy (rolling avg of last few tasks)
        avg_lat = np.mean(latency_window) if latency_window else 20.0 # Default 20ms
        
        sim_metrics = {
            'avg_load': current_load,
            'arrival_rate': 1.0 / sim.service_profiles[0].task_interarrival_s if sim.service_profiles else 10.0,
            'rsrp': current_rsrp,
            'rlf_rate': 0.0, # Not easily tracked instantaneously, assumed 0 unless failure
            'avg_latency_ms': avg_lat,
            'ue_battery_percent': current_batt,
            'avg_velocity_kmh': current_vel
        }
        
        # 3. Decision Step
        active_mode = "BALANCED"
        shield_active = False
        
        ho_decision = None
        
        if orchestrator_enabled and orchestrator:
            # ORCHESTRATOR PATH
            
            # 1. PANIC BUTTON (Reflex)
            # Check immediately at start of step
            is_panic, reason = sim.check_system_panic()
            
            if is_panic:
                 # Force Orchestrator to run NOW
                 # Note: observation is None here because we haven't requested it yet.
                 # The orchestrator should handle None observation if only doing strategic update.
                 # OR we pass a dummy or current context obs.
                 print(f"[System] PANIC TRIGGERED: {reason}")
                 dummy_obs = ho_agent.get_observation(context)
                 orchestrator.step(sim_metrics, dummy_obs, context=context, force_run=True)
                 shield_active = True # Assume panic triggers shield or similar
            
            # 2. Get Puppeteer Weights
            # The Orchestrator's internal state holds the latest "orders" (weights)
            # We inject these into the context so the Agent sees them
            current_weights = orchestrator.brain.current_weights
            # Debug/Fallback
            if not current_weights: current_weights = {'alpha':0.33, 'beta':0.33, 'gamma':0.34}
            
            # INJECT into context for get_observation
            context['intent_weights'] = current_weights
            
            # 3. Get Observation (Now includes Weights)
            obs_ho = ho_agent.get_observation(context) 
            
            # 4. Orchestrator Decision (Passes obs to internal tactical agent)
            # 4. Orchestrator Decision (Passes obs to internal tactical agent)
            ho_decision = orchestrator.step(sim_metrics, obs_ho, context=context, force_run=False)
            
            active_mode = orchestrator.brain.state.get("final_mode", "BALANCED")
            if isinstance(active_mode, ControlMode): active_mode = active_mode.value
            shield_active = orchestrator.brain.state.get("shield_active", False)
            
        else:
            # BASELINE PATH (Standard Agent - FROZEN WEIGHTS)
            
            # Baseline is "Balanced" forever
            # We fix the weights to 0.33, 0.33, 0.33
            context['intent_weights'] = {'alpha':0.33, 'beta':0.33, 'gamma':0.34}
            
            # Ensure agent is using these weights
            ho_agent.context_weights = context['intent_weights']

            obs_ho = ho_agent.get_observation(context)
            ho_action, _, _ = ho_agent.select_action_with_info(obs_ho, context=context, training=False)
            ho_decision = int(ho_action)
            
            active_mode = "BALANCED" # Implicit
            shield_active = False

        # 4. Simulation Step
        next_context, info = sim.step(ho_decision, mec_callback)
        
        # 5. Process Result
        step_reward = 0.0 # We don't have direct access to internal reward easily unless calculated
        # But we can log raw metrics
        
        task_info = info.get('task_info')
        dropped = False
        latency = 0.0
        energy = 0.0
        
        if task_info:
            latency = task_info.get('latency_s', 0) * 1000 # ms
            energy = task_info.get('energy_j', 0)
            latency_window.append(latency)
            if len(latency_window) > 10: latency_window.pop(0)
            
            if not task_info.get('deadline_met'):
                dropped = True
        
        # Log Step
        log_entry = {
            'step': i,
            'scenario': scenario_name,
            'orchestrator': orchestrator_enabled,
            'seed': seed,
            'mode': active_mode,
            'shield_active': shield_active,
            'battery': current_batt,
            'rsrp': current_rsrp,
            'load': current_load,
            'latency': latency,
            'dropped': dropped,
            'energy': energy,
            'connected': 0.0 if sim.is_rlf_active else 1.0,
            'handovers': 1 if next_context['serving_cell_id'] != prev_cell else 0
        }
        step_data.append(log_entry)
        
        context = next_context
        prev_cell = context['serving_cell_id']

    return pd.DataFrame(step_data)


def apply_day_in_life_phase(sim: NetworkSimulation, step: int):
    """
    Phase 1 (0-1000): Congestion.
    Phase 2 (1000-2000): Green (Low Traffic, Low Battery).
    Phase 3 (2000-3000): Safety (High Speed, Coverage Holes).
    """
    # RESET conditions first to avoid bleed-over
    # We need to forcefully set state
    
    if step < 1000:
        # Phase 1: Congestion
        # High load on serving cell, others moderate
        if step % 50 == 0: # Periodically refresh load
            sim.base_stations[sim.serving_cell_id].load_factor = 0.95
            for bs in sim.base_stations:
                if bs.id != sim.serving_cell_id:
                    bs.load_factor = 0.4
                    
        sim.ue.speed_mps = 5.0 # Moderate speed
        
    elif 1000 <= step < 2000:
        # Phase 2: Green Mode
        # Low traffic, Battery Critical
        sim.ue.speed_mps = 1.0 # Low speed
        sim.ue.battery_joules = 150.0 # Force 15% battery (Low/Critical)
        
        if step % 50 == 0:
            for bs in sim.base_stations:
                bs.load_factor = 0.1 # Low load
                
    elif step >= 2000:
        # Phase 3: Survival Mode
        # High Speed, RSRP Danger
        sim.ue.speed_mps = 35.0 # ~120 km/h (High speed)
        sim.ue.battery_joules = 500.0 # Restore battery so it's not the bottleneck
        
        # Inject Coverage Hole? 
        # Actually simplest is just high mobility causing RSRP drops
        # We can also inject 'shadowing' or forcing anomaly
        # Let's rely on speed + naturally varying shadowing
        pass

# -------------------------------------------------------------------------
# SCENARIO DEFINITIONS
# -------------------------------------------------------------------------

def setup_safety_scenario(sim: NetworkSimulation):
    # High Speed, URLLC
    sim.ue.speed_mps = 30.0 # 108 km/h
    sim.service_weights = {"EV": 1.0} # URLLC only
    sim.service_profiles = [SERVICE_PROFILES["EV"]]
    # Correlated shadowing already active

def setup_green_scenario(sim: NetworkSimulation):
    # Low Speed, IoT, Low Battery
    sim.ue.speed_mps = 1.0
    sim.ue.battery_joules = 100.0 # 10% -> Critical
    sim.service_weights = {"IoT": 1.0}
    sim.service_profiles = [SERVICE_PROFILES["IoT"]]

def setup_congestion_scenario(sim: NetworkSimulation):
    # eMBB, High Load
    sim.service_weights = {"VR": 1.0}
    sim.service_profiles = [SERVICE_PROFILES["VR"]]
    # Load is set dynamically or initially
    # We force serving cell load in the loop or here?
    # Better to force it here initially
    for bs in sim.base_stations:
        bs.load_factor = 0.9 if bs.id == sim.serving_cell_id else 0.3

def setup_day_in_life_scenario(sim: NetworkSimulation):
    # Dynamic - handled in loop
    sim.service_weights = {"VR": 1.0} # Default
    sim.ue.battery_joules = 1000.0

# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------

from src.agents.legacy_agents import LegacyHOAgent, LegacyMECAgent

def main():
    print("Loading Agents...")
    # Load PPO Models (Tier 2 & 3)
    ho_ppo = HOAgentPPO()
    mec_ppo = MECAgentPPO()
    
    if os.path.exists(HO_CHECKPOINT):
        ho_ppo.load(HO_CHECKPOINT)
        print(f"Loaded Obedience Policy: {HO_CHECKPOINT}")
    else:
        print("[WARNING] Obedience Policy not found! Using fallback.")
    mec_ppo.load(MEC_CHECKPOINT)
    
    # Initialize Legacy Agents (Tier 1)
    ho_legacy = LegacyHOAgent()
    mec_legacy = LegacyMECAgent()
    
    all_results = []
    
    scenarios = [
        ("Safety_Critical", setup_safety_scenario, 1000),
        ("Green_IoT", setup_green_scenario, 2000),
        ("Congestion_Collapse", setup_congestion_scenario, 1000),
        ("Day_in_the_Life", setup_day_in_life_scenario, 3000)
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
                
                # Run Episode
                df = run_scenario_episode(sc_name, sim, ho, mec, orch, steps, seed)
                
                # Tag Results
                df['setup'] = run_name
                df['agent_type'] = "Legacy" if "Legacy" in run_name else "PPO"
                
                all_results.append(df)
            
    # Combine
    final_df = pd.concat(all_results, ignore_index=True)
    out_path = os.path.join(RESULTS_DIR, "scenario_benchmark_results.csv")
    final_df.to_csv(out_path, index=False)
    print(f"\n>> Results saved to {out_path}")

if __name__ == "__main__":
    main()
