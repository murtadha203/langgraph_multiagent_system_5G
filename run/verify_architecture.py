
import os
import sys
import numpy as np

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.simulation import NetworkSimulation, SERVICE_PROFILES
from src.agents.mec_agent_ppo import MECAgentPPO
from src.agents.ho_agent_ppo import HOAgentPPO
from src.orchestrator.orchestrator import StrategicOrchestrator
from src.orchestrator.schemas import ControlMode

def test_architecture():
    print("Testing Architecture Implementation...")
    
    # 1. Setup Sim
    sim = NetworkSimulation(seed=42)
    sim.reset()
    
    # 2. Setup Orchestrator (Brain) with Context passing
    # Mock checkpoint path (it will use fallback/random if not found, usually models/ho_policy.pth exists)
    orch = StrategicOrchestrator(checkpoint_path="models/ho_obedience_policy.pth")
    
    # 3. Setup Agents (Muscle)
    ho_agent = orch.tactical.agent # Use the one inside the Muscle
    mec_agent = MECAgentPPO() # Dummy
    
    # 4. Run Loop
    metrics = {
        'avg_load': 0.5,
        'arrival_rate': 10.0,
        'rsrp': -90.0,
        'rlf_rate': 0.0, 
        'avg_latency_ms': 20.0,
        'ue_battery_percent': 80.0,
        'avg_velocity_kmh': 30.0
    }
    
    context = sim.get_context()
    
    print("\n--- Step 1: Normal Operation ---")
    # Get Observation
    obs = ho_agent.get_observation(context)
    
    # Execute Step (Brain + Muscle)
    # Force run Brain to generate weights
    action = orch.step(metrics, obs, context=context, force_run=True)
    print(f"Action: {action}")
    print(f"Brain Weights: {orch.current_weights}")
    
    # Verify Reflex Guard Logic (Mock Context)
    print("\n--- Step 2: Reflex Guard Test ---")
    # Set Serve RSRP to -105 (Good enough)
    # Set Neighbor RSRP to -106 (Worse)
    # Muscle (Agent) might want to switch if random?
    # Let's force a dangerous context
    
    context['serving_rsrp_dbm'] = -100.0
    # Make all neighbors terrible
    context['rsrp_dbm'] = [-120.0] * 7
    context['rsrp_dbm'][sim.serving_cell_id] = -100.0
    
    # Force agent to try a handover (Hard to force PPO without specific obs, but we can call _reflex_guard directly)
    
    print("Directly testing _reflex_guard...")
    # Serving is 0 (usually)
    serving_id = context['serving_cell_id']
    target_id = (serving_id + 1) % 7
    
    # Case 1: Target is Bad (-120 vs -100). Should block.
    blocked_action = orch.tactical._reflex_guard(target_id, context)
    print(f"Target (-120) vs Serving (-100). Proposed: {target_id}, Final: {blocked_action}")
    if blocked_action == serving_id:
        print("PASS: Reflex Guard blocked bad handover.")
    else:
        print("FAIL: Reflex Guard allowed bad handover.")
        
    # Case 2: Target is Good (-90 vs -100). Should allow.
    context['rsrp_dbm'][target_id] = -90.0
    allowed_action = orch.tactical._reflex_guard(target_id, context)
    print(f"Target (-90) vs Serving (-100). Proposed: {target_id}, Final: {allowed_action}")
    if allowed_action == target_id:
        print("PASS: Reflex Guard allowed good handover.")
    else:
        print(f"FAIL: Reflex Guard blocked good handover (Got {allowed_action}).")

if __name__ == "__main__":
    test_architecture()
