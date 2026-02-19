"""
Obedience Training: Goal-Conditioned RL for Handover Agent.
Phase 1: Corners (Specialists) - Force distinct behaviors for pure Alpha/Beta/Gamma.
Phase 2: Mixed (Generalist) - Randomly sample weights from Dirichlet distribution.
"""

import os
import sys
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count
import random

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.simulation import NetworkSimulation, SERVICE_PROFILES
from src.agents.ho_agent_ppo import HOAgentPPO

# Configuration
MAX_EPISODES = 1000 
PHASE_1_STEPS = 500 # Corners
STEPS_PER_EPISODE = 50
BATCH_SIZE = 128
LR = 3e-4 # Standard PPO learning rate (not low, since training from scratch)

MODELS_DIR = os.path.join(project_root, "models")
DATA_DIR = os.path.join(project_root, "data")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def get_intent_for_phase(episode_idx):
    """
    Curriculum Strategy:
    Phase 1: Corners (0-5000)
    Phase 2: Mixed (5000+)
    """
    if episode_idx < PHASE_1_STEPS:
        # Corner Training: Cycle through pure modes
        mode = episode_idx % 3
        if mode == 0: return (1.0, 0.0, 0.0) # Pure Latency (Alpha)
        if mode == 1: return (0.0, 1.0, 0.0) # Pure Energy (Beta)
        return (0.0, 0.0, 1.0) # Pure Reliability/Throughput (Gamma)
    else:
        # Phase 2: Dirichlet-like Mixing (Generalization)
        # Sample random weights summing to 1.0
        r1 = random.random()
        r2 = random.random()
        # Normalize
        total = r1 + r2 + random.random()
        return (r1/total, r2/total, (total-r1-r2)/total)

def calculate_ho_reward(context, prev_context, intent_vector, info):
    """
    Reward function conditioned on Intent Vector (Alpha, Beta, Gamma).
    """
    w_lat, w_eng, w_thr = intent_vector # Alpha=Lat, Beta=Eng, Gamma=Thr
    
    # 1. Throughput/Reliability (Gamma)
    # RSRP -120 to -60 -> Norm 0.0 to 1.0
    rsrp = context['serving_rsrp_dbm']
    g_thr = np.clip((rsrp + 120.0) / 60.0, 0.0, 1.0)
    
    # 2. Latency/Stability (Alpha)
    is_ho = (context['serving_cell_id'] != prev_context['serving_cell_id'])
    # Latency Score: 1.0 if stable, PENALTY if RLF or HO
    # Check RLF Flag (passed in info)
    is_rlf = info.get('rlf_penalty', False)
    
    if is_rlf:
        g_lat = -1.0 # Huge penalty for RLF (Latency spike)
    elif is_ho:
        g_lat = 0.0 # Zero reward for HO (latency hit)
    else:
        g_lat = 1.0 # Good latency (stable)
    
    # 3. Energy (Beta)
    # Prefer low power (no HO).
    g_eng = 1.0 if not is_ho else 0.0
    
    # Composite Score
    # Include ALL weights!
    base_score = (w_thr * g_thr) + (w_lat * g_lat) + (w_eng * g_eng)
    
    # Penalty for HO (Explicit cost to discourage ping-pong)
    penalty = 0.0
    if is_ho:
        # Reduced from 0.2 to 0.05 to prevent "Zombie Agent" freezing
        # But scale by Latency sensitivity
        penalty = (0.2 * w_lat) + 0.05 
        
    return base_score - penalty

def run_obedience_episode(episode_idx, agent_state):
    seed = int(episode_idx + 907)
    random.seed(seed)
    np.random.seed(seed)
    
    intent = get_intent_for_phase(episode_idx)
    
    sim = NetworkSimulation(num_cells=7, seed=seed, dt_s=0.1)
    
    # Init Agent (Stateless for parallel run)
    agent = HOAgentPPO(agent_id="worker")
    if agent_state:
        agent.network.load_state_dict(agent_state)
    
    # Explicitly set intent in Sim so context picks it up correctly on reset?
    # No, sim.reset usually takes it.
    context = sim.reset(isd_range=(400, 800), intent_weights=intent) 
    
    rollout = {'obs':[], 'act':[], 'logprob':[], 'rew':[], 'val':[], 'done':[]}
    stats = {'rew': 0, 'ho': 0, 'intent': intent}
    
    for _ in range(STEPS_PER_EPISODE):
        prev_context = context
        
        # Inject Intent into Context so get_observation picks it up
        # Sim reset handles it, but maybe step updates?
        # Sim stores it in self.intent_weights
        
        obs = agent.get_observation(context)
        act, logp, val = agent.select_action_with_info(obs, context=context, training=True)
        
        next_context, info = sim.step(decision=int(act))
        
        # Pass INFO to reward function
        rew = calculate_ho_reward(next_context, prev_context, intent, info)
        
        rollout['obs'].append(obs)
        rollout['act'].append(act)
        rollout['logprob'].append(logp)
        rollout['rew'].append(rew)
        rollout['val'].append(val)
        rollout['done'].append(False)
        
        stats['rew'] += rew
        if next_context['serving_cell_id'] != prev_context['serving_cell_id']:
            stats['ho'] += 1
            
        context = next_context
        
    # Mark done
    if rollout['done']: rollout['done'][-1] = True
    
    return {'rollout': rollout, 'stats': stats}

def train():
    print(">>> Starting Obedience Training (Goal-Conditioned RL) <<<")
    
    # Set Global Seeds for Reproducibility
    seed = 907
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # 1. Initialize Fresh Agent (From Scratch)
    ho_agent = HOAgentPPO(agent_id="ho_learner", lr=LR)
    print(f"Agent Initialized. Observations: {ho_agent.obs_dim}")
    
    obs_buffer = {'obs':[], 'act':[], 'logprob':[], 'rew':[], 'val':[], 'done':[]}
    stats_history = []
    
    pbar = tqdm(total=MAX_EPISODES)
    current_ep = 0
    
    with Parallel(n_jobs=-1) as parallel:
        while current_ep < MAX_EPISODES:
            n_batch = cpu_count()
            # Dynamic batching for PPO stability? No, stick to CPU count for speed
            
            task_ids = range(current_ep, current_ep + n_batch)
            agent_state = {k: v.cpu() for k, v in ho_agent.network.state_dict().items()}
            
            results = parallel(
                delayed(run_obedience_episode)(idx, agent_state)
                for idx in task_ids
            )
            
            for res in results:
                for k in obs_buffer: obs_buffer[k].extend(res['rollout'][k])
                stats_history.append(res['stats'])
                
            # Update PPO
            if len(obs_buffer['obs']) >= BATCH_SIZE:
                ho_agent.network.train()
                ho_agent.update(obs_buffer)
                obs_buffer = {'obs':[], 'act':[], 'logprob':[], 'rew':[], 'val':[], 'done':[]}
                
            current_ep += n_batch
            pbar.update(n_batch)
            
            # Progress Log
            ls = stats_history[-1]
            phase = "CORNERS" if current_ep < PHASE_1_STEPS else "MIXED"
            pbar.set_description(f"[{phase}] HO={ls['ho']} | Rew={ls['rew']:.1f}")
            
            # Save Checkpoint
            if current_ep % 1000 == 0:
                ho_agent.save(os.path.join(MODELS_DIR, "ho_obedience_policy.pth"))
                
    # Final Save
    ho_agent.save(os.path.join(MODELS_DIR, "ho_obedience_policy.pth"))
    print("\n>>> Obedience Training Complete. Replaced 'ho_obedience_policy.pth'.")

if __name__ == "__main__":
    train()
