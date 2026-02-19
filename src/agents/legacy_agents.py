import numpy as np
from typing import Dict, Any, Tuple, Optional

class LegacyHOAgent:
    """
    Tier 1 Baseline: Legacy 5G Handover Strategy (A3 Event).
    Logic:
    - Greedy connection to strongest cell (RSRP).
    - Hysteresis: 2.0 dB (Standard static margin).
    - No RL, No Context Awareness beyond RSRP.
    """
    def __init__(self, num_cells=7):
        self.num_cells = num_cells
        self.agent_id = "legacy_ho_agent"
        # Context weights are irrelevant but needed for interface compatibility
        self.context_weights = {"alpha": 0.33, "beta": 0.33, "gamma": 0.34}

    def predict(self, observation: np.ndarray, state=None, episode_start=None, deterministic=True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        SB3 Compatible Interface.
        Note: The standard 'observation' vector (normalized) loses RSRP dBm context.
        However, since normalization is monotonic ((dbm+100)/50), we can validly compared normalized values.
        
        Normalized RSRP is at indices 0, 5, 10... (stride 5) assuming standard HO Agent obs.
        
        Logic:
        1. Extract RSRPs from observation.
        2. Identify Serving Cell (Observation has 'is_serving' flag at index 4, 9, 14...).
        3. Apply Hysteresis check.
        4. Return Action.
        """
        # We need to know which cell is which.
        # Obs layout per cell (5 features): [RSRP, SINR, Dist, Bearing, IsServing]
        # Total per cell = 5.
        
        best_cell = 0
        best_rsrp = -float('inf')
        serving_cell = 0
        serving_rsrp = -float('inf')
        
        # Parse Observation
        # Obs size = 5 * num_cells + 7 (global)
        # We only care about the first 5*num_cells
        
        for i in range(self.num_cells):
            base_idx = i * 5
            if base_idx + 4 >= len(observation): break
            
            rsrp_norm = observation[base_idx]
            is_serving = observation[base_idx + 4]
            
            # De-normalize for debug/clarity (optional, comparison works on norm too)
            # rsrp_dbm = rsrp_norm * 50.0 - 100.0
            
            if rsrp_norm > best_rsrp:
                best_rsrp = rsrp_norm
                best_cell = i
                
            if is_serving > 0.5: # It's 1.0 or 0.0
                serving_cell = i
                serving_rsrp = rsrp_norm
                
        # Hysteresis Logic
        # 2.0 dB in normalized scale?
        # Scale: 50 dB range maps to 1.0 (actually 2.0 range from -1 to 1? No, (val+100)/50)
        # If val=-100 -> 0.0. If val=-50 -> 1.0. range is 1.0 for 50dB.
        # So 2.0 dB = 2.0 / 50.0 = 0.04 in norm scale.
        
        hysteresis_norm = 2.0 / 50.0
        
        if best_rsrp > serving_rsrp + hysteresis_norm:
            action = best_cell
        else:
            action = serving_cell
            
        return np.array(action), None

    def select_action_with_info(self, observation: np.ndarray, context: Dict = None, training=False):
        """
        Compatible with our benchmark loop which expects (action, log_prob, value).
        We return dummies for log_prob and value.
        """
        # If we have context, use explicit dBm reasoning (more accurate than reversed norm)
        if context:
            rsrp_map = context.get('rsrp_dbm', [])
            serving_id = context.get('serving_cell_id', -1)
            
            if rsrp_map:
                best_id = int(np.argmax(rsrp_map))
                best_val = rsrp_map[best_id]
                
                curr_val = -140.0
                if 0 <= serving_id < len(rsrp_map):
                    curr_val = rsrp_map[serving_id]
                else:
                    # If serving is invalid/init, just jump to best
                    return best_id, None, None
                
                # Hysteresis 2.0 dB
                if best_val > curr_val + 2.0:
                    return best_id, None, None
                else:
                    return serving_id, None, None
        
        # Fallback to obs-based if no context
        action, _ = self.predict(observation)
        return int(action), None, None

    def get_observation(self, context):
        """
        Return dummy observation to satisfy pipeline calls.
        Benchmark script calls this before decision.
        But wait, our 'predict' logic relies on a valid observation structure!
        So we MUST produce a valid observation or else `predict` fails.
        
        Fortunately, we can reuse `HOAgentPPO.get_observation` logic or just rely on the existing 
        `ho_agent.get_observation` if we were inheriting.
        
        Simple Fix: Since this is a standalone class, we should replicate the minimal observation construction 
        OR just accept that `benchmark_scenarios.py` might try to call it.
        
        Actually, in my plan, I said I'd update `benchmark_scenarios.py` to instantiate `LegacyHOAgent`.
        Does it have `get_observation`? 
        Function `run_scenario_episode` calls `obs_ho = ho_agent.get_observation(context)`.
        So yes, I must implement it.
        """
        # Minimal extraction for A3 logic
        # We construct a mock observation that `predict` can parse (5 features per cell)
        
        rsrp_dbm = context.get("rsrp_dbm", [])
        serving_id = context.get("serving_cell_id", 0)
        
        features = []
        for i in range(self.num_cells):
            r = rsrp_dbm[i] if i < len(rsrp_dbm) else -140.0
            r_norm = (r + 100.0) / 50.0
            is_serv = 1.0 if i == serving_id else 0.0
            # [RSRP, SINR, Dist, Bearing, Serving]
            features.extend([r_norm, 0.0, 0.0, 0.0, is_serv])
            
        # Global padding
        features.extend([0.0]*7) 
        
        return np.array(features, dtype=np.float32)

class LegacyMECAgent:
    """
    Tier 1 Baseline: No MEC.
    Logic: Always Local (Action 0).
    """
    def __init__(self):
        self.agent_id = "legacy_mec_agent"
        self.action_map = {0: "local"}

    def predict(self, observation: np.ndarray, state=None, episode_start=None, deterministic=True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        return np.array(0), None

    def select_action(self, observation, context=None, training=False):
        return 0

    def select_action_with_info(self, observation, context=None, training=False):
        return 0, None, None

    def get_observation(self, context):
        return np.zeros(19, dtype=np.float32)
