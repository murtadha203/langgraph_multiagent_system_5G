
import torch
import numpy as np
from typing import Dict, Any, Optional
from ..agents.ho_agent_ppo import HOAgentPPO
from .schemas import ControlMode

class TacticalExecutor:
    """
    Tier 1: Tactical Execution Layer.
    
    Wraps the pre-trained PPO Agent in a frozen inference shell.
    Enables dynamic reconfiguration of control parameters (Hysteresis, TTT)
    by higher-level orchestrators based on the active ControlMode.
    """
    
    # Configuration profiles for different operational modes
    MODE_CONFIGS = {
        ControlMode.BALANCED: {
            "handover_margin_db": 3.0,
            "time_to_trigger_s": 0.16,
        },
        ControlMode.SURVIVAL: {
            "handover_margin_db": 2.0,
            "time_to_trigger_s": 0.0,
        },
        ControlMode.GREEN: {
            "handover_margin_db": 12.0,
            "time_to_trigger_s": 1.0,
        }
    }

    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        """
        Initialize the Frozen PPO Agent.
        
        Args:
            checkpoint_path: Path to the .pth weight file.
            device: 'cpu' or 'cuda'.
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        
        # Initialize Agent (Architecture must match training)
        self.agent = HOAgentPPO(
            agent_id="tactical_executor",
            num_cells=7,      # Renaissance Hex Topology
            frame_stack=1,    
            lr=0.0            # Learning rate 0 (Frozen)
        )
        
        # Load Weights
        self._load_weights()
        
        # Set to Eval Mode
        self.agent.network.eval()
        
        # Current State
        self.current_mode = ControlMode.BALANCED
        self.apply_mode(self.current_mode)
        
        # State Tracking
        self.last_ho_time = -10.0
        
        print(f"[TacticalExecutor] Online. Mode: {self.current_mode.value}. Status: Frozen.")

    def _load_weights(self):
        """Safely load the pre-trained weights."""
        try:
            # map_location ensures we can load CUDA models on CPU if needed
            self.agent.load(self.checkpoint_path)
            print(f"[TacticalExecutor] Weights loaded from {self.checkpoint_path}")
        except Exception as e:
            print(f"[TacticalExecutor] CRITICAL: Failed to load weights! {e}")
            raise e

    def apply_mode(self, mode: ControlMode, overrides: Optional[Dict[str, float]] = None):
        """
        Reconfigure the agent's parameters based on the Strategic Mode.
        
        Args:
            mode: The ControlMode (BALANCED, SURVIVAL, GREEN).
            overrides: Optional dict to fine-tune specific params (e.g., from LLM).
        """
        self.current_mode = mode
        
        # Start with base profile for the mode
        config = self.MODE_CONFIGS[mode].copy()
        
        # Apply intelligent overrides
        if overrides:
            config.update(overrides)
            
        # Apply to Agent
        self.agent.handover_margin_db = config.get("handover_margin_db", 3.0)
        self.agent.time_to_trigger_s = config.get("time_to_trigger_s", 0.16)
        
        # Apply Weights if provided in overrides or config
        weights = config.get("weights")
        if weights:
            self.agent.context_weights = weights
            # print(f"[Tactical] Applied weights: {weights}")
        
        # Log the adaptation
        # Parameters updated successfully

    def act(self, observation: np.ndarray, context: Optional[Dict[str, Any]] = None, mode: str = "BALANCED", deterministic: bool = True) -> int:
        """
        Execute Tactical Action.
        
        Args:
            observation: Raw observation from environment.
            context: Environment context (RSRP, etc.) for Safety Shield.
            mode: Current Strategic Mode (affects Safety Shield strictness).
            deterministic: If True, use argmax (Greedy).
        """
        # Observation processing for inference.
        
        if deterministic:
            # Manual Forward Pass for Greedy Action
            with torch.no_grad():
                if self.agent.normalize_obs:
                     norm_obs = self.agent.obs_rms.normalize(observation)
                else:
                     norm_obs = observation
                     
                obs_t = torch.FloatTensor(norm_obs).unsqueeze(0).to(self.device)
                logits = self.agent.network.actor(obs_t)
                action = torch.argmax(logits, dim=1).item()
        else:
            action = self.agent.select_action(observation)
            
        # --- DYNAMIC REFLEX GUARD (Muscle Safety) ---
        if context:
            action = self._reflex_guard(action, context, mode)
            
            # Update State if Handover Occurs
            serving_id = context.get('serving_cell_id', -1)
            if action != serving_id:
                self.last_ho_time = context.get('time_s', 0.0)
            
        return action

    def _reflex_guard(self, action: int, context: Dict[str, Any], mode: str) -> int:
        """
        Hard Physics Rules to prevent 'stupid' decisions.
        Dynamically adjusts 'Stupid' based on Strategic Intent.
        """
        # NO MEC LOGIC HERE (Handover Agent Only)

        # Standard Logic...
        serving_id = context.get('serving_cell_id', -1)
        if action == serving_id:
            return action # Staying is usually safe
            
        # Determine Hysteresis based on Mode
        hysteresis = 2.0 # Default Balanced
        if mode == "SURVIVAL":  hysteresis = 2.0 # Increased from 0.5 to prevent ping-pong
        elif mode == "GREEN":   hysteresis = 4.0
        
        # Rule 1: Hysteresis / Stability Check
        # Block HO if Target RSRP < Serving RSRP + Margin
        
        rsrp_map = context.get('rsrp_dbm', [])
        current_rsrp = context.get('serving_rsrp_dbm', -140.0)
        
        sinr_map = context.get('sinr_db', [])
        serving_id = context.get('serving_cell_id', -1)
        current_sinr = sinr_map[serving_id] if 0 <= serving_id < len(sinr_map) else -20.0
        current_time = context.get('time_s', 0.0)

        # --- Rule 0: CRITICAL RESCUE (Panic Handover) ---
        # Trigger if RSRP is dead (< -110) OR SINR is terrible (< -5 dB)
        if current_rsrp < -110.0 or current_sinr < -5.0:
             # Find best cell based on RSRP (or SINR?) - Baseline uses SINR for decision, RSRP for action?
             # Baseline: best_cell = argmax(sinr_list)
             if sinr_map:
                 best_cell = int(np.argmax(sinr_map))
                 best_metric = sinr_map[best_cell]
                 current_metric = current_sinr
                 threshold = 5.0 # Require +5dB improvement
             else:
                 best_cell = int(np.argmax(rsrp_map))
                 best_metric = rsrp_map[best_cell]
                 current_metric = current_rsrp
                 threshold = 3.0
                 
             # Only switch if better
             if best_metric > current_metric + threshold:
                 return best_cell


        # --- Rule 0.5: Time-Based Ping-Pong Guard ---
        # If not panic, block HO if too recent (< 1.0s)
        if current_time - self.last_ho_time < 1.0:
             return serving_id
                  
        # --- Rule 1: Stability Check (Prevent Ping-Pong) ---
        if 0 <= action < len(rsrp_map):
            target_rsrp = rsrp_map[action]
            
            # If Serving is Good (> -100), enforce strict margin
            if current_rsrp > -100.0:
                if target_rsrp < current_rsrp + hysteresis:
                    # BLOCK HO (Target not good enough)
                    return serving_id
            
            # If Serving is Bad (-100 to -115), allow any improvement
            elif current_rsrp > -115.0:
                if target_rsrp < current_rsrp:
                    # BLOCK HO (Don't jump to worse)
                    return serving_id
            
        return action
