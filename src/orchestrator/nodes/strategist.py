import os
from typing import Dict, Any
from ..schemas import ControlMode, ReliabilityState, TrafficState, EnergyState, MobilityState
from ..prompts import TELELLM_SYSTEM_PROMPT
from ..local_llm import LocalTeleLLMEngine

# --- CONFIGURATION ---
# Set to True for Real AI (Cloud Benchmarks)
# Set to False for Fast Local Dev (Mock Rules)
USE_REAL_LLM = False 

class StrategistNode:
    """
    Tier 2: Strategic Orchestrator (The Brain).
    Uses a Local Tele-LLM (Quen/Llama) to reason about Symbolic State 
    and output Continuous Control Weights (Puppeteer).
    """
    def __init__(self):
        self.llm = None
        
        # Initialize Local Tele-LLM ONLY if enabled
        if USE_REAL_LLM:
            try:
                # We use the class we defined in src/orchestrator/local_llm.py
                self.llm = LocalTeleLLMEngine()
                print("[Strategist] Local Tele-LLM (1.5B) loaded successfully.")
            except Exception as e:
                print(f"[Strategist] Failed to load Local LLM: {e}")
                print("[Strategist] Defaulting to Rule-Based Logic.")
        else:
            print("[Strategist] Running in MOCK Mode (Rule-Based). Set USE_REAL_LLM=True to enable AI.")

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the Strategist Logic.
        Generates a User Prompt -> Queries LLM -> Returns Weights & Mode.
        """
        sym_state = state.get("symbolic_state", {})
        
        # 1. Try LLM Inference (The Smart Way)
        if self.llm:
            try:
                # A. Generate the Situation Report (The "Eyes")
                user_prompt = self._construct_user_prompt(state)
                
                # B. Get Continuous Weights (The "Puppeteer")
                # Returns normalized alpha (Lat), beta (Eng), gamma (Rel)
                alpha, beta, gamma = self.llm.predict_weights(TELELLM_SYSTEM_PROMPT, user_prompt)
                
                # C. Determine Pseudo-Mode (For Logging/Plotting compatibility)
                # We map the continuous weights back to a named mode so your graphs still look good.
                proposed_mode = ControlMode.BALANCED
                reasoning = f"LLM Weights: Lat={alpha:.2f}, Eng={beta:.2f}, Rel={gamma:.2f}"
                
                if beta > 0.5: 
                    proposed_mode = ControlMode.GREEN
                elif gamma > 0.5: 
                    proposed_mode = ControlMode.SURVIVAL
                elif alpha > 0.5:
                    # If Latency is prioritized heavily, we treat it as Survival (active) or Balanced
                    proposed_mode = ControlMode.SURVIVAL 

                return {
                    "proposed_mode": proposed_mode,
                    "reasoning": reasoning,
                    # This is the Critical Control Signal for the Agent
                    "applied_params": {
                        "weights": {"alpha": alpha, "beta": beta, "gamma": gamma}
                    }
                }
                
            except Exception as e:
                print(f"[Strategist] LLM Error: {e}. Reverting to Rules.")

        # 2. Fallback: Rule-Based Expert System (The "Safety Net")
        return self._rule_based_logic(sym_state)

    def _construct_user_prompt(self, state: Dict[str, Any]) -> str:
        """
        Translates the raw metrics into a text description for the LLM.
        """
        metrics = state.get("metrics", {})
        sym_state = state.get("symbolic_state", {})
        
        # Extract Raw Data
        batt_val = metrics.get("ue_battery_joules", 500.0)
        # Assume 1000J is max for normalization context, or just use raw if model understands
        # Let's use the Symbolic State for clarity which the model understands better
        
        traffic_desc = sym_state.get("traffic", "NORMAL")
        energy_desc = sym_state.get("energy", "NORMAL")
        rel_desc = sym_state.get("reliability", "SAFE")
        
        # Add numeric context where available
        rsrp = metrics.get("serving_rsrp_dbm", -80)
        load = metrics.get("serving_cell_load", 0.5)
        
        prompt = f"""Current Network Status:
- Signal Strength: {rsrp:.1f} dBm ({rel_desc})
- Network Load: {load*100:.1f}% ({traffic_desc})
- Energy State: {energy_desc}
- Mobility: {sym_state.get('mobility', 'MODERATE')}

Task: Assign priority scores (0-10) for Latency, Energy, and Reliability based on this status."""
        return prompt

    def _rule_based_logic(self, sym_state: Dict) -> Dict:
        """Deterministic fallback logic (Returns standard modes)."""
        traffic = sym_state.get("traffic", TrafficState.NORMAL)
        reliability = sym_state.get("reliability", ReliabilityState.SAFE)
        energy = sym_state.get("energy", EnergyState.NORMAL)

        # 1. Reliability Critical
        if reliability == ReliabilityState.DANGER:
             return {
                 "proposed_mode": ControlMode.SURVIVAL, 
                 "reasoning": "Rule: Danger detected.",
                 "applied_params": {"weights": {"alpha": 0.1, "beta": 0.0, "gamma": 0.9}}
             }

        # 2. Energy Critical
        if energy == EnergyState.CRITICAL:
             return {
                 "proposed_mode": ControlMode.GREEN, 
                 "reasoning": "Rule: Battery critical.",
                 "applied_params": {"weights": {"alpha": 0.0, "beta": 1.0, "gamma": 0.0}}
             }

        # 3. Congestion
        if traffic == TrafficState.CONGESTED:
             return {
                 "proposed_mode": ControlMode.SURVIVAL, 
                 "reasoning": "Rule: Congestion.",
                 "applied_params": {"weights": {"alpha": 0.8, "beta": 0.0, "gamma": 0.2}}
             }
             
        # 4. High Mobility (New Rule for Day in Life)
        mobility = sym_state.get("mobility", MobilityState.MODERATE)
        if mobility == MobilityState.HIGH_VELOCITY:
             return {
                 "proposed_mode": ControlMode.SURVIVAL,
                 "reasoning": "Rule: High Mobility (Agile HO needed).",
                 "applied_params": {"weights": {"alpha": 0.2, "beta": 0.0, "gamma": 0.8}}
             }

        # Default Balanced
        return {
            "proposed_mode": ControlMode.BALANCED, 
            "reasoning": "Rule: Normal.",
            "applied_params": {"weights": {"alpha": 0.33, "beta": 0.33, "gamma": 0.34}}
        }