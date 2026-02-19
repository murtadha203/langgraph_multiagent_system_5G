
from typing import Dict, Any
from ..symbolic_estimator import SymbolicEstimator
from ..safety_shield import SafetyShield
from ..tactical_frozen import TacticalExecutor
from ..schemas import ControlMode

class EstimatorNode:
    """
    Wraps SymbolicEstimator to translate Raw Metrics -> Symbolic State.
    """
    def __init__(self):
        self.estimator = SymbolicEstimator()

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        metrics = state.get("metrics", {})
        sym_state = self.estimator.estimate(metrics)
        
        # Flatten enums to strings for easier LLM parsing/Logging if needed
        # But Strategist expects values (strings) or Enums? 
        # SymbolicEstimator returns strings (values) based on previous view.
        return {"symbolic_state": sym_state}

class ShieldNode:
    """
    Wraps SafetyShield to enforce transition constraints.
    """
    def __init__(self):
        self.shield = SafetyShield()

    def run(self, state: Dict[str, Any], verbose: bool = False) -> Dict[str, Any]:
        proposed_mode = state.get("proposed_mode")
        current_mode = state.get("current_mode")
        # Ensure current_mode is Enum
        if isinstance(current_mode, str):
            current_mode = ControlMode(current_mode)
            
        step = state.get("step", 0) # Use step as epoch proxy
        
        # Update shield's internal tracking (it tracks current mode internally too)
        # But safely, we pass the proposed transition.
        # The SafetyShield class we viewed had `propose_transition(proposed_mode, current_epoch)`
        # It also tracked `self.current_mode`. We should sync them.
        self.shield.current_mode = current_mode
        
        final_mode = self.shield.propose_transition(proposed_mode, step, verbose=verbose)
        
        return {"final_mode": final_mode.value, "shield_active": (final_mode != proposed_mode)}

class ConfiguratorNode:
    """
    Applies the Final Mode to the TacticalExecutor (Tactical Agent).
    """
    def __init__(self, tactical_executor: TacticalExecutor):
        self.tactical = tactical_executor

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        mode_val = state.get("final_mode")
        mode = ControlMode(mode_val)
        
        # Puppeteer Extraction: 
        # Check if Brain provided explicit weights (Score-based)
        weights = state.get("proposed_weights")
        
        if not weights:
            # Fallback to Mode-based defaults
            weights = {'alpha': 0.33, 'beta': 0.33, 'gamma': 0.34}
            if mode == ControlMode.GREEN:
                 weights = {'alpha': 0.05, 'beta': 0.90, 'gamma': 0.05}
            elif mode == ControlMode.SURVIVAL:
                 weights = {'alpha': 0.10, 'beta': 0.05, 'gamma': 0.85}

        # Apply configuration (Including Weights)
        # We pass the weights as an override to apply_mode
        self.tactical.apply_mode(mode, overrides={"weights": weights})
        
        # Puppeteer Extraction: 
        # In a real LLM, the Strategist would output JSON with continuous parameters.
        # Since we are using Rule-Based Fallback for now (Mode -> Enum),
        # we map the Mode to specific "Puppeteer Weights" here to simulate the LLM's intent.
        
        # Map Mode -> Continuous Weights
        # Balanced: 0.33, 0.33, 0.34
        # Green: 0.05, 0.90, 0.05 (Energy Focus)
        # Survival: 0.10, 0.05, 0.85 (Reliability Focus)
        
        # In the future, 'reasoning' from LLM might contain specific tweaks
        
        return {"applied_params": {
            "mode": mode.value,
            "weights": weights,
            "margin": self.tactical.agent.handover_margin_db,
            "ttt": self.tactical.agent.time_to_trigger_s
        }}
