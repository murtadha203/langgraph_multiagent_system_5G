
from typing import Dict, Any, TypedDict, List
from langgraph.graph import StateGraph, END
from .schemas import ControlMode
from .tactical_frozen import TacticalExecutor
from .nodes.core_nodes import EstimatorNode, ShieldNode, ConfiguratorNode
from .nodes.strategist import StrategistNode

# Define State Schema for LangGraph
class OrchestratorState(TypedDict):
    step: int
    metrics: Dict[str, float]
    symbolic_state: Dict[str, Any]
    reasoning: str
    proposed_mode: str
    final_mode: str
    shield_active: bool
    applied_params: Dict[str, Any]
    history: List[Dict]
    current_mode: str

class StrategicOrchestrator:
    """
    The Strategic Orchestrator (Tier 2).
    
    Implemented through a Neuro-Symbolic Control Loop utilizing LangGraph:
    1. Symbolic Estimator: Maps high-level metrics to state representation.
    2. Strategist: Determines optimal control policy based on system state.
    3. Safety Shield: Enforces safety constraints on proposed modes.
    4. Configurator: Applies selected mode parameters to the execution layer.
    5. Tactical Execution: Executes actions within the simulation environment.
    """
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        # Initialize Orchestrator components
        
        # 1. Initialize Tactical Layer (Tier 1)
        # 2. Initialize Strategic Nodes (Tier 2)
        self.estimator_node = EstimatorNode()
        self.strategist_node = StrategistNode()
        self.shield_node = ShieldNode()
        if checkpoint_path is None:
             checkpoint_path = "models/ho_policy.pth"
        
        self.device = device
        # 1. Initialize Tactical Layer (Tier 1)
        self.tactical = TacticalExecutor(checkpoint_path, device)
        self.configurator_node = ConfiguratorNode(self.tactical)
        
        # 3. Build LangGraph Workflow
        self.graph = self._build_graph()
        
        # State Tracking
        self.state = {
            "step": 0,
            "current_mode": ControlMode.BALANCED.value,
            "applied_params": {},
            "history": []
        }
        
        # Puppeteer State (For external access)
        self.current_weights = {"alpha": 0.33, "beta": 0.33, "gamma": 0.34}
        
    def _build_graph(self) -> StateGraph:
        """Construct the LangGraph workflow for the strategic loop."""
        workflow = StateGraph(OrchestratorState)
        
        # Add Nodes
        workflow.add_node("estimator", lambda state: self.estimator_node.run(state))
        workflow.add_node("strategist", lambda state: self.strategist_node.run(state))
        workflow.add_node("shield", lambda state: self.shield_node.run(state))
        workflow.add_node("configurator", lambda state: self.configurator_node.run(state))
        
        # Define Edges (Linear flow: Est -> Strat -> Shield -> Config)
        workflow.set_entry_point("estimator")
        workflow.add_edge("estimator", "strategist")
        workflow.add_edge("strategist", "shield")
        workflow.add_edge("shield", "configurator")
        workflow.add_edge("configurator", END)
        
        return workflow.compile()
        
    def step(self, metrics: Dict[str, float], observation: Any, context: Dict[str, Any] = None, decision_interval: int = 100, force_run: bool = False, verbose: bool = False) -> int:
        """
        Execute one control step.
        Supports 'Puppeteer' mode where Tier 2 outputs weights for Tier 1.
        """
        self.state["step"] += 1
        
        # --- Strategic Loop (Tier 2) ---
        # Run if interval met OR forced by Panic Button
        if force_run or (self.state["step"] % decision_interval == 0):
            if verbose or force_run:
                trigger = "PANIC" if force_run else "TIMER"
                print(f"[Orchestrator] Running Strategic Cycle ({trigger}) at Step {self.state['step']}")
                
            if force_run:
                print(f"[ALERT] SYSTEM 2 INTERRUPT TRIGGERED: RSRP/Load Critical")
            
            # Prepare initial state for the graph
            graph_input = {
                "step": self.state["step"],
                "metrics": metrics,
                # Preserve existing context
                "current_mode": self.state.get("current_mode"),
                "history": self.state.get("history", [])
            }
            
            # Execute LangGraph
            final_state = self.graph.invoke(graph_input)
            
            # Update internal state with results
            self.state.update(final_state)
            
            # CRITICAL: Update State of Truth
            print(f"[Orchestrator] Step {self.state['step']}: Mode -> {self.state.get('final_mode')}")
            self.state["current_mode"] = self.state.get("final_mode", self.state["current_mode"])
            
            # Update Puppeteer Weights State for External Consumption
            params = self.state.get("applied_params", {})
            if "weights" in params:
                self.current_weights = params["weights"]
            
        # --- Tactical Action (Tier 1) ---
        if observation is None:
             return 0 
        
        # Pass Context for Reflex Guards
        # Also pass current_mode for Dynamic Hysteresis (Survival = Agile, Green = Strict)
        mode = self.state.get("current_mode", ControlMode.BALANCED.value)
        action = self.tactical.act(observation, context=context, mode=mode)
        
        return action

    def get_debug_info(self) -> Dict[str, Any]:
        """Return explainability signals."""
        return {
            "mode": self.state.get("final_mode"),
            "symbolic_state": self.state.get("symbolic_state"),
            "reasoning": self.state.get("reasoning"),
            "shield_active": self.state.get("shield_active")
        }
