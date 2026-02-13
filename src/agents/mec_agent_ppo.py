from .ppo_core import BasePPOAgent
from .mec_agent_legacy import MECAgent
import numpy as np

class MECAgentPPO(BasePPOAgent, MECAgent):
    """
    PPO Version of MEC Agent.
    """
    def __init__(self, agent_id="mec_agent_ppo", config=None, num_actions=3, lr=3e-4, **kwargs):
        self.num_actions = num_actions
        self.obs_dim = 19 # Scale-invariant observation dimension
        
        BasePPOAgent.__init__(self, obs_dim=self.obs_dim, action_dim=num_actions,
                              lr=lr, gamma=0.99, batch_size=64)
        
        self.agent_id = agent_id
        self.config = config or {}
        
        # State tracking
        self.last_offload_target = "local"
        self.last_task_success = True
        self.last_task_time = 0.0
        self.task_counter = 0
        
        self.action_map = {0: "local", 1: "edge", 2: "cloud"}

    def select_action(self, observation: np.ndarray, context=None):
        # Physics Guardrail (Safety Override)
        if context is not None:
             # Check Throughput directly (captures both RSRP and SINR/Interference issues)
             throughput = context.get("serving_throughput_bps", 0.0)
             
             # If throughput is trash (< 1 Mbps), force LOCAL.
             # Saving 15 Mbits at 0.1 Mbps takes 150 seconds = HUGE energy.
             if throughput < 1e6: 
                 return 0 # Local
        
        action, _, _ = BasePPOAgent.select_action(self, observation)
        return action

    def select_action_with_info(self, observation: np.ndarray, context=None):
        """
        Select action and return policy information.
        Includes a safety guardrail for low Throughput conditions.
        """
        if context is not None:
             throughput = context.get("serving_throughput_bps", 0.0)
             # Safety Override: Force Local (0) if Throughput < 1 Mbps
             if throughput < 1e6:
                 return 0, 0.0, 0.0  
        
        return BasePPOAgent.select_action(self, observation)

    def get_metrics(self):
        return {
            "tasks_processed": self.task_counter,
            "buffer_size": len(self.obs_buffer)
        }

    def update(self, rollout_data, last_val=0.0):
        self.obs_buffer = rollout_data['obs']
        self.action_buffer = rollout_data['act']
        self.logprob_buffer = rollout_data['logprob']
        self.reward_buffer = rollout_data['rew']
        self.value_buffer = rollout_data['val']
        self.done_buffer = rollout_data['done']
        return self.finish_episode(last_val=last_val)
    
    def load(self, path):
         BasePPOAgent.load(self, path)
         print(f"MECAgentPPO loaded from {path}")
         
    def save(self, path):
         BasePPOAgent.save(self, path)
         print(f"MECAgentPPO saved to {path}")
