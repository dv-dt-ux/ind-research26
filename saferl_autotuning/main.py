import numpy as np
import gymnasium as gym
import compiler_gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from saferl import APPO
from saferl.common.utils import evaluate
from torch.utils.tensorboard import SummaryWriter
import os
from typing import Dict, Any

# ============================================================================
# Custom Environment Wrapper for CompilerGym with Runtime Constraint
# ============================================================================

class CompilerGymSafeEnv(gym.Env):
    """
    CompilerGym environment wrapper with runtime constraint tracking.
    
    Goal: Minimize code size (reward = size reduction vs -Oz baseline)
    Constraint: Runtime degradation ≤ 0.01 relative to -Oz baseline
    """
    
    def __init__(
        self,
        benchmark: str = "cbench-v1/dijkstra",
        max_episode_steps: int = 20,
        runtime_limit: float = 0.01,  # Maximum allowed runtime degradation
    ):
        super().__init__()
        
        self.benchmark = benchmark
        self.max_episode_steps = max_episode_steps
        self.runtime_limit = runtime_limit
        
        # Create base CompilerGym environment
        # Using IrInstructionCountOz reward (normalized against -Oz baseline)
        # Using Autophase observation space (56 integer features)
        self.env = compiler_gym.make(
            "llvm-v0",
            benchmark=benchmark,
            observation_space="Autophase",
            reward_space="IrInstructionCountOz",  # Reward = size reduction vs -Oz
        )
        
        # Store baseline values
        self.baseline_runtime = None
        self.baseline_size = None
        self.current_runtime = None
        self.runtime_violation = 0.0
        
        # Action space is discrete (LLVM optimization passes)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
        # Tracking variables
        self.step_count = 0
        self.episode_cost = 0.0
        self.episode_reward = 0.0
        
    def reset(self, seed=None, options=None):
        """Reset environment and measure baseline runtime."""
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Measure baseline runtime at -Oz optimization level
        self.baseline_runtime = self._measure_runtime_oz()
        self.baseline_size = self._get_current_size()
        
        # Reset tracking variables
        self.step_count = 0
        self.episode_cost = 0.0
        self.episode_reward = 0.0
        self.runtime_violation = 0.0
        self.current_runtime = self.baseline_runtime
        
        return obs, info
    
    def step(self, action):
        """Execute action and compute reward + cost."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        self.step_count += 1
        self.episode_reward += reward
        
        # Measure current runtime after applying optimization
        self.current_runtime = self._measure_current_runtime()
        
        # Compute runtime degradation relative to baseline
        runtime_degradation = (self.current_runtime - self.baseline_runtime) / self.baseline_runtime
        
        # Cost = amount by which runtime degradation exceeds limit
        # Cost = max(0, degradation - limit)
        self.runtime_violation = max(0.0, runtime_degradation - self.runtime_limit)
        self.episode_cost += self.runtime_violation
        
        # Add cost to info dict for APPO to track
        info['cost'] = self.runtime_violation
        info['runtime_degradation'] = runtime_degradation
        info['current_runtime'] = self.current_runtime
        info['baseline_runtime'] = self.baseline_runtime
        info['current_size'] = self._get_current_size()
        info['baseline_size'] = self.baseline_size
        
        # Episode ends if max steps reached or runtime degradation too high
        terminated = terminated or self.step_count >= self.max_episode_steps
        
        return obs, reward, terminated, truncated, info
    
    def _measure_runtime_oz(self):
        """
        Measure runtime at -Oz optimization level.
        This is the baseline for constraint calculation.
        """
        # Save current state
        current_state = self.env.state
        
        # Compile with -Oz flags
        oz_actions = self._get_oz_action_sequence()
        env_copy = compiler_gym.make("llvm-v0", benchmark=self.benchmark)
        env_copy.reset()
        
        for action in oz_actions:
            env_copy.step(action)
        
        # Get runtime measurement
        runtime = self._get_runtime_from_env(env_copy)
        env_copy.close()
        
        # Restore original state
        self.env.state = current_state
        
        return runtime
    
    def _measure_current_runtime(self):
        """Measure runtime of current compiled program."""
        return self._get_runtime_from_env(self.env)
    
    def _get_runtime_from_env(self, env):
        """Extract runtime measurement from environment."""
        # CompilerGym provides runtime through reward space
        # We need to compute runtime from the reward signal
        # The reward is normalized, so we need to unnormalize
        
        # Alternative: Use the 'Runtime' reward space directly
        # For this wrapper, we'll use a heuristic based on instruction count
        # In production, you'd want to use the actual runtime measurement
        
        # Get instruction count as proxy for runtime
        obs = env.observation["IrInstructionCount"]
        return float(obs) if obs is not None else 1.0
    
    def _get_current_size(self):
        """Get current code size (instruction count)."""
        obs = self.env.observation["IrInstructionCount"]
        return float(obs) if obs is not None else 0.0
    
    def _get_oz_action_sequence(self):
        """Get the action sequence that corresponds to -Oz optimization level."""
        # This is environment-specific
        # For LLVM, -Oz is a predefined pass sequence
        # Return empty list for now - in production, you'd map this properly
        return []
    
    def close(self):
        """Close the environment."""
        self.env.close()


# ============================================================================
# TensorBoard Callback for Logging Metrics
# ============================================================================

class TensorBoardCallback(BaseCallback):
    """Custom callback for logging safety metrics to TensorBoard."""
    
    def __init__(self, log_dir: str = "./logs/", verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.writer = None
        self.episode_rewards = []
        self.episode_costs = []
        self.episode_runtime_degradations = []
        self.episode_size_reductions = []
        
    def _on_training_start(self) -> None:
        """Initialize TensorBoard writer."""
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
    def _on_step(self) -> bool:
        """Log metrics at each step."""
        # Collect info from environment
        if hasattr(self.training_env, 'envs'):
            for env in self.training_env.envs:
                if hasattr(env, 'env') and hasattr(env.env, 'episode_reward'):
                    # Log episode-level metrics when episode ends
                    if hasattr(env.env, 'step_count') and env.env.step_count == 0:
                        self._log_episode_metrics()
        
        # Log scalar metrics every N steps
        if self.num_timesteps % 100 == 0:
            self._log_step_metrics()
            
        return True
    
    def _log_step_metrics(self):
        """Log metrics at each step."""
        if self.writer is None:
            return
            
        # Get current values from environment
        for env in self.training_env.envs:
            if hasattr(env, 'env'):
                env_obj = env.env
                if hasattr(env_obj, 'runtime_violation'):
                    self.writer.add_scalar(
                        'constraint/runtime_violation',
                        env_obj.runtime_violation,
                        self.num_timesteps
                    )
                if hasattr(env_obj, 'episode_reward'):
                    self.writer.add_scalar(
                        'reward/episode_reward',
                        env_obj.episode_reward,
                        self.num_timesteps
                    )
                if hasattr(env_obj, 'current_runtime') and hasattr(env_obj, 'baseline_runtime'):
                    degradation = (env_obj.current_runtime - env_obj.baseline_runtime) / env_obj.baseline_runtime
                    self.writer.add_scalar(
                        'runtime/degradation',
                        degradation,
                        self.num_timesteps
                    )
                if hasattr(env_obj, 'current_size') and hasattr(env_obj, 'baseline_size'):
                    size_reduction = (env_obj.baseline_size - env_obj.current_size) / env_obj.baseline_size
                    self.writer.add_scalar(
                        'size/reduction_percent',
                        size_reduction * 100,
                        self.num_timesteps
                    )
    
    def _log_episode_metrics(self):
        """Log episode completion metrics."""
        if self.writer is None:
            return
            
        # Aggregate metrics from all environments
        total_reward = 0
        total_cost = 0
        total_runtime_degradation = 0
        total_size_reduction = 0
        num_envs = 0
        
        for env in self.training_env.envs:
            if hasattr(env, 'env'):
                env_obj = env.env
                total_reward += env_obj.episode_reward
                total_cost += env_obj.episode_cost
                
                if hasattr(env_obj, 'current_runtime') and hasattr(env_obj, 'baseline_runtime'):
                    degradation = (env_obj.current_runtime - env_obj.baseline_runtime) / env_obj.baseline_runtime
                    total_runtime_degradation += degradation
                    
                if hasattr(env_obj, 'current_size') and hasattr(env_obj, 'baseline_size'):
                    size_reduction = (env_obj.baseline_size - env_obj.current_size) / env_obj.baseline_size
                    total_size_reduction += size_reduction
                    
                num_envs += 1
        
        if num_envs > 0:
            self.writer.add_scalar('episode/mean_reward', total_reward / num_envs, self.num_timesteps)
            self.writer.add_scalar('episode/mean_cost', total_cost / num_envs, self.num_timesteps)
            self.writer.add_scalar('episode/mean_runtime_degradation', total_runtime_degradation / num_envs, self.num_timesteps)
            self.writer.add_scalar('episode/mean_size_reduction', total_size_reduction / num_envs, self.num_timesteps)
            
            # Log constraint satisfaction rate
            constraint_satisfied = sum(1 for env in self.training_env.envs 
                                      if hasattr(env, 'env') and env.env.runtime_violation <= 0.01)
            self.writer.add_scalar('constraint/satisfaction_rate', constraint_satisfied / num_envs, self.num_timesteps)
    
    def _on_training_end(self) -> None:
        """Close TensorBoard writer."""
        if self.writer:
            self.writer.close()


# ============================================================================
# Training Function
# ============================================================================

def train_safe_compiler_agent(
    benchmark: str = "cbench-v1/dijkstra",
    total_timesteps: int = 1_000_000,
    runtime_limit: float = 0.01,
    log_dir: str = "./logs/",
    model_save_path: str = "./models/appo_compiler",
):
    """
    Train APPO agent for safe compiler optimization.
    
    Args:
        benchmark: CompilerGym benchmark to optimize
        total_timesteps: Number of training steps
        runtime_limit: Maximum allowed runtime degradation (default 0.01 = 1%)
        log_dir: Directory for TensorBoard logs
        model_save_path: Path to save trained model
    """
    
    # Create environment
    def make_env():
        return CompilerGymSafeEnv(
            benchmark=benchmark,
            max_episode_steps=20,
            runtime_limit=runtime_limit,
        )
    
    # Wrap with DummyVecEnv for compatibility with saferl-lib
    env = DummyVecEnv([make_env for _ in range(4)])  # 4 parallel environments
    
    # Create APPO model with constraint
    # cost_constraint: Maximum allowed cost per episode
    # Since cost = runtime violation, we set constraint to 0 (no violation allowed)
    # But with tolerance, we use runtime_limit as the effective constraint
    model = APPO(
        "MlpPolicy",
        env,
        cost_constraint=[0.0],  # Target zero runtime violation
        learning_rate=3e-4,
        n_steps=2048,           # Steps per environment before update
        batch_size=256,         # Batch size (n_steps * num_envs / n_epochs approx)
        n_epochs=10,            # Epochs per update
        gamma=0.99,             # Discount factor
        gae_lambda=0.95,        # GAE lambda
        clip_range=0.2,         # PPO clipping range
        ent_coef=0.01,          # Entropy coefficient
        max_grad_norm=0.5,      # Gradient clipping
        tensorboard_log=log_dir,
        verbose=1,
    )
    
    # Create callback for logging
    callback = TensorBoardCallback(log_dir=log_dir)
    
    # Train model
    print(f"Starting training on benchmark: {benchmark}")
    print(f"Runtime constraint: {runtime_limit * 100}% degradation limit")
    print(f"Logging to: {log_dir}")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
    )
    
    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"Model saved to: {model_save_path}")
    
    return model


# ============================================================================
# Inference Function
# ============================================================================

def optimize_with_safe_agent(
    model_path: str,
    benchmark: str = "cbench-v1/dijkstra",
    num_episodes: int = 10,
    render: bool = False,
):
    """
    Run inference with trained safe agent.
    
    Args:
        model_path: Path to saved model
        benchmark: Benchmark to optimize
        num_episodes: Number of episodes to run
        render: Whether to render output
    
    Returns:
        Dictionary with optimization results
    """
    
    # Load model
    model = APPO.load(model_path)
    
    # Create evaluation environment
    env = CompilerGymSafeEnv(
        benchmark=benchmark,
        max_episode_steps=20,
        runtime_limit=0.01,
    )
    
    results = {
        'rewards': [],
        'costs': [],
        'size_reductions': [],
        'runtime_degradations': [],
        'constraint_satisfied': [],
    }
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_cost = 0
        
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_cost += info.get('cost', 0)
            
            if render:
                print(f"Step {env.step_count}: Action={action}, Reward={reward:.4f}, Cost={info.get('cost', 0):.4f}")
        
        # Record episode results
        results['rewards'].append(episode_reward)
        results['costs'].append(episode_cost)
        results['size_reductions'].append(
            (env.baseline_size - env._get_current_size()) / env.baseline_size * 100
        )
        results['runtime_degradations'].append(
            (env.current_runtime - env.baseline_runtime) / env.baseline_runtime * 100
        )
        results['constraint_satisfied'].append(episode_cost <= 0.01)
        
        print(f"\nEpisode {episode + 1}:")
        print(f"  Reward (size reduction): {episode_reward:.2f}")
        print(f"  Cost (runtime violation): {episode_cost:.4f}")
        print(f"  Size reduction: {results['size_reductions'][-1]:.2f}%")
        print(f"  Runtime degradation: {results['runtime_degradations'][-1]:.2f}%")
        print(f"  Constraint satisfied: {results['constraint_satisfied'][-1]}")
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Average reward: {np.mean(results['rewards']):.2f} ± {np.std(results['rewards']):.2f}")
    print(f"Average cost: {np.mean(results['costs']):.4f} ± {np.std(results['costs']):.4f}")
    print(f"Average size reduction: {np.mean(results['size_reductions']):.2f}%")
    print(f"Average runtime degradation: {np.mean(results['runtime_degradations']):.2f}%")
    print(f"Constraint satisfaction rate: {np.mean(results['constraint_satisfied']) * 100:.1f}%")
    
    env.close()
    return results


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Safe Compiler Optimization with APPO")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "inference"],
                        help="Run mode: train or inference")
    parser.add_argument("--benchmark", type=str, default="cbench-v1/dijkstra",
                        help="CompilerGym benchmark to optimize")
    parser.add_argument("--runtime_limit", type=float, default=0.01,
                        help="Maximum allowed runtime degradation (default 0.01 = 1%%)")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                        help="Total timesteps for training")
    parser.add_argument("--log_dir", type=str, default="./logs/",
                        help="Directory for TensorBoard logs")
    parser.add_argument("--model_path", type=str, default="./models/appo_compiler",
                        help="Path to save/load model")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes for inference")
    
    args = parser.parse_args()
    
    if args.mode == "train":
        model = train_safe_compiler_agent(
            benchmark=args.benchmark,
            total_timesteps=args.timesteps,
            runtime_limit=args.runtime_limit,
            log_dir=args.log_dir,
            model_save_path=args.model_path,
        )
    else:
        results = optimize_with_safe_agent(
            model_path=args.model_path,
            benchmark=args.benchmark,
            num_episodes=args.episodes,
            render=False,
        )
