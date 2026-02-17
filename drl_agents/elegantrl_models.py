# RL models from elegantrl
import os
import torch
import numpy as np
from train.config_fast import Arguments
from train.run import train_and_evaluate, train_and_evaluate_mp, init_agent

from drl_agents.agents import AgentDDPG, AgentPPO, AgentSAC, AgentTD3, AgentA2C

from drl_agents.agents import AgentPPO_FT

MODELS = {"ddpg": AgentDDPG, "td3": AgentTD3, "sac": AgentSAC, "ppo": AgentPPO, "ppo_ft": AgentPPO_FT, "a2c": AgentA2C}
OFF_POLICY_MODELS = ["ddpg", "td3", "sac"]
ON_POLICY_MODELS = ["ppo", "ppo_ft", "a2c"]
"""MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}"""


class DRLAgent:
    """Provides implementations for DRL algorithms
    Attributes
    ----------
        env: gym environment class
            user-defined class
    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get train_results
    """

    def __init__(self, env, price_array, tech_array, env_params, if_log,
                 sentiment_service=None, use_sentiment=False, tickers=None,
                 use_timeframe_constraint=False, timeframe=None, data_interval='1h'):
        self.env = env
        self.price_array = price_array
        self.tech_array = tech_array
        self.env_params = env_params
        self.if_log = if_log
        self.sentiment_service = sentiment_service
        self.use_sentiment = use_sentiment
        self.tickers = tickers
        self.use_timeframe_constraint = use_timeframe_constraint
        self.timeframe = timeframe
        self.data_interval = data_interval

    def get_model(self, model_name, gpu_id, model_kwargs):

        env_config = {
            "price_array": self.price_array,
            "tech_array": self.tech_array,
            "if_train": False,
        }

        # Check if using vectorized environment (has n_envs parameter)
        import inspect
        env_sig = inspect.signature(self.env.__init__)
        if 'n_envs' in env_sig.parameters:
            # Vectorized environment - pass n_envs from env_params
            n_envs = self.env_params.get('n_envs', 8)  # Default 8 parallel envs
            env = self.env(config=env_config,
                           env_params=self.env_params,
                           n_envs=n_envs,
                           if_log=self.if_log,
                           sentiment_service=self.sentiment_service,
                           use_sentiment=self.use_sentiment,
                           tickers=self.tickers,
                           use_timeframe_constraint=self.use_timeframe_constraint,
                           timeframe=self.timeframe,
                           data_interval=self.data_interval)
            env.env_num = n_envs  # CRITICAL: Set env_num for vectorized environments
        else:
            # Standard single environment
            env = self.env(config=env_config,
                           env_params=self.env_params,
                           if_log=self.if_log,
                           sentiment_service=self.sentiment_service,
                           use_sentiment=self.use_sentiment,
                           tickers=self.tickers,
                           use_timeframe_constraint=self.use_timeframe_constraint,
                           timeframe=self.timeframe,
                           data_interval=self.data_interval)
            env.env_num = 1

        # For GPU environment, free RAM after data moved to VRAM
        from environment_Alpaca_gpu import GPUBatchCryptoEnv
        if isinstance(env, GPUBatchCryptoEnv):
            # Data is now on GPU, free NumPy copies in RAM
            self.price_array = None
            self.tech_array = None
            import gc
            gc.collect()

        agent = MODELS[model_name]
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        model = Arguments(agent=agent, env=env)
        model.learner_gpus = gpu_id

        if model_name in OFF_POLICY_MODELS:
            model.if_off_policy = True
        else:
            model.if_off_policy = False

        if model_kwargs is not None:
            try:
                model.learning_rate = model_kwargs["learning_rate"]
                model.batch_size = model_kwargs["batch_size"]
                model.gamma = model_kwargs["gamma"]
                model.net_dim = model_kwargs["net_dimension"]
                model.target_step = model_kwargs["target_step"]
                model.eval_gap = model_kwargs["eval_time_gap"]
                cpu_total = os.cpu_count() or 4
                if "thread_num" in model_kwargs:
                    proposed_threads = int(model_kwargs["thread_num"])
                    model.thread_num = max(1, min(proposed_threads, cpu_total))
                if "worker_num" in model_kwargs:
                    proposed_workers = int(model_kwargs["worker_num"])
                    max_workers = max(1, cpu_total - 1)
                    model.worker_num = max(1, min(proposed_workers, max_workers))
                model.use_multiprocessing = bool(model_kwargs.get("use_multiprocessing", False))
            except BaseException:
                raise ValueError(
                    "Fail to read arguments, please check 'model_kwargs' input."
                )
        return model

    def train_model(self, model, cwd, total_timesteps=5000):
        model.cwd = cwd
        model.break_step = int(total_timesteps)

        # Keep PPO rollouts aligned with the available step budget so training
        # is allowed to iterate several update cycles. Without this clamp the
        # sampled `target_step` can dwarf `break_step`, causing the very first
        # rollout to exhaust the budget and the policy never learns.
        budget = max(int(total_timesteps), 1)
        env_max_step = getattr(model.env, "max_step", None)
        preferred = max(budget // 4, 1)
        if env_max_step:
            preferred = max(preferred, int(env_max_step))

        current_target = getattr(model, "target_step", preferred)
        if current_target is None or current_target <= 0:
            adjusted_target = preferred
        else:
            adjusted_target = min(int(current_target), budget)
            if adjusted_target > preferred:
                adjusted_target = preferred

        model.target_step = max(adjusted_target, 1)

        use_mp = bool(getattr(model, "use_multiprocessing", False))
        worker_num = int(getattr(model, "worker_num", 1))
        if use_mp and worker_num > 1:
            train_and_evaluate_mp(model)
        else:
            train_and_evaluate(model)

    @staticmethod
    def DRL_prediction(model_name, cwd, net_dimension, environment, gpu_id):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        agent = MODELS[model_name]
        environment.env_num = 1

        args = Arguments(agent=agent, env=environment)

        args.cwd = cwd
        args.net_dim = net_dimension
        # load agent
        try:
            agent = init_agent(args, gpu_id=gpu_id)
            act = agent.act
            device = agent.device
        except ValueError as e:
            # State dimension mismatch - re-raise to mark trial as PRUNED
            if "State dimension mismatch" in str(e):
                raise e
            else:
                raise ValueError(f"Fail to load agent: {str(e)}") from e
        except BaseException as e:
            raise ValueError(f"Fail to load agent: {str(e)}") from e

        # test on the testing env
        _torch = torch
        state = environment.reset()
        state = np.asarray(state, dtype=np.float32)
        episode_returns = list()  # the cumulative_return / initial_account
        episode_total_assets = list()
        episode_total_assets.append(environment.initial_total_asset)

        with _torch.no_grad():
            for i in range(environment.max_step):
                state_np = np.asarray(state, dtype=np.float32)
                if state_np.ndim == 1:
                    s_tensor = _torch.from_numpy(state_np).unsqueeze(0).to(device)
                else:
                    s_tensor = _torch.from_numpy(state_np).to(device)
                a_tensor = act(s_tensor)  # action_tanh = act.forward()
                action = (
                    a_tensor.detach().cpu().numpy()[0]
                )  # not need detach(), because with torch.no_grad() outside
                state, reward, done, _ = environment.step(action)
                state = np.asarray(state, dtype=np.float32)

                total_asset = (
                        environment.cash
                        + (
                                environment.price_array[environment.time] * environment.stocks
                        ).sum()
                )
                episode_total_assets.append(total_asset)
                episode_return = total_asset / environment.initial_total_asset
                episode_returns.append(episode_return)
                # Handle both scalar and tensor done values (vectorized envs return tensors)
                if isinstance(done, _torch.Tensor):
                    done_cpu = done.cpu() if done.is_cuda else done
                    done_val = done_cpu.any().item() if done_cpu.numel() > 1 else done_cpu.item()
                else:
                    done_val = done
                if done_val:
                    break
        print("\n Test Finished!")
        print("episode_return: ", episode_return - 1, '\n')
        return episode_total_assets
