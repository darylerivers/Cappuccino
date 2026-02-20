"""Train and evaluate a DRL agent on the provided train/test split."""

import time
import numpy as np
import gc
import torch
from pathlib import Path
from drl_agents.elegantrl_models import DRLAgent as DRLAgent_erl
from drl_agents.agents import AgentPPO, AgentPPO_FT
from utils.function_finance_metrics import (compute_data_points_per_year,
                                            compute_eqw,
                                            sharpe_iid)


def train_and_test(trial, price_array, tech_array, train_indices, test_indices, env, model_name, env_params, erl_params,
                   break_step, cwd, gpu_id, sentiment_service=None, use_sentiment=False, tickers=None,
                   use_timeframe_constraint=False, timeframe=None, data_interval='1h'):
    try:
        print(f"  [DEBUG] Starting TRAINING phase...")
        train_start = time.perf_counter()
        train_agent(price_array,
                    tech_array,
                    train_indices,
                    env, model_name,
                    env_params,
                    erl_params,
                    break_step,
                    cwd,
                    gpu_id,
                    sentiment_service=sentiment_service,
                    use_sentiment=use_sentiment,
                    tickers=tickers,
                    use_timeframe_constraint=use_timeframe_constraint,
                    timeframe=timeframe,
                    data_interval=data_interval)
        train_duration = time.perf_counter() - train_start

        print(f"  [DEBUG] Starting TESTING phase...")
        test_start = time.perf_counter()
        sharpe_bot, sharpe_eqw, drl_rets_tmp = test_agent(price_array,
                                                          tech_array,
                                                          test_indices,
                                                          env, env_params,
                                                          model_name,
                                                          cwd,
                                                          gpu_id,
                                                          erl_params,
                                                          trial,
                                                          sentiment_service=sentiment_service,
                                                          use_sentiment=use_sentiment,
                                                          tickers=tickers,
                                                          use_timeframe_constraint=use_timeframe_constraint,
                                                          timeframe=timeframe,
                                                          data_interval=data_interval)
        test_duration = time.perf_counter() - test_start

        return sharpe_bot, sharpe_eqw, drl_rets_tmp, train_duration, test_duration
    finally:
        # MEMORY LEAK FIX: Runs even if training/testing throws an exception
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()


def train_agent(price_array, tech_array, train_indices, env, model_name, env_params, erl_params, break_step, cwd,
                gpu_id, sentiment_service=None, use_sentiment=False, tickers=None,
                use_timeframe_constraint=False, timeframe=None, data_interval='1h'):
    print('No. Train Samples:', len(train_indices), '\n')
    price_array_train = price_array[train_indices, :]
    tech_array_train = tech_array[train_indices, :]

    # DEBUG: Calculate expected state_dim
    lookback = env_params.get('lookback')
    n_crypto = price_array_train.shape[1]
    n_tech = tech_array_train.shape[1]
    expected_state_dim = 1 + n_crypto + n_tech * lookback
    print(f"[DEBUG TRAIN] lookback={lookback}, n_crypto={n_crypto}, n_tech={n_tech}")
    print(f"[DEBUG TRAIN] Expected state_dim = 1 + {n_crypto} + {n_tech}*{lookback} = {expected_state_dim}")

    agent = DRLAgent_erl(env=env,
                         price_array=price_array_train,
                         tech_array=tech_array_train,
                         env_params=env_params,
                         if_log=True,
                         sentiment_service=sentiment_service,
                         use_sentiment=use_sentiment,
                         tickers=tickers,
                         use_timeframe_constraint=use_timeframe_constraint,
                         timeframe=timeframe,
                         data_interval=data_interval)

    # Check if using FT-Transformer
    use_ft_encoder = erl_params.get('use_ft_encoder', False)
    print(f"[DEBUG TRAIN] use_ft_encoder={use_ft_encoder}, model_name={model_name}")

    if use_ft_encoder and model_name == 'ppo':
        # Use FT-Transformer enhanced agent
        print(f"[DEBUG TRAIN] Using 'ppo_ft' model (AgentPPO_FT)")
        print(f"\n{'='*70}")
        print(f"Using FT-Transformer Feature Encoder")
        print(f"{'='*70}")
        print(f"  Pre-trained: {erl_params.get('pretrained_encoder_path') is not None}")
        print(f"  Freeze encoder: {erl_params.get('ft_freeze_encoder', False)}")
        print(f"  FT config: {erl_params.get('ft_config')}")
        print(f"{'='*70}\n")

        # Use 'ppo_ft' model name to get AgentPPO_FT from MODELS dictionary
        model = agent.get_model('ppo_ft',
                                gpu_id,
                                model_kwargs=erl_params)
    elif use_ft_encoder:
        print(f"[DEBUG TRAIN] WARNING: use_ft_encoder=True but model_name={model_name} (not ppo!)")

        # Add FT-Transformer parameters to model
        model.use_ft_encoder = True
        model.ft_config = erl_params.get('ft_config')
        model.pretrained_encoder_path = erl_params.get('pretrained_encoder_path')
        model.freeze_encoder = erl_params.get('ft_freeze_encoder', False)
    else:
        # Use standard baseline agent
        model = agent.get_model(model_name,
                                gpu_id,
                                model_kwargs=erl_params)

    agent.train_model(model=model,
                      cwd=cwd,
                      total_timesteps=break_step
                      )

    # MEMORY LEAK FIX: Delete training arrays and objects
    del price_array_train, tech_array_train, agent, model
    gc.collect()


def test_agent(price_array, tech_array, test_indices, env, env_params, model_name, cwd, gpu_id, erl_params, trial,
               sentiment_service=None, use_sentiment=False, tickers=None,
               use_timeframe_constraint=False, timeframe=None, data_interval='1h'):
    print('\nNo. Test Samples:', len(test_indices))
    price_array_test = price_array[test_indices, :]
    tech_array_test = tech_array[test_indices, :]

    data_config = {
        "price_array": price_array_test,
        "tech_array": tech_array_test,
        "if_train": False,
    }

    # Testing always uses single environment (not vectorized)
    # If env is vectorized, use the base CryptoEnvAlpaca class instead
    from environment_Alpaca import CryptoEnvAlpaca
    from environment_Alpaca_vectorized import VectorizedCryptoEnvAlpaca, VectorizedCryptoEnvAlpacaOptimized
    from environment_Alpaca_batch_vectorized import BatchVectorizedCryptoEnv
    from environment_Alpaca_gpu import GPUBatchCryptoEnv

    if env in (VectorizedCryptoEnvAlpaca, VectorizedCryptoEnvAlpacaOptimized, BatchVectorizedCryptoEnv, GPUBatchCryptoEnv):
        # Use base env for testing
        test_env_class = CryptoEnvAlpaca
    else:
        test_env_class = env

    # DEBUG: Calculate expected state_dim
    lookback = env_params.get('lookback')
    n_crypto = price_array_test.shape[1]
    n_tech = tech_array_test.shape[1]
    expected_state_dim = 1 + n_crypto + n_tech * lookback
    print(f"[DEBUG TEST] lookback={lookback}, n_crypto={n_crypto}, n_tech={n_tech}")
    print(f"[DEBUG TEST] Expected state_dim = 1 + {n_crypto} + {n_tech}*{lookback} = {expected_state_dim}")

    env_instance = test_env_class(config=data_config,
                                   env_params=env_params,
                                   if_log=True,
                                   sentiment_service=sentiment_service,
                                   use_sentiment=use_sentiment,
                                   tickers=tickers,
                                   use_timeframe_constraint=use_timeframe_constraint,
                                   timeframe=timeframe,
                                   data_interval=data_interval
                                   )
    print(f"[DEBUG TEST] ACTUAL state_dim from env: {env_instance.state_dim}")

    net_dimension = erl_params['net_dimension']

    # Check if using FT-Transformer (must match training!)
    use_ft_encoder = erl_params.get('use_ft_encoder', False)
    if use_ft_encoder and model_name == 'ppo':
        # Use 'ppo_ft' model for testing (must match training!)
        test_model_name = 'ppo_ft'
        account_value_erl = DRLAgent_erl.DRL_prediction(
            model_name=test_model_name,
            cwd=cwd,
            net_dimension=net_dimension,
            environment=env_instance,
            gpu_id=gpu_id
        )
    else:
        # Use standard baseline agent
        account_value_erl = DRLAgent_erl.DRL_prediction(
            model_name=model_name,
            cwd=cwd,
            net_dimension=net_dimension,
            environment=env_instance,
            gpu_id=gpu_id
        )
    lookback = env_params['lookback']
    indice_start = lookback - 1
    indice_end = len(price_array_test) - lookback

    data_points_per_year = compute_data_points_per_year(data_interval)
    account_value_eqw, eqw_rets_tmp, eqw_cumrets = compute_eqw(price_array_test, indice_start, indice_end)
    dataset_size = np.shape(eqw_rets_tmp)[0]
    factor = data_points_per_year / dataset_size
    sharpe_eqw, _ = sharpe_iid(eqw_rets_tmp, bench=0, factor=factor, log=False)

    account_value_erl = np.array(account_value_erl)
    base_account = np.maximum(account_value_erl[:-1], 1e-12)
    drl_rets_tmp = account_value_erl[1:] / base_account - 1
    sharpe_bot, _ = sharpe_iid(drl_rets_tmp, bench=0, factor=factor, log=False)

    # MEMORY LEAK FIX: Delete test arrays and environment
    del price_array_test, tech_array_test, env_instance, account_value_erl, account_value_eqw
    del eqw_rets_tmp, eqw_cumrets, base_account
    gc.collect()

    return sharpe_bot, sharpe_eqw, drl_rets_tmp
