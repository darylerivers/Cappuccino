"""Train and evaluate a DRL agent on the provided train/test split."""

import time
import numpy as np
from drl_agents.elegantrl_models import DRLAgent as DRLAgent_erl
from function_finance_metrics import (compute_data_points_per_year,
                                      compute_eqw,
                                      sharpe_iid)


def train_and_test(trial, price_array, tech_array, train_indices, test_indices, env, model_name, env_params, erl_params,
                   break_step, cwd, gpu_id, sentiment_service=None, use_sentiment=False, tickers=None):
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
                tickers=tickers)
    train_duration = time.perf_counter() - train_start

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
                                                      tickers=tickers)
    test_duration = time.perf_counter() - test_start
    return sharpe_bot, sharpe_eqw, drl_rets_tmp, train_duration, test_duration


def train_agent(price_array, tech_array, train_indices, env, model_name, env_params, erl_params, break_step, cwd,
                gpu_id, sentiment_service=None, use_sentiment=False, tickers=None):
    print('No. Train Samples:', len(train_indices), '\n')
    price_array_train = price_array[train_indices, :]
    tech_array_train = tech_array[train_indices, :]

    agent = DRLAgent_erl(env=env,
                         price_array=price_array_train,
                         tech_array=tech_array_train,
                         env_params=env_params,
                         if_log=True,
                         sentiment_service=sentiment_service,
                         use_sentiment=use_sentiment,
                         tickers=tickers)

    model = agent.get_model(model_name,
                            gpu_id,
                            model_kwargs=erl_params,
                            )

    agent.train_model(model=model,
                      cwd=cwd,
                      total_timesteps=break_step
                      )


def test_agent(price_array, tech_array, test_indices, env, env_params, model_name, cwd, gpu_id, erl_params, trial,
               sentiment_service=None, use_sentiment=False, tickers=None):
    print('\nNo. Test Samples:', len(test_indices))
    price_array_test = price_array[test_indices, :]
    tech_array_test = tech_array[test_indices, :]

    data_config = {
        "price_array": price_array_test,
        "tech_array": tech_array_test,
        "if_train": False,
    }

    env_instance = env(config=data_config,
                       env_params=env_params,
                       if_log=True,
                       sentiment_service=sentiment_service,
                       use_sentiment=use_sentiment,
                       tickers=tickers
                       )

    net_dimension = erl_params['net_dimension']

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

    data_points_per_year = compute_data_points_per_year(trial.user_attrs["timeframe"])
    account_value_eqw, eqw_rets_tmp, eqw_cumrets = compute_eqw(price_array_test, indice_start, indice_end)
    dataset_size = np.shape(eqw_rets_tmp)[0]
    factor = data_points_per_year / dataset_size
    sharpe_eqw, _ = sharpe_iid(eqw_rets_tmp, bench=0, factor=factor, log=False)

    account_value_erl = np.array(account_value_erl)
    base_account = np.maximum(account_value_erl[:-1], 1e-12)
    drl_rets_tmp = account_value_erl[1:] / base_account - 1
    sharpe_bot, _ = sharpe_iid(drl_rets_tmp, bench=0, factor=factor, log=False)

    return sharpe_bot, sharpe_eqw, drl_rets_tmp
