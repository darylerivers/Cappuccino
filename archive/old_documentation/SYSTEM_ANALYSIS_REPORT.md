# Cappuccino DRL Crypto Trading System - Technical Analysis Report

**Generated:** 2026-01-30  
**Analyst:** Claude (Sonnet 4.5)  
**System Version:** Cappuccino v0.1

---

## EXECUTIVE SUMMARY

The Cappuccino system is a Deep Reinforcement Learning (DRL) based cryptocurrency trading platform that uses Proximal Policy Optimization (PPO) agents trained via Optuna hyperparameter optimization. The system downloads historical crypto data, trains agents to maximize trading returns, and deploys them to paper trading via Alpaca's API.

**Critical Finding:** The "paper trading" component is currently a **simulation only** - it does not place actual orders with Alpaca. Actions are generated and portfolios updated in memory, but no `api.submit_order()` calls are made.

**Current Status:**
- ✅ Training: Active (Trial 0-4+, GPU optimized)
- ✅ Pipeline: Automated deployment working
- ⚠️  Paper Trading: Simulation running, but action scaling bug fixed
- ⚠️  Real Trading: Not implemented

---

## 1. SYSTEM ARCHITECTURE

### 1.1 High-Level Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    CAPPUCCINO TRADING SYSTEM                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │   DATA       │      │   TRAINING   │      │  DEPLOYMENT  │  │
│  │  INGESTION   │─────▶│   PIPELINE   │─────▶│   PIPELINE   │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│        │                      │                      │          │
│        │                      │                      │          │
│        ▼                      ▼                      ▼          │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │   Alpaca     │      │    Optuna    │      │    Paper     │  │
│  │   Market     │      │  + PPO Agent │      │   Trading    │  │
│  │     Data     │      │   Training   │      │  Simulation  │  │
│  └──────────────┘      └──────────────┘      └──────────────┘  │
│                              │                                  │
│                              ▼                                  │
│                        ┌──────────────┐                        │
│                        │  SQLite DBs  │                        │
│                        │  - Optuna    │                        │
│                        │  - Pipeline  │                        │
│                        └──────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Data Flow

```
1. Data Download (0_dl_trainval_data.py)
   └─▶ Alpaca API ─▶ Historical OHLCV ─▶ Technical Indicators ─▶ CSV/Parquet
   
2. Training (1_optimize_unified.py)
   └─▶ Load Data ─▶ Create Environment ─▶ Train PPO Agent ─▶ Evaluate ─▶ Store Best
   
3. Trial Management (pipeline_v2.py)
   └─▶ Monitor Optuna ─▶ Detect Completed ─▶ Deploy Best ─▶ Track Status
   
4. Paper Trading (paper_trader_alpaca_polling.py)
   └─▶ Load Model ─▶ Poll Alpaca ─▶ Generate Actions ─▶ Simulate Trades
```

---

## 2. CODE STRUCTURE ANALYSIS

### 2.1 Core Modules

#### **Training Module**
