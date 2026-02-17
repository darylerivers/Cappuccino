#!/usr/bin/env python3
"""
Pipeline Component Tests
Quick tests to verify pipeline components are working.
"""

import sys
from pathlib import Path

print("=" * 70)
print("PIPELINE COMPONENT TESTS")
print("=" * 70)

# Test 1: Import modules
print("\n1. Testing module imports...")
try:
    from pipeline.state_manager import PipelineStateManager
    from pipeline.gates import BacktestGate, CGEStressGate, PaperTradingGate
    from pipeline.backtest_runner import BacktestRunner
    from pipeline.cge_runner import CGEStressRunner
    from pipeline.notifications import PipelineNotifier
    print("   ✓ All modules imported successfully")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: State manager
print("\n2. Testing state manager...")
try:
    state_mgr = PipelineStateManager("test_pipeline_state.json")
    state_mgr.add_trial(999, 0.123)
    trial = state_mgr.get_trial(999)
    assert trial is not None, "Trial not found"
    assert trial["value"] == 0.123, "Trial value mismatch"
    state_mgr.update_stage(999, "backtest", "passed", metrics={"sharpe": 0.5})
    trial = state_mgr.get_trial(999)
    assert trial["stages"]["backtest"]["status"] == "passed", "Stage not updated"
    # Cleanup
    Path("test_pipeline_state.json").unlink(missing_ok=True)
    print("   ✓ State manager working correctly")
except Exception as e:
    print(f"   ✗ State manager failed: {e}")
    sys.exit(1)

# Test 3: Configuration
print("\n3. Testing configuration loading...")
try:
    import json
    config_path = Path("config/pipeline_config.json")
    assert config_path.exists(), "Config file not found"
    
    with open(config_path) as f:
        config = json.load(f)
    
    assert "pipeline" in config, "Missing pipeline section"
    assert "gates" in config, "Missing gates section"
    assert "backtest" in config["gates"], "Missing backtest gate config"
    assert "cge_stress" in config["gates"], "Missing CGE gate config"
    print("   ✓ Configuration valid")
except Exception as e:
    print(f"   ✗ Configuration failed: {e}")
    sys.exit(1)

# Test 4: Gates
print("\n4. Testing validation gates...")
try:
    # Backtest gate
    backtest_gate = BacktestGate(config["gates"]["backtest"])
    
    # Should pass
    metrics = {
        "total_return": 0.05,
        "sharpe": 0.5,
        "max_drawdown": 0.10
    }
    passed, error = backtest_gate.validate(999, metrics)
    assert passed, f"Backtest gate failed unexpectedly: {error}"
    
    # Should fail
    metrics_bad = {
        "total_return": -0.60,
        "sharpe": -5.0,
        "max_drawdown": 0.90
    }
    passed, error = backtest_gate.validate(999, metrics_bad)
    assert not passed, "Backtest gate passed when it should fail"
    
    # CGE gate
    cge_gate = CGEStressGate(config["gates"]["cge_stress"])
    
    # Should pass
    cge_metrics = {
        "median_sharpe": 0.3,
        "profitable_pct": 0.50,
        "max_drawdown": 0.20,
        "catastrophic_failures": 0
    }
    passed, error = cge_gate.validate(999, cge_metrics)
    assert passed, f"CGE gate failed unexpectedly: {error}"
    
    # Should fail
    cge_metrics_bad = {
        "median_sharpe": -0.5,
        "profitable_pct": 0.20,
        "max_drawdown": 0.50,
        "catastrophic_failures": 5
    }
    passed, error = cge_gate.validate(999, cge_metrics_bad)
    assert not passed, "CGE gate passed when it should fail"
    
    print("   ✓ Gates working correctly")
except Exception as e:
    print(f"   ✗ Gates failed: {e}")
    sys.exit(1)

# Test 5: Directories
print("\n5. Testing directory structure...")
try:
    required_dirs = [
        "config",
        "pipeline",
        "deployments",
        "stress_test_results",
        "logs"
    ]
    
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        assert dir_path.exists(), f"Directory missing: {dir_name}"
    
    print("   ✓ All required directories exist")
except Exception as e:
    print(f"   ✗ Directory check failed: {e}")
    sys.exit(1)

# Test 6: Notifications
print("\n6. Testing notifications...")
try:
    notifier = PipelineNotifier(config["notifications"])
    # Test notification (will only log, not send desktop notification in test)
    notifier.gate_passed(999, "test_gate", {"test": True})
    
    # Check log file was created
    log_file = Path("logs/pipeline_notifications.log")
    assert log_file.exists(), "Notification log not created"
    
    print("   ✓ Notifications working correctly")
except Exception as e:
    print(f"   ✗ Notifications failed: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("ALL TESTS PASSED ✓")
print("=" * 70)
print("\nPipeline is ready to use!")
print("Start with: python pipeline_orchestrator.py --once")
print("=" * 70)
