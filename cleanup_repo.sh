#!/bin/bash
# Repository Cleanup - Archives deprecated scripts and consolidates documentation
# SAFE: Archives to archive/ instead of deleting

# Note: Not using 'set -e' so script continues even if some files don't exist

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

DRY_RUN=false
if [[ "$1" == "--dry-run" ]]; then
    DRY_RUN=true
    echo -e "${YELLOW}DRY RUN MODE - No files will be moved${NC}\n"
fi

# Create archive directories
if [[ "$DRY_RUN" == false ]]; then
    mkdir -p archive/{deprecated_scripts,old_documentation,test_scripts,dashboard_variants,training_variants,analysis_scripts}
fi

moved_count=0

move_file() {
    local file=$1
    local dest=$2

    if [[ -f "$file" ]]; then
        if [[ "$DRY_RUN" == true ]]; then
            echo "  [WOULD MOVE] $file → archive/$dest/"
        else
            mv "$file" "archive/$dest/"
            echo -e "  ${GREEN}✓${NC} $file"
        fi
        ((moved_count++))
    fi
}

echo "=========================================="
echo "CAPPUCCINO REPOSITORY CLEANUP"
echo "=========================================="
echo ""

# ============================================================================
# 1. DEPRECATED TRAINING SCRIPTS (15 files)
# ============================================================================
echo -e "${YELLOW}1. Deprecated Training Scripts${NC}"
move_file "train_maxvram.py" "deprecated_scripts"
move_file "train_ensemble.py" "deprecated_scripts"
move_file "launch_max_training.py" "deprecated_scripts"
move_file "run_two_phase_training.py" "deprecated_scripts"
move_file "phase1_timeframe_optimizer.py" "deprecated_scripts"
move_file "phase2_feature_maximizer.py" "deprecated_scripts"
move_file "config_two_phase.py" "deprecated_scripts"
move_file "rerun_best_trial.py" "deprecated_scripts"
move_file "TRIAL_13_BEST_CONFIG.py" "deprecated_scripts"

# Training shell scripts
move_file "train_alpaca_model.sh" "deprecated_scripts"
move_file "train_alpaca_2workers.sh" "deprecated_scripts"
move_file "train_multi_timeframe.sh" "deprecated_scripts"
move_file "train_short_timeframe.sh" "deprecated_scripts"
move_file "launch_parallel.sh" "training_variants"
move_file "launch_parallel_no_prealloc.sh" "training_variants"
move_file "launch_phase2.sh" "training_variants"
move_file "launch_stable_max.sh" "training_variants"
move_file "launch_fundamental_test.sh" "training_variants"
move_file "start_training_72h.sh" "training_variants"
move_file "start_training_visible.sh" "training_variants"
move_file "start_pipeline_visible.sh" "training_variants"
move_file "restart_training.sh" "training_variants"
move_file "restart_fresh_training.sh" "training_variants"
move_file "start_fresh_weekly_training.sh" "training_variants"
move_file "start_fresh_weekly_training_incremental.sh" "training_variants"
move_file "start_high_performance_training.sh" "training_variants"
move_file "training_full.sh" "training_variants"
move_file "training_minimal.sh" "training_variants"
move_file "run_parallel_training.sh" "training_variants"

echo ""

# ============================================================================
# 2. TEST SCRIPTS (15 files)
# ============================================================================
echo -e "${YELLOW}2. Test Scripts${NC}"
move_file "test_arena_pruning.py" "test_scripts"
move_file "test_concentration_limit_fix.py" "test_scripts"
move_file "test_cpcv.py" "test_scripts"
move_file "test_dashboard_pages.py" "test_scripts"
move_file "test_fee_fixes.py" "test_scripts"
move_file "test_fee_tracking.py" "test_scripts"
move_file "test_fundamental_fixes.py" "test_scripts"
move_file "test_pipeline.py" "test_scripts"
move_file "test_rerun_model.py" "test_scripts"
move_file "test_trailing_stop_loss.py" "test_scripts"
move_file "stress_test_cge.py" "test_scripts"
move_file "stress_test_trial965.py" "test_scripts"
move_file "stress_test_trial965_results.csv" "test_scripts"
move_file "test_dashboard_page6.sh" "test_scripts"
move_file "test_ollama.sh" "test_scripts"

echo ""

# ============================================================================
# 3. DASHBOARD VARIANTS (6 files - keep only dashboard.py)
# ============================================================================
echo -e "${YELLOW}3. Dashboard Variants${NC}"
move_file "dashboard_backup_20251125_104511.py" "dashboard_variants"
move_file "dashboard_optimized.py" "dashboard_variants"
move_file "dashboard_unified.py" "dashboard_variants"
move_file "dashboard_ensemble_votes.py" "dashboard_variants"
move_file "dashboard_training_detailed.py" "dashboard_variants"
move_file "dashboard_snapshot.py" "dashboard_variants"
move_file "dashboard_paper_trading.py" "dashboard_variants"

echo ""

# ============================================================================
# 4. OLD PIPELINE VERSIONS (3 files - keep only pipeline_v2.py)
# ============================================================================
echo -e "${YELLOW}4. Old Pipeline Versions${NC}"
move_file "pipeline_orchestrator.py" "deprecated_scripts"
move_file "pipeline_viewer.py" "deprecated_scripts"
move_file "deploy_v2.py" "deprecated_scripts"

echo ""

# ============================================================================
# 5. DEBUG SCRIPTS (5 files)
# ============================================================================
echo -e "${YELLOW}5. Debug Scripts${NC}"
move_file "debug_monitor_training.sh" "deprecated_scripts"
move_file "debug_phase1_mini.sh" "deprecated_scripts"
move_file "debug_show_progress.sh" "deprecated_scripts"
move_file "demo_pipeline.sh" "deprecated_scripts"
move_file "run_validation.sh" "deprecated_scripts"

echo ""

# ============================================================================
# 6. ANALYSIS SCRIPTS (Keep only essential ones)
# ============================================================================
echo -e "${YELLOW}6. Duplicate Analysis Scripts${NC}"
move_file "analyze_training.py" "analysis_scripts"
move_file "analyze_test_results.py" "analysis_scripts"
move_file "analyze_arena_trades.py" "analysis_scripts"
move_file "compare_trading_performance.py" "analysis_scripts"

echo ""

# ============================================================================
# 7. UNRELATED UTILITIES (Job search, hardware guides)
# ============================================================================
echo -e "${YELLOW}7. Unrelated Utilities${NC}"
move_file "job_hunter.py" "deprecated_scripts"
move_file "job_apply_helper.py" "deprecated_scripts"
move_file "job_search_quickstart.sh" "deprecated_scripts"

echo ""

# ============================================================================
# 8. MIGRATION/SETUP SCRIPTS (Already completed)
# ============================================================================
echo -e "${YELLOW}8. Completed Migration Scripts${NC}"
move_file "migrate_old_trials.sh" "deprecated_scripts"
move_file "migrate_to_amd.sh" "deprecated_scripts"
move_file "preview_migration.sh" "deprecated_scripts"
move_file "apply_pytorch_optimizations.sh" "deprecated_scripts"

echo ""

# ============================================================================
# 9. DUPLICATE/OLD UTILITIES
# ============================================================================
echo -e "${YELLOW}9. Duplicate Utilities${NC}"
move_file "ultra_simple_ensemble.py" "deprecated_scripts"
move_file "create_simple_ensemble.py" "deprecated_scripts"
move_file "simple_portfolio_forecast.py" "deprecated_scripts"
move_file "monitor_concentration_fix.sh" "deprecated_scripts"
move_file "status_arena.sh" "deprecated_scripts"
move_file "optimize_gpu.sh" "deprecated_scripts"
move_file "optimize_memory.sh" "deprecated_scripts"

echo ""

# ============================================================================
# 10. MARKDOWN DOCUMENTATION (Consolidate 142 → ~20)
# ============================================================================
echo -e "${YELLOW}10. Consolidating Markdown Documentation (142 → 20)${NC}"

# Alert System (keep 1 of 3)
move_file "ALERT_SYSTEM_DEPLOYED.md" "old_documentation"
move_file "ALERT_SYSTEM_README.md" "old_documentation"
# KEEP: ALERT_SYSTEM_GUIDE.md

# Automation (keep 1 of multiple)
move_file "AUTOMATION_COMPLETE.md" "old_documentation"
move_file "AUTONOMOUS_OPERATION_FIXES_20260111.md" "old_documentation"
# KEEP: AUTOMATION_GUIDE.md (if exists, or keep best one)

# Dashboard (keep 1 of 5)
move_file "DASHBOARD_USAGE.md" "old_documentation"
move_file "DASHBOARD_NAVIGATION.md" "old_documentation"
move_file "DASHBOARD_ENHANCEMENTS.md" "old_documentation"
move_file "DASHBOARD_TRAINING_GUIDE.md" "old_documentation"
# KEEP: DASHBOARD_README.md

# Ensemble (keep 1 of 3)
move_file "ENSEMBLE_VOTING_DASHBOARD.md" "old_documentation"
# KEEP: ENSEMBLE_VOTING_GUIDE.md or ENSEMBLE_AUTO_SYNC_GUIDE.md

# Pipeline (keep 1 of 6)
move_file "PIPELINE_README.md" "old_documentation"
move_file "PIPELINE_STATUS.md" "old_documentation"
move_file "PIPELINE_TROUBLESHOOTING.md" "old_documentation"
move_file "PIPELINE_ANALYSIS.md" "old_documentation"
move_file "PIPELINE_IMPLEMENTATION_SUMMARY.md" "old_documentation"
# KEEP: PIPELINE_V2_DESIGN.md

# Training docs (keep 1 of 5+)
move_file "TRAINING_MONITOR.md" "old_documentation"
move_file "TRAINING_STATUS_REPORT.md" "old_documentation"
move_file "TRAINING_CONTROL_QUICK_REF.md" "old_documentation"
move_file "DUAL_TRAINING_STATUS.md" "old_documentation"
move_file "MAX_TRAINING_STATUS.md" "old_documentation"
# KEEP: TRAINING_CONTROL_README.md

# AI Training (keep 1 of 3)
move_file "AI_TRAINING_QUICKSTART.md" "old_documentation"
move_file "CGE_TRAINING_GUIDE.md" "old_documentation"
# KEEP: AI_TRAINING_GUIDE.md

# Quick Start (keep 1 of multiple)
move_file "QUICK_DIAGNOSTICS.md" "old_documentation"
move_file "QUICK_MONITOR.sh" "old_documentation"
# KEEP: QUICK_REFERENCE.md

# System docs (keep 1 of 5)
move_file "SYSTEM_CONFIGURATION.md" "old_documentation"
move_file "SYSTEM_REPORT_20251206.md" "old_documentation"
move_file "SYSTEM_STATUS_SUMMARY.md" "old_documentation"
move_file "SYSTEM_ANALYSIS_REPORT.md" "old_documentation"
# KEEP: SYSTEM_ARCHITECTURE.md

# Deployment docs (keep current)
move_file "DEPLOYMENT_LIMIT_IMPLEMENTED.md" "old_documentation"
# KEEP: DEPLOYMENT_READY.md

# Problem/bug reports (archive all)
move_file "BUG_REPORT_CONCENTRATION_LIMIT.md" "old_documentation"
move_file "CRITICAL_ISSUES_FOUND.md" "old_documentation"
move_file "ROOT_CAUSE_IDENTIFIED.md" "old_documentation"
move_file "OVERFITTING_PROBLEM.md" "old_documentation"
move_file "ALPHA_DECAY_SOLUTION.md" "old_documentation"

# Fee documentation (keep 1 of 4)
move_file "COMPLETE_FEE_FIXES.md" "old_documentation"
move_file "CRITICAL_FEE_ANALYSIS.md" "old_documentation"
move_file "FEE_TIERS_ANALYSIS.md" "old_documentation"
# KEEP: FEE_TRACKING_IMPLEMENTATION.md

# Concentration docs (keep 1 of 3)
move_file "CONCENTRATION_ALERT_TUNING.md" "old_documentation"
move_file "CONCENTRATION_FIX_SUMMARY.md" "old_documentation"
# KEEP: CONCENTRATION_LIMIT_FIX_APPLIED.md

# Status snapshots with dates
move_file "STATUS.md" "old_documentation"
move_file "STATUS_EFFICIENCY_REPORT_20260116.md" "old_documentation"
move_file "TRADING_PERFORMANCE_GUIDE.md" "old_documentation"

# Summaries/Reports
move_file "TODAY_SUMMARY.md" "old_documentation"
move_file "ENHANCEMENTS_SUMMARY.md" "old_documentation"
move_file "OPTIMIZATION_SUMMARY.md" "old_documentation"
move_file "COMPLETE_OPTIMIZATION_SUMMARY.md" "old_documentation"
move_file "FUNDAMENTAL_FIXES_SUMMARY.md" "old_documentation"
move_file "FIXES_APPLIED_20251207.md" "old_documentation"
move_file "IMPROVEMENTS_IMPLEMENTED_20260116.md" "old_documentation"
move_file "INSIGHTS_REPORT_20260116.md" "old_documentation"
move_file "PRAGMATIC_IMPROVEMENTS.md" "old_documentation"
move_file "CODE_OPTIMIZATIONS_APPLIED.md" "old_documentation"

# Analysis docs
move_file "DEEP_DIVE_ANALYSIS.md" "old_documentation"
move_file "TECHNICAL_ANALYSIS_REPORT.md" "old_documentation"
move_file "ARENA_ANALYSIS_20251215.md" "old_documentation"
move_file "ARENA_BENCHMARKS_AND_TIBURTINA.md" "old_documentation"
move_file "ARENA_VS_ENSEMBLE_EXPLAINED.md" "old_documentation"

# Implementation guides (dated)
move_file "STEP_2_INTEGRATION_SUMMARY.md" "old_documentation"
move_file "STEP_3_ROLLING_MEAN_GUIDE.md" "old_documentation"
move_file "TWO_PHASE_IMPLEMENTATION_COMPLETE.md" "old_documentation"
move_file "CACHING_IMPLEMENTATION_SUMMARY.md" "old_documentation"
move_file "MEMORY_OPTIMIZATION_SUMMARY.md" "old_documentation"

# Hardware/unrelated guides
move_file "STORAGE_BUYING_GUIDE.md" "old_documentation"
move_file "EBAY_HDD_BUYING_GUIDE.md" "old_documentation"
move_file "SMALL_CAPITAL_TRADING_NOTES.md" "old_documentation"

# GPU migration (keep consolidated in infrastructure/)
move_file "AMD_GPU_MIGRATION_GUIDE.md" "old_documentation"
move_file "AMD_GPU_TIMELINE_IMPACT.md" "old_documentation"

# Multi-timeframe docs
move_file "README_MULTI_TIMEFRAME.md" "old_documentation"
move_file "MULTI_TIMEFRAME_LAUNCH_GUIDE.md" "old_documentation"
move_file "SHORT_TIMEFRAME_GUIDE.md" "old_documentation"

# Two-phase training
move_file "README_TWO_PHASE_TRAINING.md" "old_documentation"
move_file "TWO_PHASE_AUTOMATION_GUIDE.md" "old_documentation"
move_file "TWO_PHASE_QUICK_REFERENCE.txt" "old_documentation"

# Data/deployment status
move_file "24H_FRESH_DATA_RESULTS_20260113.md" "old_documentation"
move_file "DATA_UPDATE_IN_PROGRESS_20260112.md" "old_documentation"
move_file "FRESH_DATA_DEPLOYED_20260112.md" "old_documentation"
move_file "LAUNCH_READINESS_ASSESSMENT_20260112.md" "old_documentation"
move_file "FRESH_START_PLAN.md" "old_documentation"

# Various guides
move_file "CRYPTO_METRICS_GUIDE.md" "old_documentation"
move_file "FULL_PROCESS_GUIDE.md" "old_documentation"
move_file "GRADING_AND_LIVE_TRADING_SYSTEM.md" "old_documentation"
move_file "GRADING_CRITERIA_UPDATE.md" "old_documentation"
move_file "HYBRID_TRADING_GUIDE.md" "old_documentation"
move_file "LIVE_TRADING_TIMELINE.md" "old_documentation"
move_file "MARKET_ANALYSIS_GUIDE.md" "old_documentation"
move_file "MEMORY_OPTIMIZATION_GUIDE.md" "old_documentation"
move_file "NEWS_API_INTEGRATION.md" "old_documentation"
move_file "PERFORMANCE_METRICS_FIX.md" "old_documentation"
move_file "PORTFOLIO_FORECAST_GUIDE.md" "old_documentation"
move_file "STOP_LOSS_DASHBOARD_UPDATE.md" "old_documentation"
move_file "TRADE_HISTORY_GUIDE.md" "old_documentation"
move_file "WEEKLY_TRAINING_GUIDE.md" "old_documentation"

# Tiburtina
move_file "TIBURTINA_CACHING_GUIDE.md" "old_documentation"

# Misc
move_file "CODE_JANITOR_SETUP.md" "old_documentation"
move_file "CORRECT_SYSTEM_DESIGN.md" "old_documentation"
move_file "CRITICAL_CONTEXT.md" "old_documentation"
move_file "DIAGNOSTIC_GUIDE.md" "old_documentation"
move_file "DIAGNOSTICS_README.md" "old_documentation"
move_file "PROCESS_FLOWCHART.txt" "old_documentation"
move_file "PROJECT_OVERVIEW_FOR_OPUS.md" "old_documentation"
move_file "WORKING_SETUP.md" "old_documentation"
move_file "TRADERS_RESTARTED.md" "old_documentation"
move_file "PAGE8_INSTANT_LOADING.md" "old_documentation"
move_file "fix_gpu_idle_pattern.md" "old_documentation"
move_file "advanced_optimizations.md" "old_documentation"

echo ""

# ============================================================================
# Summary
# ============================================================================
echo "=========================================="
if [[ "$DRY_RUN" == true ]]; then
    echo -e "${YELLOW}DRY RUN COMPLETE${NC}"
    echo "Would move: $moved_count files"
    echo ""
    echo "To actually perform cleanup:"
    echo "  ./cleanup_repo.sh"
else
    echo -e "${GREEN}CLEANUP COMPLETE!${NC}"
    echo "Archived: $moved_count files"
    echo ""
    echo "Archive structure:"
    echo "  archive/deprecated_scripts/    - Old Python/shell scripts"
    echo "  archive/old_documentation/     - Consolidated markdown"
    echo "  archive/test_scripts/          - Test files"
    echo "  archive/dashboard_variants/    - Old dashboard versions"
    echo "  archive/training_variants/     - Old training scripts"
    echo "  archive/analysis_scripts/      - Duplicate analysis tools"
    echo ""
    echo -e "${GREEN}Repository is now much cleaner!${NC}"
    echo ""
    echo "To restore a file:"
    echo "  mv archive/<category>/<filename> ."
fi
echo "=========================================="
