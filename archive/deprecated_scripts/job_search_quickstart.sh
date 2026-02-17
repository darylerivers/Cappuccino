#!/bin/bash
# Quick Start Script for Job Search System

echo "════════════════════════════════════════════════════════════"
echo "   🎯 Credit Risk Analysis Job Search System"
echo "════════════════════════════════════════════════════════════"
echo ""

# Check if first run
if [ ! -f "job_applications.db" ]; then
    echo "🆕 First time setup detected!"
    echo ""
    echo "This system will help you:"
    echo "  ✅ Find Credit Risk Analysis remote jobs"
    echo "  ✅ Rank opportunities by match quality"
    echo "  ✅ Generate tailored cover letters"
    echo "  ✅ Track your applications"
    echo ""
    echo "📖 Full guide: JOB_SEARCH_GUIDE.md"
    echo ""
    read -p "Press Enter to start your first job search..."
fi

echo ""
echo "🔍 Searching for Credit Risk Analysis remote positions..."
echo "────────────────────────────────────────────────────────────"
python job_hunter.py search

echo ""
echo "────────────────────────────────────────────────────────────"
echo "📊 Your Job Search Status:"
echo "────────────────────────────────────────────────────────────"
python job_hunter.py status

echo ""
echo "────────────────────────────────────────────────────────────"
echo "🎯 Top Opportunities (70+ match score):"
echo "────────────────────────────────────────────────────────────"
python job_hunter.py list --min-score 70

echo ""
echo "════════════════════════════════════════════════════════════"
echo "   Next Steps"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "To prepare application materials for a job:"
echo "  python job_apply_helper.py prepare <job_id>"
echo ""
echo "To prepare for ALL high-scoring jobs:"
echo "  python job_apply_helper.py bulk-prepare"
echo ""
echo "To track an application:"
echo "  python job_hunter.py track <job_id> --status applied"
echo ""
echo "Run this script daily to find new opportunities!"
echo ""
