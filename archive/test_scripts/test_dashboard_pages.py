#!/usr/bin/env python3
"""Test all dashboard pages render without errors."""

import sys
import os

from dashboard import CappuccinoDashboard

def test_all_pages():
    """Test rendering all 4 pages."""
    dash = CappuccinoDashboard()

    pages = [
        (0, "Main Dashboard"),
        (1, "Ensemble Voting"),
        (2, "Portfolio History"),
        (3, "Training Statistics"),
    ]

    print("Testing Dashboard Pages:")
    print("=" * 50)

    # Redirect render output to null
    devnull = open(os.devnull, 'w')

    for page_num, page_name in pages:
        dash.current_page = page_num
        try:
            # Temporarily redirect stdout
            old_stdout = sys.stdout
            sys.stdout = devnull
            dash.render()
            sys.stdout = old_stdout
            print(f"✓ Page {page_num}: {page_name}")
        except Exception as e:
            sys.stdout = old_stdout
            print(f"✗ Page {page_num}: {page_name} - ERROR: {e}")
            devnull.close()
            return False

    devnull.close()

    print("=" * 50)
    print("✓ All pages rendered successfully!")
    return True

if __name__ == "__main__":
    success = test_all_pages()
    sys.exit(0 if success else 1)
