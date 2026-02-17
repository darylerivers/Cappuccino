#!/usr/bin/env python3
"""
Code Janitor - Automatic code cleanup and quality checks

Integrates with Claude Code to maintain clean, high-quality code.
"""
import ast
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import argparse


class CodeJanitor:
    """Automated code cleanup and quality checker"""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.issues_found = []
        self.fixes_applied = []

    def log(self, message: str):
        """Log message if verbose"""
        if self.verbose:
            print(f"  {message}")

    def clean_file(self, file_path: Path) -> Dict:
        """Clean a single Python file"""
        if not file_path.suffix == '.py':
            return {"skipped": True, "reason": "Not a Python file"}

        self.log(f"Cleaning {file_path}")

        try:
            content = file_path.read_text()
            original_content = content

            # Apply cleanup rules
            content = self.remove_trailing_whitespace(content, file_path)
            content = self.fix_blank_lines(content, file_path)
            content = self.remove_debug_prints(content, file_path)
            content = self.fix_imports(content, file_path)
            content = self.remove_commented_code(content, file_path)

            # Check for issues
            self.check_security_issues(content, file_path)
            self.check_code_quality(content, file_path)

            # Write back if changed
            if content != original_content:
                file_path.write_text(content)
                self.log(f"✓ Updated {file_path}")
                return {"cleaned": True, "changes": len(self.fixes_applied)}
            else:
                return {"cleaned": False, "reason": "No changes needed"}

        except Exception as e:
            self.log(f"✗ Error cleaning {file_path}: {e}")
            return {"error": str(e)}

    def remove_trailing_whitespace(self, content: str, file_path: Path) -> str:
        """Remove trailing whitespace from lines"""
        lines = content.split('\n')
        cleaned_lines = []
        changed = False

        for i, line in enumerate(lines):
            if line != line.rstrip():
                changed = True
                self.fixes_applied.append(f"{file_path}:{i+1} - Removed trailing whitespace")
            cleaned_lines.append(line.rstrip())

        if changed:
            self.log("Fixed trailing whitespace")

        return '\n'.join(cleaned_lines)

    def fix_blank_lines(self, content: str, file_path: Path) -> str:
        """Fix excessive blank lines"""
        # Replace 3+ consecutive blank lines with 2
        original = content
        content = re.sub(r'\n\n\n+', '\n\n', content)

        if content != original:
            self.log("Fixed excessive blank lines")
            self.fixes_applied.append(f"{file_path} - Fixed excessive blank lines")

        return content

    def remove_debug_prints(self, content: str, file_path: Path) -> str:
        """Remove obvious debug print statements"""
        lines = content.split('\n')
        cleaned_lines = []
        removed_count = 0

        debug_patterns = [
            r'^\s*print\s*\(\s*["\']debug',
            r'^\s*print\s*\(\s*["\']DEBUG',
            r'^\s*print\s*\(\s*["\']test',
            r'^\s*print\s*\(\s*["\']TEST',
            r'^\s*print\s*\(\s*f?["\']---+',  # Separator prints
            r'^\s*print\s*\(\s*f?["\']===+',
        ]

        for i, line in enumerate(lines):
            is_debug = any(re.match(pattern, line, re.IGNORECASE) for pattern in debug_patterns)

            if is_debug:
                removed_count += 1
                self.fixes_applied.append(f"{file_path}:{i+1} - Removed debug print")
                self.log(f"Removed debug print: {line.strip()}")
            else:
                cleaned_lines.append(line)

        if removed_count > 0:
            self.log(f"Removed {removed_count} debug print statements")

        return '\n'.join(cleaned_lines)

    def fix_imports(self, content: str, file_path: Path) -> str:
        """Fix import statements"""
        try:
            # Parse to check for unused imports
            tree = ast.parse(content)

            # Find all imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        for alias in node.names:
                            imports.append(f"{node.module}.{alias.name}")

            # Check if autoflake is available
            try:
                result = subprocess.run(
                    ['autoflake', '--check', str(file_path)],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode != 0 and 'would be reformatted' in result.stdout:
                    self.issues_found.append(f"{file_path} - Has unused imports (run: autoflake --in-place --remove-unused-variables {file_path})")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        except SyntaxError:
            self.issues_found.append(f"{file_path} - Syntax error, cannot parse")

        return content

    def remove_commented_code(self, content: str, file_path: Path) -> str:
        """Remove blocks of commented-out code"""
        lines = content.split('\n')
        cleaned_lines = []
        in_comment_block = False
        comment_block_lines = []
        removed_blocks = 0

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Detect start of commented code block (3+ consecutive commented lines)
            if stripped.startswith('#') and not stripped.startswith('##'):
                comment_block_lines.append((i, line))

                # Check if this looks like code (has Python keywords or symbols)
                code_indicators = ['def ', 'class ', 'import ', 'from ', 'if ', 'for ', 'while ',
                                  'return ', '= ', '==', '!=', '+=', '-=']
                if any(indicator in stripped[1:] for indicator in code_indicators):
                    in_comment_block = True
            else:
                # End of comment block
                if in_comment_block and len(comment_block_lines) >= 3:
                    # This looks like commented-out code, skip it
                    removed_blocks += 1
                    self.log(f"Removed commented code block at lines {comment_block_lines[0][0]+1}-{comment_block_lines[-1][0]+1}")
                    self.fixes_applied.append(f"{file_path}:{comment_block_lines[0][0]+1}-{comment_block_lines[-1][0]+1} - Removed commented code block")
                else:
                    # Keep the comments (they're actual comments, not code)
                    for _, comment_line in comment_block_lines:
                        cleaned_lines.append(comment_line)

                comment_block_lines = []
                in_comment_block = False
                cleaned_lines.append(line)

        # Handle any remaining comment block at end of file
        if not in_comment_block or len(comment_block_lines) < 3:
            for _, comment_line in comment_block_lines:
                cleaned_lines.append(comment_line)

        if removed_blocks > 0:
            self.log(f"Removed {removed_blocks} blocks of commented code")

        return '\n'.join(cleaned_lines)

    def check_security_issues(self, content: str, file_path: Path):
        """Check for common security issues"""
        lines = content.split('\n')

        # Check for hardcoded secrets
        secret_patterns = [
            (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password"),
            (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key"),
            (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret"),
            (r'token\s*=\s*["\'][^"\']+["\']', "Hardcoded token"),
        ]

        for i, line in enumerate(lines):
            for pattern, issue in secret_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # Exclude if it's getting from environment
                    if 'os.environ' not in line and 'os.getenv' not in line:
                        self.issues_found.append(f"{file_path}:{i+1} - Security: {issue}")

        # Check for SQL injection risks
        if 'execute(' in content or 'executemany(' in content:
            if re.search(r'execute\s*\([^?]*%[sf]', content):
                self.issues_found.append(f"{file_path} - Security: Possible SQL injection (string formatting in execute)")

        # Check for command injection risks
        if 'subprocess.' in content or 'os.system' in content:
            if re.search(r'(subprocess\.|os\.system)\s*\([^)]*\+', content):
                self.issues_found.append(f"{file_path} - Security: Possible command injection (string concatenation)")

    def check_code_quality(self, content: str, file_path: Path):
        """Check general code quality issues"""
        lines = content.split('\n')

        # Check for overly long lines
        for i, line in enumerate(lines):
            if len(line) > 120 and not line.strip().startswith('#'):
                self.issues_found.append(f"{file_path}:{i+1} - Quality: Line too long ({len(line)} > 120 chars)")

        # Check for TODO/FIXME comments
        for i, line in enumerate(lines):
            if re.search(r'#\s*(TODO|FIXME|XXX|HACK)', line, re.IGNORECASE):
                self.issues_found.append(f"{file_path}:{i+1} - Quality: {line.strip()}")

        # Check for bare except
        for i, line in enumerate(lines):
            if re.match(r'^\s*except\s*:', line):
                self.issues_found.append(f"{file_path}:{i+1} - Quality: Bare except clause (catches all exceptions)")

    def run_formatters(self, file_path: Path) -> bool:
        """Run black and isort if available"""
        formatted = False

        # Try black
        try:
            result = subprocess.run(
                ['black', '--quiet', str(file_path)],
                capture_output=True,
                timeout=10
            )
            if result.returncode == 0:
                self.log("Applied black formatting")
                self.fixes_applied.append(f"{file_path} - Applied black formatting")
                formatted = True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Try isort
        try:
            result = subprocess.run(
                ['isort', '--quiet', str(file_path)],
                capture_output=True,
                timeout=10
            )
            if result.returncode == 0:
                self.log("Sorted imports with isort")
                self.fixes_applied.append(f"{file_path} - Sorted imports")
                formatted = True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return formatted


def main():
    parser = argparse.ArgumentParser(description="Code Janitor - Clean up messy code")
    parser.add_argument('paths', nargs='+', help='Files or directories to clean')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('--format', action='store_true', help='Run black/isort formatters')
    parser.add_argument('--check-only', action='store_true', help='Check only, do not modify files')

    args = parser.parse_args()

    janitor = CodeJanitor(verbose=args.verbose)

    # Collect all Python files
    files_to_clean = []
    for path_str in args.paths:
        path = Path(path_str)
        if path.is_file():
            files_to_clean.append(path)
        elif path.is_dir():
            files_to_clean.extend(path.rglob('*.py'))

    print(f"🧹 Code Janitor - Cleaning {len(files_to_clean)} files...")
    print()

    # Clean each file
    cleaned_count = 0
    error_count = 0

    for file_path in files_to_clean:
        if not args.check_only:
            result = janitor.clean_file(file_path)
            if result.get('cleaned'):
                cleaned_count += 1
            elif result.get('error'):
                error_count += 1

        # Run formatters if requested
        if args.format and not args.check_only:
            janitor.run_formatters(file_path)

    # Report results
    print()
    print("=" * 60)
    print("📊 Cleanup Report")
    print("=" * 60)

    if janitor.fixes_applied:
        print(f"\n✅ Applied {len(janitor.fixes_applied)} fixes:")
        for fix in janitor.fixes_applied[:20]:  # Show first 20
            print(f"  • {fix}")
        if len(janitor.fixes_applied) > 20:
            print(f"  ... and {len(janitor.fixes_applied) - 20} more")

    if janitor.issues_found:
        print(f"\n⚠️  Found {len(janitor.issues_found)} issues:")
        for issue in janitor.issues_found[:20]:  # Show first 20
            print(f"  • {issue}")
        if len(janitor.issues_found) > 20:
            print(f"  ... and {len(janitor.issues_found) - 20} more")

    if not janitor.fixes_applied and not janitor.issues_found:
        print("\n✨ All files are clean!")

    print()
    print(f"Files cleaned: {cleaned_count}")
    print(f"Errors: {error_count}")
    print()

    # Exit code
    if args.check_only and (janitor.issues_found or janitor.fixes_applied):
        sys.exit(1)  # Issues found
    elif error_count > 0:
        sys.exit(2)  # Errors occurred
    else:
        sys.exit(0)  # Success


if __name__ == '__main__':
    main()
