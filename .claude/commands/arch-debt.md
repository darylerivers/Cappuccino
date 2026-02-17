---
description: Identify and prioritize technical debt in the codebase
---

# Technical Debt Analysis Command

Use the architect agent to identify and prioritize technical debt in the codebase.

## Task

Scan the codebase for:

1. **Code Smells**
   - Functions > 100 lines
   - Deep nesting (> 4 levels)
   - Duplicated code blocks
   - Magic numbers without constants

2. **Architecture Issues**
   - Circular dependencies
   - Tight coupling between modules
   - Missing abstractions
   - Inconsistent patterns

3. **Maintenance Burden**
   - Outdated comments
   - Dead code
   - TODO/FIXME comments
   - Missing error handling

4. **Security Concerns**
   - Hardcoded credentials
   - Unvalidated inputs
   - Unsafe file operations

Focus area: $ARGUMENTS

## Output

Provide a prioritized debt backlog:

| Priority | Type | Location | Effort | Impact | Description |
|----------|------|----------|--------|--------|-------------|
| P0 | ... | file:line | S/M/L | H/M/L | ... |

Include quick wins (high impact, low effort) at the top.
