---
description: Perform a comprehensive architecture review of the Cappuccino trading system
---

# Architecture Review Command

Use the architect agent to perform a comprehensive architecture review of the Cappuccino trading system.

## Task

Analyze the following aspects of the codebase:

1. **Directory Structure** - Map the overall organization
2. **Core Components** - Identify key modules and their responsibilities
3. **Dependencies** - Map inter-module dependencies and find circular deps
4. **Critical Paths** - Trace the training and trading execution flows
5. **Technical Debt** - Identify areas needing refactoring
6. **Risk Areas** - Find potential failure points

Focus on: $ARGUMENTS

If no specific focus provided, do a general architecture assessment.

## Output

Provide:
- Executive summary (2-3 sentences)
- Component health scorecard
- Top 5 issues ranked by severity
- Recommended next steps

Use the Task tool with subagent_type='Explore' for thorough codebase exploration.
