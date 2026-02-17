---
description: Map and analyze module dependencies in the codebase
---

# Dependency Analysis Command

Use the architect agent to map and analyze dependencies in the codebase.

## Task

Analyze dependencies for: $ARGUMENTS

If no specific module provided, analyze the entire codebase.

### Analysis Steps

1. **Import Mapping**
   - List all imports for target module(s)
   - Identify external vs internal dependencies
   - Find transitive dependencies

2. **Dependency Graph**
   - Create ASCII visualization of dependencies
   - Identify hub modules (many dependents)
   - Find leaf modules (no dependents)

3. **Problem Detection**
   - Circular dependencies
   - Unnecessary dependencies
   - Missing abstractions (everything depends on one file)
   - Fragile dependencies (implementation details exposed)

4. **Refactoring Opportunities**
   - Modules that could be extracted
   - Dependencies that could be inverted
   - Interfaces that should be introduced

## Output

```
Target: [module name]

Direct Dependencies (imports):
  - module_a (internal)
  - module_b (internal)
  - pandas (external)

Dependents (imported by):
  - module_c
  - module_d

Dependency Graph:
  [ASCII diagram]

Issues Found:
  - [issue description]

Recommendations:
  - [specific refactoring suggestion]
```
