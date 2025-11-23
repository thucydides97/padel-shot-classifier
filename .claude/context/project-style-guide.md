---
created: 2025-11-22T21:32:40Z
last_updated: 2025-11-22T21:32:40Z
version: 1.0
author: Claude Code PM System
---

# Project Style Guide

## Code Style

### Python Conventions
- Follow PEP 8 guidelines
- Use snake_case for functions and variables
- Use PascalCase for classes
- 4-space indentation

### Naming Conventions
- **Files:** snake_case.py (e.g., pose_extractor.py)
- **Functions:** descriptive_verb_noun (e.g., calculate_features)
- **Variables:** descriptive names (e.g., wrist_velocity, elbow_angle)
- **Constants:** UPPER_SNAKE_CASE

### Module Structure
```python
"""Module docstring describing purpose."""

import standard_library
import third_party
import local_modules

CONSTANTS = ...

class ClassName:
    ...

def function_name():
    ...

if __name__ == "__main__":
    # CLI interface
```

## File Organization
- One class/major function per module
- Related utilities in same file
- Keep modules focused and cohesive

## Documentation

### Docstrings
- Use triple quotes for all public functions
- Include parameters, returns, and examples
- NumPy docstring style preferred

### Comments
- Explain "why" not "what"
- Use sparingly for complex logic
- Keep comments up to date

## Git Practices

### Commit Messages
- Imperative mood: "Add feature" not "Added feature"
- Short summary (<50 chars) + detailed body if needed
- Reference issues where applicable

### Branching
- main: stable release
- feature/*: new features
- fix/*: bug fixes

## Testing
- Run tests before committing
- Test edge cases (empty data, missing poses)
- Use descriptive test names

## Dependencies
- Add via Poetry: `poetry add package`
- Pin major versions
- Document why each dependency is needed
