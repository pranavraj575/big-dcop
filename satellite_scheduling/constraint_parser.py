"""Constraint function expression parser and evaluator for COSPSolver.

This module provides safe evaluation of constraint utility functions
from pydcop problem definitions, enabling dynamic constraint modification
(e.g., penalty functions in iterative pricing).
"""

import logging
import re
from typing import Dict, Union, Any

logger = logging.getLogger(__name__)


class ConstraintFunctionEvaluator:
    """Safely evaluates constraint utility function expressions.
    
    Supports pydcop constraint functions like:
    - "1 / (sum_var ** 2)"
    - "-penalty * variable_name"
    - "constraint_func(x, y)"
    
    Uses restricted eval() with limited scope for safety.
    """

    # Allowed operators and functions in expressions
    ALLOWED_BUILTINS = {
        'abs': abs,
        'min': min,
        'max': max,
        'sum': sum,
        'pow': pow,
        'round': round,
        'len': len,
    }

    # Regular expression to validate variable names
    VARIABLE_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')

    def __init__(self):
        """Initialize the constraint function evaluator."""
        self.cache = {}  # Cache compiled expressions
        self.error_count = 0

    def evaluate(self, function_str: str, variable_values: Dict[str, Union[int, float]]) -> float:
        """Evaluate a constraint utility function expression.
        
        Args:
            function_str: Expression string, e.g., "-10.5 * penalty_var" or "1 / (sum ** 2)"
            variable_values: Dict mapping variable names to their values (typically 0 or 1)
        
        Returns:
            Utility value as float. Returns 0.0 on error.
        
        Examples:
            >>> evaluator = ConstraintFunctionEvaluator()
            >>> evaluator.evaluate("-10 * x", {"x": 1})
            -10.0
            >>> evaluator.evaluate("1 / (sum ** 2)", {"sum": 2})
            0.25
        """
        if not function_str or not isinstance(function_str, str):
            return 0.0

        try:
            # Validate and prepare expression
            function_str = function_str.strip()

            # Create safe namespace with allowed operations
            safe_namespace = dict(self.ALLOWED_BUILTINS)
            safe_namespace.update(variable_values)

            # Evaluate with restricted scope
            result = eval(
                function_str,
                {"__builtins__": {}},
                safe_namespace
            )

            # Ensure result is numeric
            return float(result) if result is not None else 0.0

        except ZeroDivisionError:
            logger.warning(f"Division by zero in constraint function: {function_str}")
            return 0.0
        except (NameError, KeyError) as e:
            logger.warning(f"Undefined variable in constraint function '{function_str}': {e}")
            return 0.0
        except (ValueError, TypeError) as e:
            logger.warning(f"Type error in constraint function '{function_str}': {e}")
            return 0.0
        except Exception as e:
            self.error_count += 1
            logger.warning(f"Error evaluating constraint function '{function_str}': {e}")
            return 0.0

    def validate_expression(self, function_str: str) -> bool:
        """Check if expression is valid (doesn't crash on evaluation).
        
        Args:
            function_str: Expression to validate
        
        Returns:
            True if expression is safe to evaluate, False otherwise
        """
        try:
            # Test with dummy variables
            test_vars = {
                'x': 1, 'y': 1, 'z': 1,
                'penalty': 1, 'var': 1, 'sum': 2,
                'v_a1_0': 1, 'v_a2_0': 1,  # Common variable names
            }
            safe_namespace = dict(self.ALLOWED_BUILTINS)
            safe_namespace.update(test_vars)

            eval(
                function_str,
                {"__builtins__": {}},
                safe_namespace
            )
            return True

        except Exception as e:
            logger.debug(f"Expression validation failed for '{function_str}': {e}")
            return False

    def extract_variables(self, function_str: str) -> set:
        """Extract variable names from a constraint function expression.
        
        Args:
            function_str: Expression string
        
        Returns:
            Set of variable names found in the expression
        
        Examples:
            >>> evaluator = ConstraintFunctionEvaluator()
            >>> evaluator.extract_variables("-penalty * var_x")
            {'penalty', 'var_x'}
        """
        variables = set()

        if not function_str or not isinstance(function_str, str):
            return variables

        try:
            # Find all potential variable names (identifiers)
            # This pattern matches Python identifiers
            pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
            candidates = re.findall(pattern, function_str)

            # Filter out keywords and built-in names
            keywords = {
                'and', 'or', 'not', 'in', 'is', 'if', 'else',
                'True', 'False', 'None',
                'abs', 'min', 'max', 'sum', 'pow', 'round', 'len'
            }

            for name in candidates:
                if name not in keywords and not name[0].isdigit():
                    variables.add(name)

        except Exception as e:
            logger.debug(f"Error extracting variables from '{function_str}': {e}")

        return variables

    def get_error_count(self) -> int:
        """Get count of errors encountered during evaluation."""
        return self.error_count

    def reset_error_count(self) -> None:
        """Reset error counter."""
        self.error_count = 0


def create_constraint_evaluator() -> ConstraintFunctionEvaluator:
    """Factory function to create a constraint evaluator instance.
    
    Returns:
        ConstraintFunctionEvaluator instance
    """
    return ConstraintFunctionEvaluator()


# Module-level instance for convenience
_default_evaluator = None


def get_default_evaluator() -> ConstraintFunctionEvaluator:
    """Get or create the default evaluator instance.
    
    Returns:
        ConstraintFunctionEvaluator instance
    """
    global _default_evaluator
    if _default_evaluator is None:
        _default_evaluator = ConstraintFunctionEvaluator()
    return _default_evaluator


if __name__ == "__main__":
    # Test the evaluator
    evaluator = ConstraintFunctionEvaluator()

    # Test cases
    test_cases = [
        ("-10 * penalty_var", {"penalty_var": 1}, -10.0),
        ("-10 * penalty_var", {"penalty_var": 0}, 0.0),
        ("1 / (sum ** 2)", {"sum": 2}, 0.25),
        ("1 / (sum ** 2)", {"sum": 1}, 1.0),
        ("100 - x * y", {"x": 5, "y": 10}, 50.0),
        ("max(a, b)", {"a": 10, "b": 20}, 20.0),
        ("abs(-x)", {"x": 5}, 5.0),
    ]

    print("Testing ConstraintFunctionEvaluator:")
    print("-" * 60)

    for expr, vars_dict, expected in test_cases:
        result = evaluator.evaluate(expr, vars_dict)
        status = "✓" if abs(result - expected) < 1e-6 else "✗"
        vars_str = str(vars_dict)[:25]
        print(f"{status} {expr:30} {vars_str:25} => {result:10.2f}")

    print("-" * 60)
    print(f"Errors encountered: {evaluator.get_error_count()}")
