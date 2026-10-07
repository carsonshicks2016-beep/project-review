import io
import sys
import traceback
from typing import Dict, Any, Tuple

def execute_and_test(user_code: str, test_assertions: list[str]) -> Tuple[bool, str]:
    """
    Executes the user's code and then runs a series of assert statements against it.
    Returns (success_boolean, output_or_error_message).
    """
    # Create a clean dictionary to act as the local namespace for the executed code
    local_vars: Dict[str, Any] = {}
    
    # We want to capture anything the user prints
    captured_output = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = captured_output

    try:
        # 1. Execute the user's code
        try:
            exec(user_code, {}, local_vars)
        except SyntaxError as e:
            sys.stdout = original_stdout
            return False, f"Syntax Error: {e.msg} on line {e.lineno}\n{e.text}"
        except Exception as e:
            sys.stdout = original_stdout
            tb = traceback.format_exc()
            return False, f"Runtime Error during execution:\n{tb}"

        # Inject the captured stdout into the local namespace so the LLM can test it
        local_vars['__stdout__'] = captured_output.getvalue()

        # 2. Run the tests in the same namespace
        for assertion in test_assertions:
            try:
                # We exec the assertion so it can test state or function returns
                exec(assertion, {}, local_vars)
            except AssertionError:
                sys.stdout = original_stdout
                return False, f"Test Failed: `{assertion}`"
            except Exception as e:
                sys.stdout = original_stdout
                return False, f"Error while running test `{assertion}`:\n{e}"

        # If we made it here, all tests passed
        sys.stdout = original_stdout
        stdout_value = captured_output.getvalue()
        
        success_message = "All tests passed!"
        if stdout_value:
            success_message += f"\n\nOutput:\n{stdout_value}"
            
        return True, success_message
        
    finally:
        # Ensure we always restore stdout, even if something wild happens
        sys.stdout = original_stdout
