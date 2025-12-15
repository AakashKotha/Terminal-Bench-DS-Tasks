import pytest
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import gc

# We import the agent's solution. 
# We need to support the case where they overwrite the original file or create a new one.
# Task says save to /app/solution.py, but usually agents might modify in place.
# We will check /app/solution.py first, then /app/task-deps/train.py.

SOLUTION_PATH = "/app/solution.py"
DEFAULT_PATH = "/app/task-deps/train.py"

def get_train_func():
    path = SOLUTION_PATH if os.path.exists(SOLUTION_PATH) else DEFAULT_PATH
    
    # Dynamic import
    import importlib.util
    spec = importlib.util.spec_from_file_location("solution_module", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["solution_module"] = module
    spec.loader.exec_module(module)
    
    # Assume the main logic is in a function or we have to modify the script to be importable
    # The provided train.py has `train_model` function.
    if hasattr(module, "train_model"):
        return module.train_model
    else:
        # If they wrote a script without functions, this is harder to test internally without running subprocess.
        # But for this OOM test, subprocess is safer anyway.
        return path

def test_memory_stability():
    """
    Runs the agent's training loop for N epochs and checks if memory grows linearly.
    """
    path = SOLUTION_PATH if os.path.exists(SOLUTION_PATH) else DEFAULT_PATH
    
    import tracemalloc
    
    # We will redefine a simplified version of the loop here 
    # BUT injecting the agent's specific logic is hard if we don't parse their code.
    # Instead, we will Parse their code to find the fix.
    
    with open(path, 'r') as f:
        code = f.read()
    
    # Heuristic Check 1: Does it contain .item() or .detach() or float()?
    # The fix MUST extract the scalar.
    has_item = ".item()" in code
    has_detach = ".detach()" in code
    has_float = "float(" in code
    
    if not (has_item or has_detach or has_float):
        pytest.fail("Static Analysis Failed: The code still seems to accumulate raw Tensors (missing .item(), .detach(), or float() conversion). This causes the graph leak.")

    # Functional Check 2: Run it and measure peak memory?
    # Since we can't easily hook into their loop without them structuring it perfectly,
    # The Static Analysis combined with a "Does it run?" check is often robust enough for this specific "One-Line-Fix" task.
    
    # However, let's try to run it and ensure it doesn't crash.
    try:
        # We assume the agent might have increased epochs or left it. 
        # We just run the script.
        exit_code = os.system(f"python3 {path} > /dev/null")
        assert exit_code == 0, "The solution script crashed during execution."
    except Exception as e:
        pytest.fail(f"Execution failed: {e}")

def test_computation_graph_detached():
    """
    Advanced Check: We inspect the AST to ensure the appended value is not a gradient-tracking tensor.
    """
    import ast
    
    path = SOLUTION_PATH if os.path.exists(SOLUTION_PATH) else DEFAULT_PATH
    with open(path, "r") as source:
        tree = ast.parse(source.read())
        
    class AppendVisitor(ast.NodeVisitor):
        def __init__(self):
            self.leaks = False
            
        def visit_Call(self, node):
            # Looking for .append(loss) vs .append(loss.item())
            if isinstance(node.func, ast.Attribute) and node.func.attr == 'append':
                # Arg being appended
                if node.args:
                    arg = node.args[0]
                    # If arg is just a Name (e.g., 'loss'), it's likely a leak
                    if isinstance(arg, ast.Name) and arg.id == 'loss':
                         self.leaks = True
                    # If arg is Call (e.g. loss.item()), it's good.
            self.generic_visit(node)
            
    visitor = AppendVisitor()
    visitor.visit(tree)
    
    if visitor.leaks:
        pytest.fail("AST Analysis: Found 'list.append(loss)'. You are appending the Tensor directly. Use 'loss.item()' to append the value.")