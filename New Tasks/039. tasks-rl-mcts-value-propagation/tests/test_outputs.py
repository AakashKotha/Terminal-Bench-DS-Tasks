import pytest
import sys
import os
import random
import numpy as np

# Ensure we can import 
sys.path.append("/app")
sys.path.append("/app/task-deps")

# Attempt to load solution, fallback to original to verify harness works (it should fail with original)
if os.path.exists("/app/solution.py"):
    import solution as mcts_module
    from solution import MCTS
else:
    import mcts as mcts_module
    from mcts import MCTS

from game import Connect4

def play_game(agent_cls_or_func, seed=None):
    """
    Plays one game. 
    Agent is P1 (starts). Random is P2.
    Returns: 1 if Agent wins, -1 if Random wins, 0 if draw.
    """
    if seed:
        random.seed(seed)
        np.random.seed(seed)
        
    game = Connect4()
    
    while not game.is_terminal():
        if game.player == 1:
            # Agent Turn
            # Use small iterations for test speed, but enough for simple tactics
            agent = MCTS(game, iterations=200) 
            move = agent.run()
        else:
            # Random Turn
            move = random.choice(game.get_valid_moves())
            
        game.step(move)
        
    return game.get_result(1) # Return result from Agent's perspective

def test_mcts_logic_fix():
    """
    Sanity check: Does backpropagate handle alternating layers or perspective?
    We inspect the code logic or behavior on a forced win.
    """
    # 1. Setup a board where P1 can win immediately.
    # . . . . . . .
    # . . . . . . .
    # . . . . . . .
    # . . . . . . .
    # 1 1 1 0 . . . 
    # 2 2 2 0 . . .
    # P1 (1) to move. Col 3 is valid. Winning move is Col 0, 1, 2... wait.
    # Let's set up:
    # 1 1 1 _ _ _ _
    # If P1 plays col 3, P1 wins.
    
    game = Connect4()
    # Force state
    game.board[0, 0] = 1
    game.board[0, 1] = 1
    game.board[0, 2] = 1
    # Opponent pieces elsewhere
    game.board[1, 0] = -1
    game.board[1, 1] = -1
    game.board[1, 2] = -1
    
    game.player = 1
    
    agent = MCTS(game, iterations=100)
    move = agent.run()
    
    # With the bug, MCTS might not distinguish winning from losing paths sharply enough
    # or might accidentally pick a move that lets P2 win next if it was deeper.
    # In a 1-step win, simulation should return +1 immediately. 
    # Even buggy MCTS might get this right because depth=1. 
    # Real failure is depth > 1.
    
    if move != 3:
        print(f"Agent failed simple 1-step win check. Played {move}, expected 3.")
        # We don't hard fail here, we rely on the win-rate test, but this is a bad sign.

def test_win_rate_against_random():
    """
    The main benchmark. 
    Broken MCTS winrate vs Random is ~50-60% (since it plans for opponent to help it).
    Fixed MCTS winrate vs Random (with 200 iter) should be > 90% easily on Connect 4.
    """
    wins = 0
    draws = 0
    losses = 0
    n_games = 20
    
    print(f"Running {n_games} games against Random Agent...")
    
    for i in range(n_games):
        result = play_game(MCTS, seed=42+i)
        if result == 1:
            wins += 1
        elif result == 0:
            draws += 1
        else:
            losses += 1
            
    print(f"Results: Wins={wins}, Draws={draws}, Losses={losses}")
    
    win_rate = wins / n_games
    
    # With bug: typically fails significantly.
    # With fix: typically 19/20 or 20/20.
    if win_rate < 0.85:
        pytest.fail(f"MCTS Agent performed poorly (Win Rate: {win_rate:.2%}). "
                    "Expected > 85%. The backpropagation logic likely still reinforces opponent moves incorrectly.")
        
    print("Test Passed: Agent dominates random player.")