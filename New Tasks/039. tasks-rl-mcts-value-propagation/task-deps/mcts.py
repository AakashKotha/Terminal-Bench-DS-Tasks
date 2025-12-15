import math
import random
import numpy as np

class Node:
    def __init__(self, game_state, parent=None, move_taken=None):
        self.game_state = game_state
        self.parent = parent
        self.move_taken = move_taken
        self.children = []
        self.visits = 0
        self.value = 0.0 # Cumulative value
        
        # Expandable moves
        self.untried_moves = game_state.get_valid_moves()

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, c_param=1.414):
        # Upper Confidence Bound (UCT)
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt((2 * math.log(self.visits) / child.visits))
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

class MCTS:
    def __init__(self, root_game, iterations=1000):
        self.root = Node(root_game)
        self.iterations = iterations
        self.root_player = root_game.player

    def run(self):
        for _ in range(self.iterations):
            node = self.tree_policy(self.root)
            reward = self.default_policy(node.game_state)
            self.backpropagate(node, reward)
        
        # Robust child: most visits
        best_move = max(self.root.children, key=lambda c: c.visits).move_taken
        return best_move

    def tree_policy(self, node):
        while not node.game_state.is_terminal():
            if not node.is_fully_expanded():
                return self.expand(node)
            else:
                node = node.best_child()
        return node

    def expand(self, node):
        move = node.untried_moves.pop()
        new_state = node.game_state.copy()
        new_state.step(move)
        child_node = Node(new_state, parent=node, move_taken=move)
        node.children.append(child_node)
        return child_node

    def default_policy(self, state):
        # Simulation: Random rollout
        current_state = state.copy()
        while not current_state.is_terminal():
            possible_moves = current_state.get_valid_moves()
            move = random.choice(possible_moves)
            current_state.step(move)
        
        # --- CRITICAL CONTEXT ---
        # The result here is returned from the perspective of the *root player*.
        # If root_player wins, +1. If loses, -1.
        return current_state.get_result(self.root_player)

    def backpropagate(self, node, reward):
        # --- THE BUG IS HERE ---
        # The logic simply adds the 'reward' (which is from Root Player's perspective)
        # to every node up the chain.
        #
        # IN A ZERO-SUM GAME:
        # If the root is Player 1.
        # - Depth 1 (Root's children) are states where Player 1 just moved. 
        #   (Technically, these are P2 to-move states).
        #   Usually MCTS nodes store "Value for the player who made the move leading to this node".
        #   OR "Value for the player whose turn it is at this node".
        #
        # Standard UCT Implementation assumption:
        # parent.best_child() maximizes (child.value/child.visits).
        # This implies 'child.value' should represent how good the move was for the PARENT.
        #
        # Current Logic:
        # If Reward = +1 (P1 Wins).
        # We add +1 to a node at Depth 2 (P2 made a move).
        # This tells P2: "This was a great move!" (because it led to P1 winning).
        # P2 will then choose this move again.
        # 
        # FIX: The reward accumulation must respect whose turn it was, OR we must flip
        # the reward at every level if we are using the Negamax assumption in MCTS.
        
        while node is not None:
            node.visits += 1
            node.value += reward # BUG: This reinforces moves for BOTH players equally.
            node = node.parent