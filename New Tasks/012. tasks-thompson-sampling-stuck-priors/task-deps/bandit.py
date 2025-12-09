import numpy as np

class ThompsonSampler:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        
        # --- THE TRAP IS HERE ---
        # The engineer initialized priors assuming a "fair coin" (50% CTR)
        # with massive confidence (sum=2000).
        # In reality, Ads have low CTR (e.g., 5%).
        # Because these numbers are huge, new data (0 or 1) barely changes the distribution.
        # The variance is tiny, so exploration is dead.
        
        self.alphas = np.array([1000.0] * n_arms)
        self.betas = np.array([1000.0] * n_arms)
        # ------------------------

    def select_arm(self):
        """
        Sample from Beta(alpha, beta) for each arm and pick the max.
        """
        samples = [np.random.beta(a, b) for a, b in zip(self.alphas, self.betas)]
        return np.argmax(samples)

    def update(self, arm, reward):
        """
        Update posterior:
        Alpha += Reward (Success)
        Beta += 1 - Reward (Failure)
        """
        self.alphas[arm] += reward
        self.betas[arm] += (1 - reward)