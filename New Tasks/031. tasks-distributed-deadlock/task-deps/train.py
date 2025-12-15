import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time
import random

def init_process(rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    # Use localhost for simulation
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    
    # Initialize process group
    dist.init_process_group(backend, rank=rank, world_size=size)
    
    # Ensure all processes start roughly together
    dist.barrier()
    fn(rank, size)

def simulate_compute_gradients(rank, epoch, step):
    """
    Simulates a training step. 
    Returns a 'gradient' tensor.
    """
    # Simulate random computation time
    time.sleep(random.uniform(0.01, 0.05))
    
    # SIMULATION: Rank 0 sometimes produces "empty" gradients 
    # (e.g., due to sparse data or specific model branching).
    if rank == 0 and random.random() < 0.25:
        return torch.zeros(10)
    
    return torch.randn(10) + rank

def train(rank, size):
    torch.manual_seed(rank)
    print(f"[Rank {rank}] Process started.")

    num_epochs = 5
    steps_per_epoch = 10

    for epoch in range(num_epochs):
        for step in range(steps_per_epoch):
            # 1. Compute Gradients
            grads = simulate_compute_gradients(rank, epoch, step)
            
            # --- THE BUG IS HERE ---
            # The developer added this check to "optimize" bandwidth.
            # If the gradient is effectively zero, they skip the sync.
            # PROBLEM: 'all_reduce' is a collective call. If Rank 0 skips it
            # while Ranks 1-3 enter it, Ranks 1-3 will wait forever for Rank 0.
            grad_norm = torch.norm(grads)
            if grad_norm < 1e-6:
                print(f"[Rank {rank}] Grads near zero (Epoch {epoch}, Step {step}). Skipping sync.")
                continue 
            # -----------------------

            # 2. Synchronize Gradients (All-Reduce)
            try:
                # This call blocks until ALL processes arrive.
                dist.all_reduce(grads, op=dist.ReduceOp.SUM)
                # Average gradients
                grads /= size
            except RuntimeError as e:
                print(f"[Rank {rank}] Error: {e}")
                return

        print(f"[Rank {rank}] Finished Epoch {epoch}")

    print(f"[Rank {rank}] Training Complete.")
    
    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    # We simulate 4 GPUs/Processes on one machine
    WORLD_SIZE = 4
    
    # We use 'spawn' to create new processes
    mp.spawn(init_process,
             args=(WORLD_SIZE, train),
             nprocs=WORLD_SIZE,
             join=True)