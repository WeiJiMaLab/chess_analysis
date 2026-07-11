"""This GPU-bound check (does re-running real PUCT+lc0 generation match pack.py's
retroactive replay?) is not run via pytest -- see:

    slurm/pipeline/_verify_forward_backward_unpack.py   (standalone script)
    slurm/pipeline/verify_forward_backward_unpack.slurm (GPU SLURM launcher)

Run it with: sbatch slurm/pipeline/verify_forward_backward_unpack.slurm [NUM_FENS]
"""
