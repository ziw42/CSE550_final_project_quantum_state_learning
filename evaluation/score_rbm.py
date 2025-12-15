import argparse
import numpy as np

from metrics_utils import (
    enumerate_sequences,
    load_rbm_params,
    noisy_mps_probabilities,
    rbm_probabilities,
    distribution_metrics,
)


def infer_visible_dim(path):
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = np.fromstring(line, sep=' ', dtype=np.float64)
            if vals.size == 0:
                continue
            return vals.size
    raise ValueError(f"No samples found in {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate an RBM checkpoint against the theoretical Bell distribution"
    )
    parser.add_argument('--params', required=True, help='Path to RBM parameters (.npz)')
    parser.add_argument('--data', required=True, help='Sample file used to infer L and sanity-check dimensions')
    parser.add_argument('--tensor', default='data/training_data_1D_TFIM_model/Matrix_product_state/tensor.txt', help='Path to tensor.txt describing the reference MPS')
    parser.add_argument('--povm', default='Tetra', choices=['Tetra', '4Pauli', 'Trine', 'Pauli'], help='POVM used for sampling and evaluation')
    parser.add_argument('--full-n', type=int, default=50, help='Total number of qubits in the stored MPS tensor')
    parser.add_argument('--p', type=float, default=0.0, help='Local depolarizing noise strength for the Bell distribution')
    parser.add_argument('--K', type=int, default=4, help='Outcomes per site of the POVM (K)')
    parser.add_argument('--L', type=int, default=None, help='Number of measured sites (prefix length). If omitted, infer from data and K')
    parser.add_argument('--hidden-states', type=int, default=2, help='States per hidden unit in the RBM')
    parser.add_argument('--eps', type=float, default=1e-12, help='Numerical epsilon for logs and normalization')
    return parser.parse_args()


def main():
    args = parse_args()

    vis_dim = infer_visible_dim(args.data)
    if args.K <= 0:
        raise ValueError('K must be positive')
    inferred_L = vis_dim // args.K
    if vis_dim % args.K != 0:
        raise ValueError(
            f'Sample dimension {vis_dim} is not divisible by K={args.K}; please specify L explicitly.'
        )
    L = args.L if args.L is not None else inferred_L
    expected_dim = L * args.K
    if vis_dim != expected_dim:
        raise ValueError(
            f'Sample dimension {vis_dim} does not match L*K={expected_dim}. Set --L accordingly.'
        )

    seq_indices, seq_vectors = enumerate_sequences(args.K, L)
    clean_probs = noisy_mps_probabilities(
        args.povm,
        args.tensor,
        args.full_n,
        args.p,
        L,
    )

    W, vb, hb = load_rbm_params(args.params)
    rbm_probs = rbm_probabilities(
        seq_vectors,
        W,
        vb,
        hb,
        hidden_states=args.hidden_states,
        eps=args.eps,
    )

    dkl, classical_fid = distribution_metrics(clean_probs, rbm_probs, args.eps)

    print('D_KL(P_Bell || P_RBM):', dkl)
    print('F_C(P_Bell, P_RBM):', classical_fid)


if __name__ == '__main__':
    main()
