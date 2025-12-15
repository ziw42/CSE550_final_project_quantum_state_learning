import argparse
import math
import os
import sys

import numpy as np
import tensorflow as tf

from metrics_utils import (
    enumerate_sequences,
    noisy_mps_probabilities,
    distribution_metrics,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from models.RNN.rnn import LatentAttention

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained RNN checkpoints against Bell distributions")
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint prefix (without extensions)')
    parser.add_argument('--data', default='data/train.txt', help='Training data file used to infer shapes')
    parser.add_argument('--tensor', default='data/training_data_1D_TFIM_model/Matrix_product_state/tensor.txt', help='Reference tensor.txt for the Bell state')
    parser.add_argument('--povm', default='Tetra', choices=['Tetra', '4Pauli', 'Trine', 'Pauli'], help='POVM used in sampling')
    parser.add_argument('--p', type=float, default=0.0, help='Local depolarizing noise strength for the target Bell distribution')
    parser.add_argument('--full-n', type=int, default=50, help='Total qubits in the MPS tensor')
    parser.add_argument('--L', type=int, required=True, help='Measured prefix length (must match training)')
    parser.add_argument('--K', type=int, required=True, help='Outcomes per site (POVM dimension)')
    parser.add_argument('--latent', type=int, default=100, help='Latent size used during training')
    parser.add_argument('--hidden', type=int, default=100, help='GRU hidden size used during training')
    parser.add_argument('--num-samples', type=int, default=60000, help='Number of autoregressive samples to draw from the RNN')
    parser.add_argument('--samples-out', default='', help='Optional path to save sampled integer sequences')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed for reproducible sampling')
    parser.add_argument('--eps', type=float, default=1e-12, help='Numerical epsilon for logs')
    return parser.parse_args()


def empirical_distribution_from_samples(samples, K, L):
    seq_indices, _ = enumerate_sequences(K, L)
    index_map = {seq: idx for idx, seq in enumerate(seq_indices)}
    counts = np.zeros(len(seq_indices), dtype=np.float64)
    for row in samples:
        seq = tuple(int(x) for x in row[:L])
        counts[index_map[seq]] += 1
    if counts.sum() == 0:
        raise ValueError("No samples collected for empirical distribution")
    return counts / counts.sum()


def main():
    args = parse_args()

    if not os.path.exists(args.checkpoint + '.index'):
        raise FileNotFoundError(f"Checkpoint {args.checkpoint}.index not found")
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file {args.data} not found")
    if args.num_samples <= 0:
        raise ValueError('--num-samples must be positive')

    np.random.seed(args.seed)
    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.set_random_seed(args.seed)

    model = LatentAttention(
        data=args.data,
        K=args.K,
        Number_qubits=args.L,
        latent_rep_size=args.latent,
        gru_hidden=args.hidden,
        decoder='TimeDistributed_mol',
        Nsamples=0,
    )

    saver = tf.compat.v1.train.Saver()
    total_batches = int(math.ceil(args.num_samples / model.batchsize))
    collected = []
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.restore(sess, args.checkpoint)
        remaining = args.num_samples
        for _ in range(total_batches):
            batch = sess.run(model.sample_RNN)
            collected.append(batch)
            remaining -= batch.shape[0]
            if remaining <= 0:
                break
    samples = np.vstack(collected)[:args.num_samples]

    if args.samples_out:
        directory = os.path.dirname(args.samples_out)
        if directory:
            os.makedirs(directory, exist_ok=True)
        np.savetxt(args.samples_out, samples.astype(np.int64), fmt='%d')

    clean_probs = noisy_mps_probabilities(args.povm, args.tensor, args.full_n, args.p, args.L)
    emp_probs = empirical_distribution_from_samples(samples, args.K, args.L)
    dkl, fid = distribution_metrics(clean_probs, emp_probs, args.eps)

    print('Collected samples:', samples.shape[0])
    print('D_KL(P_Bell || P_RNN):', dkl)
    print('F_C(P_Bell, P_RNN):', fid)


if __name__ == '__main__':
    main()
