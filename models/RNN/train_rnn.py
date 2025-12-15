########## RNN model training script ##########

import argparse
import os

import tensorflow as tf

from rnn import LatentAttention

# CLI arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train the POVM RNN model")
    parser.add_argument('--data', default='data/train.txt', help='Path to one-hot POVM samples (each line is L*K entries)')
    parser.add_argument('--p', type=float, default=0.0, help='Noise value p (0-1) to corrupt training data')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for Adam optimizer')
    parser.add_argument('--L', type=int, default=6, help='Number of qubits / measurement sites (max_length)')
    parser.add_argument('--latent', type=int, default=100, help='Latent representation size z')
    parser.add_argument('--hidden', type=int, default=100, help='Hidden size of each GRU layer')
    parser.add_argument('--K', type=int, default=4, help='Local POVM outcomes per site')
    parser.add_argument('--samples', type=int, default=0, help='Number of autoregressive samples to dump per epoch (0 disables sampling)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    return parser.parse_args()

# Main training function
def main():
    args = parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Training data {args.data} not found. Run the sampler first or point --data elsewhere.")

    tf.compat.v1.disable_eager_execution()
    tf.compat.v1.reset_default_graph()

    model = LatentAttention(
        data=args.data,
        K=args.K,
        Number_qubits=args.L,
        latent_rep_size=args.latent,
        gru_hidden=args.hidden,
        decoder='TimeDistributed_mol',
        Nsamples=args.samples,
        noise_p=args.p,
        learning_rate=args.lr,
        epochs=args.epochs,
    )
    model.train()


if __name__ == '__main__':
    main()
