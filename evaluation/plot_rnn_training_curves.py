import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import tensorflow as tf

from metrics_utils import (
    enumerate_sequences,
    noisy_mps_probabilities,
    distribution_metrics,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in os.sys.path:
    os.sys.path.append(ROOT)

from models.RNN.rnn import LatentAttention


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate RNN checkpoints across noise levels and plot evaluation curves"
    )
    parser.add_argument(
        "--p-values",
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma separated list of depolarizing noise strengths p",
    )
    parser.add_argument(
        "--epochs",
        default="0-49:1",
        help="Epoch checkpoints to evaluate. Use comma list and ranges like start-end:step",
    )
    parser.add_argument(
        "--param-pattern",
        default="RNN_parameters/L{L}_K{K}_latent{latent}_hid{hidden}_p{p}",
        help=(
            "Directory name pattern for checkpoint folders. "
            "Use placeholders {p},{L},{K},{latent},{hidden}."
        ),
    )
    parser.add_argument(
        "--data-pattern",
        default="data/train_p{p}.txt",
        help="Path pattern for measurement data aligned with each p",
    )
    parser.add_argument(
        "--checkpoint-prefix",
        default="model",
        help='Checkpoint prefix inside the folder (TensorFlow saver uses "model-{epoch}")',
    )

    parser.add_argument("--povm", default="Tetra", help="POVM type")
    parser.add_argument(
        "--tensor",
        default="data/training_data_1D_TFIM_model/Matrix_product_state/tensor.txt",
        help="Path to tensor.txt for the MPS",
    )
    parser.add_argument("--full-n", type=int, default=50, help="Total qubits in the stored MPS")
    parser.add_argument("--L", type=int, default=2, help="Measured prefix length (Number_qubits)")
    parser.add_argument("--K", type=int, default=4, help="Outcomes per site (POVM dimension)")
    parser.add_argument("--latent", type=int, default=100, help="Latent size used during training")
    parser.add_argument("--hidden", type=int, default=100, help="GRU hidden size used during training")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=60000,
        help="Number of autoregressive samples to draw from the RNN for each checkpoint",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed for reproducible sampling")

    parser.add_argument(
        "--metrics-csv",
        default="evaluation/epoch_metrics_rnn.csv",
        help="Where to save the raw metrics table",
    )
    parser.add_argument(
        "--figure",
        default="evaluation/epoch_metrics_rnn.png",
        help="Where to save the figure",
    )
    parser.add_argument("--eps", type=float, default=1e-12, help="Numerical epsilon for logs")
    return parser.parse_args()


def parse_value_list(spec):
    values = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk and ":" in chunk:
            left, step = chunk.split(":")
            start, end = left.split("-")
            start = int(start)
            end = int(end)
            step = int(step)
            values.extend(list(range(start, end + 1, step)))
        else:
            values.append(int(chunk))
    return sorted(set(values))


def parse_float_list(spec):
    vals = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        vals.append(float(chunk))
    return vals


def ensure_path(path):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)


def format_p(p):
    return f"{p:.1f}"


def apply_template(template, args, p_str):
    return template.format(
        p=p_str,
        L=args.L,
        K=args.K,
        latent=args.latent,
        hidden=args.hidden,
        POVM=args.povm,
        povm=args.povm,
        FULL_N=args.full_n,
        full_n=args.full_n,
    )


def read_samples(path):
    samples = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = np.fromstring(line, sep=" ", dtype=np.float64)
            samples.append(vals)
    if not samples:
        raise ValueError(f"No data found in {path}")
    return np.vstack(samples)


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


def sample_from_checkpoint(checkpoint_prefix, args):
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")

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
        decoder="TimeDistributed_mol",
        Nsamples=0,
    )

    saver = tf.compat.v1.train.Saver()
    total_batches = int(np.ceil(args.num_samples / model.batchsize))
    collected = []
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        saver.restore(sess, checkpoint_prefix)
        remaining = args.num_samples
        for _ in range(total_batches):
            batch = sess.run(model.sample_RNN)
            collected.append(batch)
            remaining -= batch.shape[0]
            if remaining <= 0:
                break
    return np.vstack(collected)[: args.num_samples]


def main():
    args = parse_args()

    p_values = parse_float_list(args.p_values)
    epoch_values = parse_value_list(args.epochs)

    results = []

    for p in p_values:
        p_tag = format_p(p)

        args.data = apply_template(args.data_pattern, args, p_tag)
        if not os.path.exists(args.data):
            raise FileNotFoundError(
                f"Missing data for p={p}: {args.data}. Ensure data/train_p{p_tag}.txt exists."
            )
        samples_onehot = read_samples(args.data)
        expected_dim = args.L * args.K
        if samples_onehot.shape[1] != expected_dim:
            raise ValueError(
                f"Dataset {args.data} has visible dim {samples_onehot.shape[1]}, expected {expected_dim} (L={args.L}, K={args.K})."
            )

        clean_probs = noisy_mps_probabilities(args.povm, args.tensor, args.full_n, p, args.L)

        ckpt_dir = apply_template(args.param_pattern, args, p_tag)
        if not os.path.isdir(ckpt_dir):
            raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} not found for p={p}")

        for epoch in epoch_values:
            ckpt_prefix = os.path.join(ckpt_dir, f"{args.checkpoint_prefix}-{epoch}")
            if not os.path.exists(ckpt_prefix + ".index"):
                continue

            samples_int = sample_from_checkpoint(ckpt_prefix, args)
            emp_probs = empirical_distribution_from_samples(samples_int, args.K, args.L)
            dkl, fc = distribution_metrics(clean_probs, emp_probs, args.eps)

            results.append({"p": p, "epoch": epoch, "dkl": dkl, "fc": fc})

    if not results:
        raise RuntimeError(
            "No metrics computed. Ensure checkpoints and data files exist for the requested settings."
        )

    results.sort(key=lambda r: (r["p"], r["epoch"]))

    ensure_path(args.metrics_csv)
    with open(args.metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["p", "epoch", "D_KL(P||P_RNN)", "F_C(P,P_RNN)"])
        for row in results:
            writer.writerow([row["p"], row["epoch"], row["dkl"], row["fc"]])

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharex=False)
    cmap = LinearSegmentedColormap.from_list(
        "orange_green",
        ["#f97306", "#ffb347", "#8bc34a", "#0f9d58"],
    )
    p_sorted = sorted({row["p"] for row in results})
    color_positions = np.linspace(0, 1, max(len(p_sorted), 2))

    for idx, p in enumerate(p_sorted):
        rows = [r for r in results if abs(r["p"] - p) < 1e-9]
        if not rows:
            continue
        rows.sort(key=lambda r: r["epoch"])
        epochs = [r["epoch"] for r in rows]
        dkl_vals = [r["dkl"] for r in rows]
        fc_vals = [r["fc"] for r in rows]
        color = cmap(color_positions[idx]) if idx < len(color_positions) else cmap(1.0)

        axes[0].plot(epochs, dkl_vals, marker="o", linewidth=1.2, markersize=3, color=color)
        axes[1].plot(epochs, fc_vals, marker="o", linewidth=1.2, markersize=3, color=color)

    axes[0].set_title("(a) D_KL(P || P_RNN)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("D_KL")

    axes[1].set_title("(b) F_C(P, P_RNN)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F_C")
    axes[1].set_ylim(0.95, 1.01)

    fig.subplots_adjust(left=0.09, right=0.98, top=0.92, bottom=0.18, wspace=0.32)
    ensure_path(args.figure)
    fig.savefig(args.figure, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
