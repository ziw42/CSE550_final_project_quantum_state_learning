import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from MPS_POVM_sampler.noisygeneration import PaMPS
from evaluation.ncon import ncon


def enumerate_sequences(K, L):
    """Return lexicographically ordered tuples for all length-L sequences over range(K)."""
    indices = []
    vectors = []
    for idx in np.ndindex(*(K,) * L):
        indices.append(tuple(idx))
        vec = np.zeros(L * K, dtype=np.float64)
        for site, outcome in enumerate(idx):
            vec[site * K + outcome] = 1.0
        vectors.append(vec)
    return indices, np.vstack(vectors)


def load_rbm_params(npz_path):
    data = np.load(npz_path)
    W = data["weights"]
    vb = data["visible_bias"].reshape(-1)
    hb = data["hidden_bias"].reshape(-1)
    return W, vb, hb


def free_energy(v, W, vb, hb, hidden_states=2):
    x = np.asarray(v, dtype=np.float64)
    f1 = -float(x.dot(vb))
    Wx = W.T.dot(x) + hb
    if Wx.size % hidden_states != 0:
        raise ValueError(
            "Hidden dimension %d not divisible by hidden_states=%d"
            % (Wx.size, hidden_states)
        )
    H = Wx.size // hidden_states
    logits = Wx.reshape(H, hidden_states)
    m = np.max(logits, axis=1)
    f3 = m + np.log(np.sum(np.exp(logits - m[:, None]), axis=1))
    return f1 - float(np.sum(f3))


def rbm_probabilities(seq_vectors, W, vb, hb, hidden_states=2, eps=1e-12):
    logp = -np.array([
        free_energy(v, W, vb, hb, hidden_states=hidden_states) for v in seq_vectors
    ])
    m = np.max(logp)
    probs = np.exp(logp - m)
    total = float(np.sum(probs))
    if total <= 0:
        raise ValueError("Sum of RBM probabilities is non-positive")
    return probs / total


def distribution_metrics(target, model, eps=1e-12):
    target = np.asarray(target, dtype=np.float64)
    model = np.asarray(model, dtype=np.float64)
    if target.shape != model.shape:
        raise ValueError("Mismatched distributions: %s vs %s" % (target.shape, model.shape))
    if np.any(target < 0) or np.any(model < 0):
        raise ValueError("Negative probabilities encountered.")
    target_sum = float(np.sum(target))
    model_sum = float(np.sum(model))
    if target_sum <= 0 or model_sum <= 0:
        raise ValueError("Invalid probability sums: target=%f model=%f" % (target_sum, model_sum))
    target = target / target_sum
    model = model / model_sum
    kl = float(np.sum(target * (np.log(target + eps) - np.log(model + eps))))
    fidelity = float(np.sum(np.sqrt((target + eps) * (model + eps))))
    return kl, fidelity


def _enumerate_branch(pamps, L, prefix, site_idx, PP, prefix_prob, probs):
    if site_idx >= L:
        probs[tuple(prefix)] = float(prefix_prob)
        return
    if site_idx == 0:
        raise ValueError("Internal error: site_idx=0 should be handled separately")
    Pnum = np.real(ncon((PP, pamps.l_P[site_idx]), ([1, 2], [1, 2, -1])))
    for outcome in range(pamps.K):
        prob_next = float(Pnum[outcome])
        if prob_next <= 0:
            continue
        new_prefix = prefix + [outcome]
        if site_idx == L - 1:
            probs[tuple(new_prefix)] = prob_next
        else:
            PP_next = ncon(
                (PP, pamps.LocxM[:, :, outcome], pamps.MPS[site_idx], pamps.MPS[site_idx]),
                ([1, 2], [3, 4], [1, 3, -1], [2, 4, -2]),
            )
            _enumerate_branch(pamps, L, new_prefix, site_idx + 1, PP_next, prob_next, probs)


def noisy_mps_probabilities(povm, tensor_path, full_n, noise_p, L):
    """Compute exact joint distribution for the first L POVM outcomes with depolarizing noise p."""
    sampler = PaMPS(POVM=povm, Number_qubits=full_n, MPS=tensor_path, p=noise_p)
    if L < 1 or L > sampler.N:
        raise ValueError("Invalid L=%d for sampler length %d" % (L, sampler.N))
    probs = {}
    base = np.real(sampler.l_P[0])
    for outcome in range(sampler.K):
        prob0 = float(base[outcome])
        if prob0 <= 0:
            continue
        prefix = [outcome]
        if L == 1:
            probs[tuple(prefix)] = prob0
            continue
        PP = ncon(
            (sampler.M[outcome], sampler.locMixer, sampler.MPS[0], sampler.MPS[0]),
            ([2, 1], [1, 4, 3, 2], [3, -1], [4, -2]),
        )
        _enumerate_branch(sampler, L, prefix, 1, PP, prob0, probs)
    seq_indices, _ = enumerate_sequences(sampler.K, L)
    clean = np.zeros(len(seq_indices), dtype=np.float64)
    for idx, seq in enumerate(seq_indices):
        clean[idx] = probs.get(seq, 0.0)
    total = float(np.sum(clean))
    if total <= 0:
        raise ValueError("Computed zero probability mass for POVM distribution")
    return clean / total