"""A small bidirectional GRU over the same features, at 5 fps, for temporal context.

Training needs PyTorch (requirements-train.txt); inference is plain numpy (`NumpyGRU`), so
regenerating the submission needs no PyTorch. Inputs are the frame features, robust-scaled with
statistics from the training matches only, NaN set to 0. The home/away swap is applied to half of
the training chunks, and predictions average the match and its swap.
"""

import numpy as np

from . import features

STEP = 5           # 25 fps -> 5 fps
CHUNK = 256        # ~51 s per training chunk
HIDDEN = 64
EPOCHS = 10
BATCH = 16
LR = 2e-3
CODE = {"H": 0, "A": 1, "D": 2}


def scaler(Xs):
    """(median, iqr) per feature over the training matches (5 fps rows)."""
    Z = np.vstack([X[::STEP] for X in Xs])
    med = np.nanmedian(Z, axis=0)
    q1, q3 = np.nanpercentile(Z, [25, 75], axis=0)
    iqr = np.where((q3 - q1) > 1e-6, q3 - q1, 1.0)
    return np.nan_to_num(med).astype(np.float32), iqr.astype(np.float32)


def scale(X, sc):
    Z = (X - sc[0]) / sc[1]
    return np.clip(np.nan_to_num(Z, nan=0.0), -10, 10).astype(np.float32)


def sequences(X, y, halves, sc):
    """[(Z (n, F), codes (n,) with -1 for X)] per half at 5 fps."""
    out = []
    for _, a, b in halves:
        Z = scale(X[a:b + 1:STEP], sc)
        c = np.full(len(Z), -1, np.int64)
        if y is not None:
            yy = y[a:b + 1:STEP]
            for k, v in CODE.items():
                c[yy == k] = v
        out.append((Z, c))
    return out


def train(seqs, names, seed=0, epochs=EPOCHS, threads=8):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)
    F = seqs[0][0].shape[1]
    si = features.swap_index(names)
    net = Net(F)
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    lossf = nn.CrossEntropyLoss(ignore_index=-1)
    for ep in range(epochs):
        starts = [(i, s) for i, (Z, _) in enumerate(seqs)
                  for s in range(-int(rng.integers(CHUNK)), len(Z), CHUNK)]
        order = rng.permutation(len(starts))
        net.train()
        for k in range(0, len(order), BATCH):
            xb, yb = [], []
            for j in order[k:k + BATCH]:
                i, s = starts[j]
                Z, c = seqs[i]
                s = max(0, s)
                z, cc = Z[s:s + CHUNK], c[s:s + CHUNK]
                if len(z) < CHUNK:
                    pad = CHUNK - len(z)
                    z = np.vstack([z, np.zeros((pad, F), np.float32)])
                    cc = np.r_[cc, np.full(pad, -1)]
                if rng.random() < 0.5:
                    z = z[:, si]
                    cc = np.where(cc == 0, 1, np.where(cc == 1, 0, cc))
                xb.append(z)
                yb.append(cc)
            xb = torch.from_numpy(np.stack(xb))
            yb = torch.from_numpy(np.stack(yb))
            opt.zero_grad()
            out = net(xb)
            loss = lossf(out.reshape(-1, 3), yb.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    net.eval()
    return export(net)


def Net(F):
    from torch import nn

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(F, HIDDEN)
            self.act = nn.GELU(approximate="tanh")
            self.gru = nn.GRU(HIDDEN, HIDDEN, num_layers=2, batch_first=True, bidirectional=True)
            self.out = nn.Linear(2 * HIDDEN, 3)

        def forward(self, x):
            h, _ = self.gru(self.act(self.inp(x)))
            return self.out(h)
    return _Net()


def export(net):
    return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}


def _gelu(x):
    """tanh-approximate GELU, as torch.nn.GELU(approximate="tanh")."""
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def _gru_dir(x, W_ih, W_hh, b_ih, b_hh, reverse):
    n, H = len(x), W_hh.shape[1]
    gi = x @ W_ih.T + b_ih                     # (n, 3H)
    h = np.zeros(H, np.float32)
    out = np.empty((n, H), np.float32)
    rng = range(n - 1, -1, -1) if reverse else range(n)
    for t in rng:
        gh = W_hh @ h + b_hh
        r = 1.0 / (1.0 + np.exp(-(gi[t, :H] + gh[:H])))
        z = 1.0 / (1.0 + np.exp(-(gi[t, H:2 * H] + gh[H:2 * H])))
        nn_ = np.tanh(gi[t, 2 * H:] + r * gh[2 * H:])
        h = (1.0 - z) * nn_ + z * h
        out[t] = h
    return out


def forward(w, Z):
    """Numpy forward pass (same maths as the PyTorch module) -> (n, 3) probabilities."""
    x = _gelu(Z @ w["inp.weight"].T + w["inp.bias"]).astype(np.float32)
    for layer in range(2):
        f = _gru_dir(x, w["gru.weight_ih_l{}".format(layer)], w["gru.weight_hh_l{}".format(layer)],
                     w["gru.bias_ih_l{}".format(layer)], w["gru.bias_hh_l{}".format(layer)], False)
        r = _gru_dir(x, w["gru.weight_ih_l{}_reverse".format(layer)], w["gru.weight_hh_l{}_reverse".format(layer)],
                     w["gru.bias_ih_l{}_reverse".format(layer)], w["gru.bias_hh_l{}_reverse".format(layer)], True)
        x = np.concatenate([f, r], axis=1)
    logits = x @ w["out.weight"].T + w["out.bias"]
    logits -= logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def predict(w, sc, X, halves, names):
    """(T, 3) probabilities at 25 fps: the 5 fps outputs, mean of the match and its swap,
    linearly interpolated back to every frame."""
    si = features.swap_index(names)
    p = np.full((len(X), 3), 1 / 3, np.float32)
    for _, a, b in halves:
        Z = scale(X[a:b + 1:STEP], sc)
        q = forward(w, Z)
        qs = forward(w, Z[:, si])[:, [1, 0, 2]]
        q = 0.5 * (q + qs)
        t5 = np.arange(len(q)) * STEP
        t = np.arange(b + 1 - a)
        for k in range(3):
            p[a:b + 1, k] = np.interp(t, t5, q[:, k])
    return p


def load(path):
    """(weights, scaler) from the committed numpy file."""
    z = np.load(path)
    w = {k.replace("__", "."): z[k] for k in z.files if not k.startswith("scale_")}
    return w, (z["scale_median"], z["scale_iqr"])
