# Gate 0.5 statistics (CPU kernel, free): reads acts_{aligned,drifted}.npy from kaggle/gate05 (GPU kernel), per layer:
#   rho        scene-explained fraction (Balestriero & LeCun Thm 3; estimator of their App. C, see lit_review.md):
#              all 5 languages pooled, and per language as the EN+L pair (m=2)
#   procrustes per language EN->L: whiten each side on a train half, orthogonal Procrustes, residual + top-1
#              retrieval on the held-out half; repeated over random splits
# Circuit components are not known yet (Gate 1 comes later), so this uses the residual stream at every layer.
# Local self-test of the maths: python -c "import stats"   (from kaggle/gate05_stats)
import json, os, time
import numpy as np

LANGS = ["en", "hi", "zh", "mr", "hinglish"]


def whiten_fit(X, k):
    """PCA-whitening map fitted on X (n x d): returns (mean, W) with (X - mean) @ W ~ zero-mean, identity cov, k dims."""
    mu = X.mean(0)
    Xc = X - mu
    if len(X) < X.shape[1]:  # n < d: eigendecompose the n x n Gram matrix (much faster than SVD of n x d)
        lam, U = np.linalg.eigh(Xc @ Xc.T)
        lam, U = lam[::-1][:k], U[:, ::-1][:, :k]
        V = Xc.T @ U / np.sqrt(lam)
    else:
        lam, V = np.linalg.eigh(Xc.T @ Xc)
        lam, V = lam[::-1][:k], V[:, ::-1][:, :k]
    return mu, V / np.sqrt(lam / (len(X) - 1))


def rho(A, ks):
    """A: (states, m, d) observations of each state. Per k: tr(C_f^{1/2})/k after whitening the pooled top-k PCs.
    C_f = Cov(per-state means) - mean within-state covariance / m (unbiased for the state part); eigenvalues clipped to [0,1]."""
    n, m, d = A.shape
    mu, W = whiten_fit(A.reshape(n * m, d), max(ks))  # PCs are ordered, so top-k = first k columns
    out = {}
    for k in ks:
        F = (A - mu) @ W[:, :k]               # n, m, k
        M = F.mean(1)
        Cm = np.cov(M, rowvar=False)
        Wc = sum(np.cov(F[:, j] - M, rowvar=False) * (n - 1) for j in range(m)) / (n * (m - 1))  # within-state cov
        lam = np.clip(np.linalg.eigvalsh(Cm - Wc / m), 0, 1)
        out[k] = round(float(np.sqrt(lam).sum() / k), 4)
    return out


def procrustes(A, ks, n_splits=10, seed=0):
    """A: (states, m, d), column 0 = EN. For each other column L: per split, whiten each side on the train half,
    Q = argmin ||A Q - B|| over O(k); on the test half: residual sum||aQ - b||^2 / (sum||a||^2 + sum||b||^2)
    (0 = perfect, ~1 = unrelated) and top-1 retrieval accuracy (cosine) among test rows.
    Returns {j: {k: stats}} for j = 1..m-1. Same splits for every language (paired across languages)."""
    rng = np.random.default_rng(seed)
    n, m, _ = A.shape
    res = {(j, k): [] for j in range(1, m) for k in ks}
    acc = {(j, k): [] for j in range(1, m) for k in ks}
    for _ in range(n_splits):
        p = rng.permutation(n)
        tr, te = p[: n // 2], p[n // 2:]
        wh = [whiten_fit(A[tr, j], max(ks)) for j in range(m)]
        for j in range(1, m):
            for k in ks:
                (ma, Wa), (mb, Wb) = (wh[0][0], wh[0][1][:, :k]), (wh[j][0], wh[j][1][:, :k])
                U, _, Vt = np.linalg.svd(((A[tr, 0] - ma) @ Wa).T @ ((A[tr, j] - mb) @ Wb))
                At, Bt = (A[te, 0] - ma) @ Wa @ (U @ Vt), (A[te, j] - mb) @ Wb
                # normalise by test energy, not 2k: test rows whitened with train stats are not unit-variance (n_train < d)
                res[j, k].append(float(((At - Bt) ** 2).sum() / ((At ** 2).sum() + (Bt ** 2).sum())))
                An = At / np.linalg.norm(At, axis=1, keepdims=True)
                Bn = Bt / np.linalg.norm(Bt, axis=1, keepdims=True)
                acc[j, k].append(float(((An @ Bn.T).argmax(1) == np.arange(len(te))).mean()))
    f = lambda x: round(float(x), 4)
    return {j: {k: {"resid": f(np.mean(res[j, k])), "resid_sd": f(np.std(res[j, k])),
                    "retr": f(np.mean(acc[j, k])), "retr_sd": f(np.std(acc[j, k]))} for k in ks} for j in range(1, m)}


def geometry(H, ks=(32, 128)):
    """H: (states, 5 langs, layers, d). Per layer and k: pooled rho, pairwise EN+L rho, EN->L Procrustes."""
    ks = [k for k in ks if k < H.shape[0] // 2]  # Procrustes needs n_train >= k (Prop. 2)
    out = []
    for li in range(H.shape[2]):
        A = H[:, :, li].astype(np.float64)
        ra, pr = rho(A, ks), procrustes(A, ks)
        rp = {j: rho(A[:, [0, j]], ks) for j in range(1, len(LANGS))}
        # null: break the pairing (shuffle L's prompts) -> rho a finite-sample estimator gives with no shared state
        perm = np.random.default_rng(li).permutation(len(A))
        rn = {j: rho(np.stack([A[:, 0], A[perm, j]], 1), ks) for j in range(1, len(LANGS))}
        out.append({"layer": li + 1, **{f"k{k}": {"rho_all": ra[k], **{l: {"rho_pair": rp[j][k], "rho_pair_null": rn[j][k], **pr[j][k]}
                                                                       for j, l in enumerate(LANGS[1:], 1)}} for k in ks}})
    return out


if __name__ != "__main__":  # local self-test on synthetic data with known answers
    g = np.random.default_rng(1)
    n, k, d = 600, 16, 64
    Z = g.standard_normal((n, k)) @ g.standard_normal((k, d))      # state part, rank k
    sig = 1.0
    # nuisance: isotropic in the state subspace -> whitened total cov = (1 + sig^2) I on it -> rho = 1/sqrt(1+sig^2)
    basis = np.linalg.qr(Z.T)[0][:, :k]
    Zw = Z @ basis
    Zw = (Zw - Zw.mean(0)) @ np.linalg.inv(np.linalg.cholesky(np.cov(Zw, rowvar=False))).T  # exactly white state
    A = np.stack([(Zw + sig * g.standard_normal((n, k))) @ basis.T for _ in range(5)], 1)
    r = rho(A, [k])[k]
    assert abs(r - 1 / np.sqrt(1 + sig ** 2)) < 0.05, r
    assert rho(np.stack([Zw @ basis.T] * 3, 1) + 1e-6 * g.standard_normal((n, 3, d)), [k])[k] > 0.98
    R = np.linalg.qr(g.standard_normal((d, d)))[0]
    p = procrustes(np.stack([Z, Z @ R * 3 + 5], 1), [k])[1][k]          # exact rotation + scale + shift -> residual ~0, retrieval 1
    assert p["resid"] < 1e-6 and p["retr"] == 1.0, p
    p = procrustes(np.stack([Z, g.standard_normal((n, d))], 1), [k])[1][k]  # unrelated -> residual ~1 (overfit on train only)
    assert p["resid"] > 0.8 and p["retr"] < 0.05, p
    raise SystemExit("self-test ok")

# ---------------------------------------------------------------- Kaggle part (CPU)
import glob, time
W = "/kaggle/working"
src = os.path.dirname(glob.glob("/kaggle/input/**/acts_aligned.npy", recursive=True)[0])
meta = json.load(open(f"{src}/gate05.json"))
harm = [i for i, s in enumerate(meta["state_ids"]) if s.startswith("harmful_eval:")]
R = {"source_cfg": meta["cfg"], "n_states": meta["n_states"], "ks": [32, 128], "n_splits": 10, "split_seed": 0,
     "started": time.strftime("%Y-%m-%d %H:%M:%S")}
for name in ("aligned", "drifted"):
    t = time.time()
    H = np.load(f"{src}/acts_{name}.npy")
    R[name] = {"all": geometry(H, R["ks"]), "harmful_only": geometry(H[harm], [32]), "sec": round(time.time() - t)}
    json.dump(R, open(f"{W}/gate05_stats.json", "w"), indent=1)
    print(name, R[name]["sec"], flush=True)
R["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
json.dump(R, open(f"{W}/gate05_stats.json", "w"), indent=1)
