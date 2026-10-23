"""Duncan's multiple range test from summary statistics (mean, SD, n per group)."""
import numpy as np
from scipy.stats import studentized_range

def duncan_letters(means, sds, n=3, alpha=0.05):
    means = np.asarray(means, float); sds = np.asarray(sds, float)
    k = len(means)
    mse = np.mean(sds ** 2)            # equal n: pooled within-group variance
    df = k * (n - 1)
    se = np.sqrt(mse / n)
    order = np.argsort(-means)          # descending
    m = means[order]
    def crit(p):
        ap = 1 - (1 - alpha) ** (p - 1)
        return studentized_range.ppf(1 - ap, p, df) * se
    # homogeneous[i][j]: range i..j (in sorted order) not significant (with step-down protection)
    hom = np.zeros((k, k), bool)
    for span in range(k, 0, -1):
        for i in range(0, k - span + 1):
            j = i + span - 1
            if span == 1:
                hom[i, j] = True; continue
            # protected if contained in a homogeneous larger range
            contained = any(hom[a, b] for a in range(0, i + 1) for b in range(j, k) if (b - a + 1) > span)
            hom[i, j] = contained or (m[i] - m[j]) <= crit(span)
    # maximal homogeneous subsets
    subsets = []
    for i in range(k):
        j = max(jj for jj in range(i, k) if hom[i, jj])
        if not any(s[0] <= i and j <= s[1] for s in subsets):
            subsets.append((i, j))
    letters_sorted = [""] * k
    for li, (i, j) in enumerate(subsets):
        for t in range(i, j + 1):
            letters_sorted[t] += "abcdefghijklmnopqrstuvwxyz"[li]
    out = [""] * k
    for pos, idx in enumerate(order):
        out[idx] = letters_sorted[pos]
    return out, mse, df

def duncan_letters_n(means, sds, ns, alpha=0.05):
    """Duncan's test for unequal group sizes: pooled within-group variance with df = sum(n_i - 1) and the Kramer
    pairwise standard error sqrt(MSE / 2 * (1/n_i + 1/n_j))."""
    means = np.asarray(means, float); sds = np.asarray(sds, float); ns = np.asarray(ns, float)
    k = len(means)
    df = (ns - 1).sum(); mse = ((ns - 1) * sds ** 2).sum() / df
    order = np.argsort(-means); m = means[order]; nn = ns[order]
    def crit(p, i, j):
        ap = 1 - (1 - alpha) ** (p - 1)
        return studentized_range.ppf(1 - ap, p, df) * np.sqrt(mse / 2 * (1 / nn[i] + 1 / nn[j]))
    hom = np.zeros((k, k), bool)
    for span in range(k, 0, -1):
        for i in range(0, k - span + 1):
            j = i + span - 1
            if span == 1:
                hom[i, j] = True; continue
            contained = any(hom[a, b] for a in range(0, i + 1) for b in range(j, k) if (b - a + 1) > span)
            hom[i, j] = contained or (m[i] - m[j]) <= crit(span, i, j)
    subsets = []
    for i in range(k):
        j = max(jj for jj in range(i, k) if hom[i, jj])
        if not any(s[0] <= i and j <= s[1] for s in subsets):
            subsets.append((i, j))
    letters_sorted = [""] * k
    for li, (i, j) in enumerate(subsets):
        for t in range(i, j + 1):
            letters_sorted[t] += "abcdefghijklmnopqrstuvwxyz"[li]
    out = [""] * k
    for pos, idx in enumerate(order):
        out[idx] = letters_sorted[pos]
    return out, mse, df


if __name__ == "__main__":
    import pickle, os
    D = os.path.dirname(os.path.abspath(__file__))
    d = pickle.load(open(os.path.join(D, "paper1_data.pkl"), "rb"))
    hs = d["heat_series"]
    agree = tot = 0
    for ind in ["总酚", "总黄酮", "可溶性蛋白", "DPPH", "总还原能力"]:
        for k in ["WB", "HK", "ZZ", "SZ"]:
            s = hs[(hs["指标"] == ind) & (hs["处理"] == k)]
            L, mse, df = duncan_letters(s["均值"].values, s["SD"].values)
            thesis = list(s["字母"])
            same = [a == b for a, b in zip(L, thesis)]
            agree += sum(same); tot += len(same)
            print(ind, k, "calc", L, "thesis", thesis, "OK" if all(same) else "DIFF")
    print("agreement", agree, "/", tot)
