"""JAR merged paper, sixth version: C1-C3 (continuous partitioning prediction, structural generalisation, ablation and
chemical-class confounding).
All models of one comparison share the same outer splits; scaling, alpha/hyper-parameter selection happen inside the training
fold only (inner 3-fold).  Predictions of the bounded endpoints R and D are clipped to [0, 1] (rule fixed in advance).
Random scheme: 5-fold x 10 repeats.  Structural scheme: Butina clusters of Morgan fingerprints (radius 2, 2048 bit, Tanimoto
distance cut-off 0.6) kept whole, clusters assigned to 5 folds by a size-balanced greedy rule after shuffling, 10 repeats.
Run: python models.py  (writes v6out/models.pkl and CSV tables)"""
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed
from xgboost import XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core as C                                                                 # noqa: E402

warnings.filterwarnings("ignore")
SEED = 2026
NJOBS = max(1, min(10, (os.cpu_count() or 2) - 1))
NREP = 10
ALPHAS = np.logspace(-3, 3, 13)
XGB_GRID = {"max_depth": [1, 2, 3], "n_estimators": [100, 300]}


# ---------------------------------------------------------------------------------------------------------------- splits
def random_splits(n, nrep=NREP, k=5, seed=SEED):
    out = []
    for r in range(nrep):
        kf = KFold(k, shuffle=True, random_state=seed + r)
        out.append([(tr, te) for tr, te in kf.split(np.arange(n))])
    return out


def group_splits(groups, nrep=NREP, k=5, seed=SEED):
    groups = np.asarray(groups)
    ug = np.unique(groups)
    out = []
    for r in range(nrep):
        rng = np.random.default_rng(seed + 100 + r)
        order = rng.permutation(ug)
        sizes = {g: (groups == g).sum() for g in ug}
        order = sorted(order, key=lambda g: -sizes[g])          # largest first, ties broken by the random permutation
        fold_of, load = {}, np.zeros(k)
        for g in order:
            f = int(np.argmin(load + rng.random(k) * 1e-6)); fold_of[g] = f; load[f] += sizes[g]
        fid = np.array([fold_of[g] for g in groups])
        out.append([(np.where(fid != f)[0], np.where(fid == f)[0]) for f in range(k)])
    return out


# ---------------------------------------------------------------------------------------------------------------- models
def fit_predict(kind, Xtr, ytr, Xte, inner_seed=0):
    if kind == "baseline":
        return np.full(len(Xte), ytr.mean()), {}
    if kind == "ridge":
        gs = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": ALPHAS},
                          cv=KFold(3, shuffle=True, random_state=inner_seed), scoring="neg_mean_absolute_error")
        gs.fit(Xtr, ytr)
        return gs.predict(Xte), {"alpha": gs.best_params_["ridge__alpha"],
                                 "coef": gs.best_estimator_[-1].coef_.copy()}
    if kind == "xgb":
        base = XGBRegressor(learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=2,
                            reg_lambda=1.0, n_jobs=1, random_state=inner_seed, verbosity=0)
        gs = GridSearchCV(base, XGB_GRID, cv=KFold(3, shuffle=True, random_state=inner_seed),
                          scoring="neg_mean_absolute_error")
        gs.fit(Xtr, ytr)
        est = gs.best_estimator_
        import xgboost as xgb
        contrib = est.get_booster().predict(xgb.DMatrix(Xte), pred_contribs=True)[:, :-1]
        return est.predict(Xte), {"params": gs.best_params_, "shap_te": contrib}
    raise ValueError(kind)


def metrics(y, p):
    e = p - y
    sst = ((y - y.mean()) ** 2).sum()
    return dict(MAE=np.abs(e).mean(), RMSE=np.sqrt((e ** 2).mean()), R2=1 - (e ** 2).sum() / sst,
                rho=stats.spearmanr(y, p).correlation if np.std(p) > 0 else np.nan)


def _task(k, X, y, tr, te, seed):
    p, ex = fit_predict(k, X[tr], y[tr], X[te], inner_seed=seed)
    return p, ex


def run_cv(X, y, splits, kinds, clip=True, keep_extra=False, n_jobs=NJOBS):
    """returns per-repeat metrics, mean OOF prediction per compound and per-repeat OOF matrix.
    The (repeat, fold) fits are independent and run in parallel; results are identical to the serial loop."""
    n = len(y); res = {k: dict(rep=[], oof=np.zeros((len(splits), n)), extra=[]) for k in kinds}
    for k in kinds:
        jobs = [(r, f, tr, te) for r, folds in enumerate(splits) for f, (tr, te) in enumerate(folds)]
        if k == "xgb" and n_jobs > 1:
            outs = Parallel(n_jobs=n_jobs)(delayed(_task)(k, X, y, tr, te, r * 10 + f) for r, f, tr, te in jobs)
        else:
            outs = [_task(k, X, y, tr, te, r * 10 + f) for r, f, tr, te in jobs]
        oofs = np.zeros((len(splits), n))
        for (r, f, tr, te), (p, ex) in zip(jobs, outs):
            oofs[r, te] = np.clip(p, 0, 1) if clip else p
            if keep_extra:
                ex = dict(ex); ex["te"] = te; ex["rep"] = r; res[k]["extra"].append(ex)
        res[k]["oof"] = oofs
        res[k]["rep"] = pd.DataFrame([metrics(y, oofs[r]) for r in range(len(splits))])
    return res


def summarise(res, label):
    rows = []
    for k, v in res.items():
        d = v["rep"]
        rows.append(dict(方案=label, 模型=k, **{f"{m}_mean": d[m].mean() for m in ["MAE", "RMSE", "R2", "rho"]},
                         **{f"{m}_sd": d[m].std(ddof=1) for m in ["MAE", "RMSE", "R2", "rho"]}))
    return pd.DataFrame(rows)


def nn_sim(S, splits):
    """for every compound and repeat: maximum Tanimoto similarity to the training compounds of its fold"""
    n = S.shape[0]; out = np.zeros((len(splits), n))
    for r, folds in enumerate(splits):
        for tr, te in folds:
            out[r, te] = S[np.ix_(te, tr)].max(1)
    return out


def perm_test(X, y, kind="ridge", nperm=300, seed=SEED):
    """y-permutation with one 5-fold split; returns observed MAE, null MAEs and P"""
    sp = random_splits(len(y), nrep=1)
    obs = run_cv(X, y, sp, [kind])[kind]["rep"]["MAE"].iloc[0]
    rng = np.random.default_rng(seed); null = []
    for i in range(nperm):
        yp = rng.permutation(y)
        null.append(run_cv(X, yp, sp, [kind])[kind]["rep"]["MAE"].iloc[0])
    null = np.array(null)
    return obs, null, (1 + (null <= obs).sum()) / (nperm + 1)


# ---------------------------------------------------------------------------------------------------------------- main
def main():
    os.makedirs(C.OUT, exist_ok=True)
    b = C.load()
    m = b[b["主分析集"]].reset_index(drop=True)
    S = C.tanimoto(m)
    X = m[C.DESC].values.astype(float)
    groups = m["butina_main_0.6"].values.astype(int)
    R = {"meta": m, "S": S}
    sp_r = random_splits(len(m)); sp_g = group_splits(groups)
    R["splits"] = {"random": sp_r, "group": sp_g}
    R["nnsim"] = {"random": nn_sim(S, sp_r), "group": nn_sim(S, sp_g)}
    tabs = []
    # ---- C1/C2 main comparison
    for ep in ["R", "D"]:
        y = m[ep].values
        for sch, sp in [("random", sp_r), ("group", sp_g)]:
            print("C1/C2", ep, sch, flush=True)
            res = run_cv(X, y, sp, ["baseline", "ridge", "xgb"], keep_extra=True)
            R[(ep, sch)] = res; tabs.append(summarise(res, f"{ep}|{sch}"))
    # sensitivity: cluster cut-offs 0.5 / 0.7 (ridge vs baseline) and S1 endpoints, all 115 compounds
    for cut in ["0.5", "0.7"]:
        spc = group_splits(m[f"butina_main_{cut}"].values.astype(int))
        for ep in ["R", "D"]:
            res = run_cv(X, m[ep].values, spc, ["baseline", "ridge", "xgb"])
            R[(ep, f"group{cut}")] = res; tabs.append(summarise(res, f"{ep}|group{cut}"))
    for ep in ["R_S1", "D_S1"]:
        res = run_cv(X, m[ep].values, sp_r, ["baseline", "ridge", "xgb"])
        R[(ep, "random")] = res; tabs.append(summarise(res, f"{ep}|random"))
    a = b.reset_index(drop=True); Xa = a[C.DESC].values.astype(float)
    spa = random_splits(len(a)); spag = group_splits(a["butina_all_0.6"].values.astype(int))
    for ep in ["R", "D"]:
        for sch, sp in [("random", spa), ("group", spag)]:
            res = run_cv(Xa, a[ep].values, sp, ["baseline", "ridge", "xgb"])
            R[(ep, f"all115_{sch}")] = res; tabs.append(summarise(res, f"{ep}|all115_{sch}"))
    R["all115"] = a
    perf = pd.concat(tabs, ignore_index=True); R["perf"] = perf
    print(perf.round(3).to_string(), flush=True)
    # ---- permutation tests (ridge, random)
    R["perm"] = {}
    for ep in ["R", "D"]:
        obs, null, p = perm_test(X, m[ep].values, "ridge", nperm=300)
        R["perm"][ep] = dict(obs=obs, null=null, p=p); print("perm", ep, round(obs, 4), round(null.mean(), 4), p, flush=True)
    # ---- class leave-out stress test
    rows = []
    cls = m["新类别"].values
    for ep in ["R", "D"]:
        y = m[ep].values
        for c in C.CLS7:
            te = np.where(cls == c)[0]; tr = np.where(cls != c)[0]
            for k in ["baseline", "ridge", "xgb"]:
                p, _ = fit_predict(k, X[tr], y[tr], X[te]); p = np.clip(p, 0, 1)
                rows.append(dict(终点=ep, 留出类别=c, n=len(te), 模型=k, MAE=np.abs(p - y[te]).mean(),
                                 偏差=(p - y[te]).mean()))
    R["classout"] = pd.DataFrame(rows)
    # ---- C3 ablation (ridge and xgb, same random splits within each subset)
    onehot = pd.get_dummies(m["新类别"]).reindex(columns=C.CLS7, fill_value=0).values.astype(float)
    FS = {"类别": None, "极性": ["TPSA", "HBD", "HBA", "O/C"], "大小/疏水": ["MW", "XLogP", "RotB"],
          "芳香环": ["芳香环"], "8 项描述符": C.DESC, "8 项 + 类别": "D+C", "极性 + 类别": "P+C"}

    def design(sub, onehot_sub, key):
        v = FS[key]
        if v is None:
            return onehot_sub
        if v == "D+C":
            return np.hstack([sub[C.DESC].values.astype(float), onehot_sub])
        if v == "P+C":
            return np.hstack([sub[FS["极性"]].values.astype(float), onehot_sub])
        return sub[v].values.astype(float)

    abl = []
    subsets = {"主分析集": np.ones(len(m), bool), "去酸类": m["新类别"].values != "酸类",
               "有实验沸点": m["bp_exp"].notna().values}
    R["abl_oof"] = {}
    for sname, mask in subsets.items():
        sub = m[mask].reset_index(drop=True); oh = onehot[mask]
        oh = oh[:, oh.sum(0) > 0]
        sp = random_splits(len(sub))
        keys = list(FS)
        if sname == "有实验沸点":
            keys = keys + ["沸点", "8 项 + 沸点", "极性 + 沸点"]
        for ep in ["R", "D"]:
            y = sub[ep].values
            base = run_cv(np.zeros((len(sub), 1)), y, sp, ["baseline"])["baseline"]["rep"]
            for key in keys:
                if key == "沸点":
                    Xs = sub[["bp_exp"]].values.astype(float)
                elif key == "8 项 + 沸点":
                    Xs = sub[C.DESC + ["bp_exp"]].values.astype(float)
                elif key == "极性 + 沸点":
                    Xs = sub[FS["极性"] + ["bp_exp"]].values.astype(float)
                else:
                    Xs = design(sub, oh, key)
                for kind in ["ridge", "xgb"]:
                    res = run_cv(Xs, y, sp, [kind])[kind]
                    d = res["rep"]
                    R["abl_oof"][(sname, ep, key, kind)] = res["oof"].mean(0)
                    abl.append(dict(子集=sname, n=len(sub), 终点=ep, 特征组=key, 模型=kind,
                                    MAE=d["MAE"].mean(), MAE_sd=d["MAE"].std(ddof=1), R2=d["R2"].mean(),
                                    R2_sd=d["R2"].std(ddof=1), 基线MAE=base["MAE"].mean(),
                                    相对基线降低=(1 - d["MAE"] / base["MAE"]).mean() * 100))
            print("abl", sname, ep, flush=True)
    R["abl"] = pd.DataFrame(abl)
    # ---- standardized ridge coefficients with bootstrap (all 8 descriptors; main set and without acids)
    R["boot"] = {}
    for sname, mask in [("主分析集", np.ones(len(m), bool)), ("去酸类", m["新类别"].values != "酸类")]:
        sub = m[mask]; Xs = StandardScaler().fit_transform(sub[C.DESC].values.astype(float))
        for ep in ["R", "D"]:
            y = sub[ep].values
            gs = GridSearchCV(Ridge(), {"alpha": ALPHAS}, cv=KFold(5, shuffle=True, random_state=SEED),
                              scoring="neg_mean_absolute_error").fit(Xs, y)
            al = gs.best_params_["alpha"]; coef0 = Ridge(alpha=al).fit(Xs, y).coef_
            rng = np.random.default_rng(SEED); bs = []
            for i in range(2000):
                ii = rng.integers(0, len(y), len(y))
                bs.append(Ridge(alpha=al).fit(Xs[ii], y[ii]).coef_)
            bs = np.array(bs)
            R["boot"][(sname, ep)] = dict(alpha=al, coef=coef0, lo=np.percentile(bs, 2.5, 0), hi=np.percentile(bs, 97.5, 0),
                                          same_sign=(np.sign(bs) == np.sign(coef0)).mean(0))
    # ---- Spearman correlations of each descriptor with R and D (main, without acids, esters only), BH within block
    rows = []
    for sname, mask in [("主分析集", np.ones(len(m), bool)), ("去酸类", m["新类别"].values != "酸类"),
                        ("酯类", m["新类别"].values == "酯类"), ("醇类", m["新类别"].values == "醇类")]:
        sub = m[mask]
        for ep in ["R", "D"]:
            ps = []
            for dsc in C.DESC + ["bp_exp"]:
                ok = sub[dsc].notna()
                if sub.loc[ok, dsc].nunique() < 3:
                    rows.append(dict(子集=sname, n=int(ok.sum()), 终点=ep, 描述符=dsc, rho=np.nan, P=np.nan)); continue
                r_ = stats.spearmanr(sub.loc[ok, dsc], sub.loc[ok, ep])
                rows.append(dict(子集=sname, n=int(ok.sum()), 终点=ep, 描述符=dsc, rho=r_.correlation, P=r_.pvalue))
    cor = pd.DataFrame(rows)
    from statsmodels.stats.multitest import multipletests
    cor["q"] = np.nan
    for (s_, e_), g in cor.groupby(["子集", "终点"]):
        ok = g["P"].notna()
        if ok.sum():
            cor.loc[g.index[ok], "q"] = multipletests(g.loc[ok, "P"], method="fdr_bh")[1]
    R["cor"] = cor
    # ---- TPSA threshold: fixed 37 A2 rule and in-fold threshold search (balanced accuracy) for the late label
    lab = m["late"].values; tp = m["TPSA"].values
    fixed = dict(TP=int(((tp >= 37) & (lab == 1)).sum()), FN=int(((tp < 37) & (lab == 1)).sum()),
                 FP=int(((tp >= 37) & (lab == 0)).sum()), TN=int(((tp < 37) & (lab == 0)).sum()))
    fixed["sens"] = fixed["TP"] / (fixed["TP"] + fixed["FN"]); fixed["spec"] = fixed["TN"] / (fixed["TN"] + fixed["FP"])
    thr_rows = []
    cands = np.unique(tp)
    for r in range(NREP):
        skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=SEED + r)
        for tr, te in skf.split(tp, lab):
            best, bt = -1, None
            for t in cands:
                pr = tp[tr] >= t
                sens = (pr & (lab[tr] == 1)).sum() / max(1, (lab[tr] == 1).sum())
                spec = (~pr & (lab[tr] == 0)).sum() / max(1, (lab[tr] == 0).sum())
                if (sens + spec) / 2 > best + 1e-12:
                    best, bt = (sens + spec) / 2, t
            pr = tp[te] >= bt
            thr_rows.append(dict(rep=r, thr=bt,
                                 sens=(pr & (lab[te] == 1)).sum() / max(1, (lab[te] == 1).sum()),
                                 spec=(~pr & (lab[te] == 0)).sum() / max(1, (lab[te] == 0).sum())))
    R["thr_fixed"] = fixed; R["thr_fold"] = pd.DataFrame(thr_rows)
    # logistic baseline for the binary label kept as an auxiliary reference (same random splits, AUC)
    from sklearn.metrics import roc_auc_score
    aucs = []
    for r, folds in enumerate(sp_r):
        oof = np.zeros(len(m))
        for tr, te in folds:
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
            clf.fit(X[tr], lab[tr]); oof[te] = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(lab, oof))
    R["late_auc"] = np.array(aucs)
    with open(os.path.join(C.OUT, "models.pkl"), "wb") as fh:
        pickle.dump(R, fh)
    perf.to_csv(os.path.join(C.OUT, "perf.csv"), index=False, encoding="utf-8-sig")
    R["abl"].to_csv(os.path.join(C.OUT, "ablation.csv"), index=False, encoding="utf-8-sig")
    R["classout"].to_csv(os.path.join(C.OUT, "classout.csv"), index=False, encoding="utf-8-sig")
    cor.to_csv(os.path.join(C.OUT, "cor.csv"), index=False, encoding="utf-8-sig")
    print(R["abl"].round(3).to_string()); print(R["classout"].round(3).to_string()); print(fixed)
    print(R["thr_fold"].describe().round(3)); print("late AUC", R["late_auc"].mean(), R["late_auc"].std())


if __name__ == "__main__":
    main()
