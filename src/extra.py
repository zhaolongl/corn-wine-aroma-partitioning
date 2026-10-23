"""JAR merged paper, sixth version: supplementary sensitivity analyses for C0-C3.
(1) compounds detected in at least two fractions (single-fraction detections give R = 0 or 1 by construction and are
    sensitive to the detection limit);  (2) non-acid compounds with an experimental boiling point;
(3) fold-wise TreeSHAP importance of the XGBoost D model (random scheme) and its rank stability;
(4) detection breadth versus abundance;  (5) Ridge D model trained on all 96 compounds applied to the in-domain
    co-fermented-wine-only compounds (candidate predictions without measured labels).
Run: python extra.py"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core as C                                                                 # noqa: E402
import models as M                                                               # noqa: E402


def main():
    b = C.load(); m = b[b["主分析集"]].reset_index(drop=True)
    out = {}
    tabs = []
    # (1) multi-fraction subset
    mf = m[m["n_det"] >= 2].reset_index(drop=True)
    sp = M.random_splits(len(mf))
    for ep in ["R", "D"]:
        res = M.run_cv(mf[C.DESC].values.astype(float), mf[ep].values, sp, ["baseline", "ridge", "xgb"])
        tabs.append(M.summarise(res, f"{ep}|多馏分检出（n = {len(mf)}）"))
    # (2) non-acid compounds with bp
    na = m[(m["新类别"] != "酸类") & m["bp_exp"].notna()].reset_index(drop=True)
    sp = M.random_splits(len(na))
    rows = []
    for key, cols in [("沸点", ["bp_exp"]), ("极性", ["TPSA", "HBD", "HBA", "O/C"]), ("8 项描述符", C.DESC),
                      ("8 项 + 沸点", C.DESC + ["bp_exp"]), ("极性 + 沸点", ["TPSA", "HBD", "HBA", "O/C", "bp_exp"])]:
        for ep in ["D"]:
            base = M.run_cv(np.zeros((len(na), 1)), na[ep].values, sp, ["baseline"])["baseline"]["rep"]
            for kind in ["ridge", "xgb"]:
                d = M.run_cv(na[cols].values.astype(float), na[ep].values, sp, [kind])[kind]["rep"]
                rows.append(dict(子集=f"去酸且有沸点（n = {len(na)}）", 终点=ep, 特征组=key, 模型=kind, MAE=d["MAE"].mean(),
                                 MAE_sd=d["MAE"].std(ddof=1), R2=d["R2"].mean(), 基线MAE=base["MAE"].mean(),
                                 相对基线降低=(1 - d["MAE"] / base["MAE"]).mean() * 100))
    out["nonacid_bp"] = pd.DataFrame(rows)
    out["perf_extra"] = pd.concat(tabs, ignore_index=True)
    # (3) fold-wise TreeSHAP of the XGBoost D model
    R = pickle.load(open(os.path.join(C.OUT, "models.pkl"), "rb"))
    ex = R[("D", "random")]["xgb"]["extra"]
    imp = np.array([np.abs(e["shap_te"]).mean(0) for e in ex])
    ranks = np.array([(-v).argsort().argsort() + 1 for v in imp])
    out["shap_D"] = pd.DataFrame({"描述符": C.DESC, "mean|SHAP|": imp.mean(0), "SD": imp.std(0, ddof=1),
                                  "前3频率": (ranks <= 3).mean(0), "中位秩": np.median(ranks, 0)})
    sv = np.zeros((len(m), len(C.DESC))); cnt = np.zeros(len(m))
    for e in ex:
        sv[e["te"]] += e["shap_te"]; cnt[e["te"]] += 1
    out["shap_D_point"] = sv / cnt[:, None]
    # (4) detection breadth vs abundance
    tot = (b[C.FR].values * C.VOL).sum(1)
    out["det"] = dict(rho=stats.spearmanr(b["n_det"], tot).correlation, P=stats.spearmanr(b["n_det"], tot).pvalue,
                      single=int((m["n_det"] == 1).sum()), R0_single=int(((m["R"] == 0) & (m["n_det"] == 1)).sum()),
                      R0=int((m["R"] == 0).sum()), R1=int((m["R"] == 1).sum()),
                      R1_single=int(((m["R"] == 1) & (m["n_det"] == 1)).sum()),
                      med_head_conc_R0=float(m.loc[(m["R"] == 0) & (m["p_头"] == 1), "头馏分_mg/L"].median()),
                      med_head_conc_other=float(m.loc[m["头馏分_mg/L"] > 0, "头馏分_mg/L"].median()))
    # (5) wine-only in-domain compounds: candidate D predictions
    w = pd.read_csv(os.path.join(C.CFG.RAW, "cofermented_wine_compounds.csv"))
    X = m[C.DESC].values.astype(float)
    gs = GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": M.ALPHAS},
                      cv=KFold(5, shuffle=True, random_state=M.SEED), scoring="neg_mean_absolute_error").fit(X, m["D"].values)
    wo = w[(~w["共有"]) & (w["适用域"] == "域内")].copy()
    Xw = wo[C.DESC].values.astype(float)
    wo["D_pred"] = np.clip(gs.predict(Xw), 0, 1)
    S = np.load(os.path.join(C.DER, "tanimoto115.npy"))
    out["wine_pred"] = wo[["编号", "化合物", "CID", "类别_统一", "TPSA", "预测类型", "一致性", "h", "D_pred", "OAVmax"]]
    out["wine_alpha"] = gs.best_params_["ridge__alpha"]
    with open(os.path.join(C.OUT, "extra.pkl"), "wb") as fh:
        pickle.dump(out, fh)
    pd.set_option("display.width", 250)
    print(out["perf_extra"].round(3).to_string()); print(out["nonacid_bp"].round(3).to_string())
    print(out["shap_D"].round(3).to_string()); print(out["det"])
    print(len(wo), wo[["化合物", "预测类型", "D_pred", "OAVmax"]].sort_values("OAVmax", ascending=False).head(8).round(3).to_string())


if __name__ == "__main__":
    main()
