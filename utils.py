
# Contains data generation, model zoo, training, and SHAP helper functions.

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from math import ceil
from scipy.ndimage import gaussian_filter
RNG = 123
# Optional libraries: xgboost and gplearn
try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from gplearn.genetic import SymbolicRegressor
    _HAS_GPLEARN = True
except ImportError:
    _HAS_GPLEARN = False
    
# ==== ONE MIXED DATASET: correlated blocks + independent features (spatial maps) ====


rng = np.random.default_rng(123)

def _smooth(Z, sigma):
    return gaussian_filter(Z, sigma=sigma, mode="reflect") if sigma and sigma>0 else Z

def _standardize(A, eps=1e-12):
    A = A.astype(float)
    return (A - A.mean()) / (A.std() + eps)

def _flatten(Z):  # (ny,nx) -> (ny*nx,)
    return Z.reshape(-1)

def _proj_out(v, basis):
    """Remove components of v along all basis columns (least-squares)."""
    if basis.size == 0:
        return v
    # basis: shape (n, k); v: shape (n,)
    # Projection matrix: B(B^+ v), with B^+ = (B^T B)^{-1} B^T
    BtB = basis.T @ basis
    coef = np.linalg.pinv(BtB) @ (basis.T @ v)
    return v - basis @ coef

def make_spatial_mixed(
    ny=28, nx=28, d=7,
    groups=([[0,1,2],[3,4]]),      # two correlated blocks
    rhos=(0.85, 0.75),             # within-group rhos (match len(groups))
    independent_idxs=(5,6),        # these will be decorrelated from group fields
    spatial_sigma=3.0,
    target_kind="quad_sin",
    noise_sigma=0.25,
    seed=42
):
    rng = np.random.default_rng(seed)
    assert len(groups) == len(rhos), "rhos must match number of groups"
    assert max([i for g in groups for i in g] + list(independent_idxs)) < d, "index out of range"

    # 1) Build one smooth shared field per group
    ny, nx = int(ny), int(nx)
    group_shared = []
    for _ in groups:
        gfield = _standardize(_smooth(rng.normal(size=(ny,nx)), spatial_sigma))
        group_shared.append(gfield)

    # 2) Construct each feature
    X_maps = [None]*d
    for gid, g in enumerate(groups):
        a = np.sqrt(max(0.0, rhos[gid])); b = np.sqrt(max(0.0, 1.0 - rhos[gid]))
        G = group_shared[gid]
        for j in g:
            Ej = _standardize(_smooth(rng.normal(size=(ny,nx)), spatial_sigma))
            X_maps[j] = _standardize(a*G + b*Ej)

    # 3) Independent features: make sure they are spatially smooth BUT orthogonal
    #    to *all* group shared fields (to keep correlations ≈ 0)
    if len(independent_idxs) > 0:
        # Build basis from all group shared fields (flattened)
        basis = np.column_stack([_flatten(_standardize(gs)) for gs in group_shared]) if group_shared else np.empty((ny*nx, 0))
        for j in independent_idxs:
            raw = _flatten(_standardize(_smooth(rng.normal(size=(ny,nx)), spatial_sigma)))
            clean = _proj_out(raw, basis)          # remove shared components
            clean = (clean - clean.mean()) / (clean.std() + 1e-12)
            X_maps[j] = clean.reshape(ny, nx)

    # 4) Any remaining feature indices not in groups or independent: make plain smooth random
    covered = set(independent_idxs) | set([k for g in groups for k in g])
    for j in range(d):
        if X_maps[j] is None:
            X_maps[j] = _standardize(_smooth(rng.normal(size=(ny,nx)), spatial_sigma))

    X_stack = np.stack(X_maps, axis=2)  # (ny,nx,d)

    # 5) Ground-truth target (same forms as before)
    def _target(Xs, kind="quad_sin"):
        if kind == "linear":
            w = np.linspace(1.0, 0.2, Xs.shape[2])
            return np.tensordot(Xs, w, axes=([2],[0]))
        if kind == "quad_sin":
            X = Xs if Xs.shape[2] >= 3 else np.pad(Xs, ((0,0),(0,0),(0,3-Xs.shape[2])), mode='constant')
            return 1.5*(X[:,:,0]**2) + np.sin(X[:,:,1]) + 0.5*X[:,:,2]
        if kind == "interaction":
            X = Xs if Xs.shape[2] >= 4 else np.pad(Xs, ((0,0),(0,0),(0,4-Xs.shape[2])), mode='constant')
            return X[:,:,0]*X[:,:,1] + 0.7*X[:,:,2] - 0.5*X[:,:,3]
        if kind == "symbolic":
            X = Xs if Xs.shape[2] >= 3 else np.pad(Xs, ((0,0),(0,0),(0,3-Xs.shape[2])), mode='constant')
            return np.exp(-X[:,:,0]) * np.cos(X[:,:,1]) + 0.2*X[:,:,2]
        return Xs[:,:,0]

    f_map = _target(X_stack, target_kind)
    y_map = f_map + rng.normal(0, noise_sigma, size=(ny, nx))

    # 6) DataFrame (flatten) + coords
    cols = [f"x{i}" for i in range(d)]
    df = pd.DataFrame(
        np.column_stack([X_stack[:,:,i].ravel() for i in range(d)] + [y_map.ravel()]),
        columns=cols + ["y"]
    )
    rr, cc = np.indices((ny, nx)); df["row"] = rr.ravel(); df["col"] = cc.ravel()

    meta = dict(n=ny*nx, d=d, ny=ny, nx=nx, groups=groups, rhos=rhos,
                independent=list(independent_idxs), spatial_sigma=spatial_sigma,
                target_kind=target_kind)
    return df, meta, X_stack, y_map

def plot_maps(X_stack, y_map, title_prefix="MIXED • "):
    ny, nx, d = X_stack.shape
    k = d + 1
    ncols = 4
    nrows = int(ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2*ncols, 2.8*nrows))
    axes = np.atleast_2d(axes).ravel()
    for j in range(d):
        im = axes[j].imshow(X_stack[:,:,j], origin="upper", cmap="viridis")
        axes[j].set_title(f"{title_prefix}x{j}")
        axes[j].set_xticks([]); axes[j].set_yticks([])
        plt.colorbar(im, ax=axes[j], fraction=0.046, pad=0.04)
    im = axes[d].imshow(y_map, origin="upper", cmap="viridis")
    axes[d].set_title(f"{title_prefix}y"); axes[d].set_xticks([]); axes[d].set_yticks([])
    plt.colorbar(im, ax=axes[d], fraction=0.046, pad=0.04)
    for ax in axes[k:]: ax.axis("off")
    plt.tight_layout(); plt.show()

def correlation_heatmap(df, title="Feature×Feature correlation"):
    xcols = [c for c in df.columns if c.startswith("x")]
    C = df[xcols].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(5.4,4.8))
    im = ax.imshow(C, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(xcols))); ax.set_yticks(range(len(xcols)))
    ax.set_xticklabels(xcols, rotation=45, ha="right"); ax.set_yticklabels(xcols)
    ax.set_title(title); plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.show()




def build_model_zoo(random_state=RNG):
    """Return dict: name -> (estimator_or_pipeline, param_dist)."""
    zoo = {}

    # --- Linear family ---
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
    from sklearn.svm import SVR
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    import numpy as np

    zoo["Linear"] = (
        Pipeline([("scaler", StandardScaler()), ("m", LinearRegression())]),
        {"scaler": [StandardScaler(), RobustScaler(), MinMaxScaler()]}
    )
    zoo["Ridge"] = (
        Pipeline([("scaler", StandardScaler()), ("m", Ridge(random_state=random_state))]),
        {"scaler":[StandardScaler(), RobustScaler(), MinMaxScaler()],
         "m__alpha": np.logspace(-4, 3, 50)}
    )
    zoo["Lasso"] = (
        Pipeline([("scaler", StandardScaler()), ("m", Lasso(random_state=random_state, max_iter=10000))]),
        {"scaler":[StandardScaler(), RobustScaler(), MinMaxScaler()],
         "m__alpha": np.logspace(-4, 1, 50)}
    )
    zoo["ElasticNet"] = (
        Pipeline([("scaler", StandardScaler()), ("m", ElasticNet(random_state=random_state, max_iter=10000))]),
        {"scaler":[StandardScaler(), RobustScaler(), MinMaxScaler()],
         "m__alpha": np.logspace(-4, 1, 50),
         "m__l1_ratio": np.linspace(0.05, 0.95, 19)}
    )

    # --- Kernel / neighbors ---
    zoo["SVR-RBF"] = (
        Pipeline([("scaler", StandardScaler()), ("m", SVR(kernel="rbf"))]),
        {"scaler":[StandardScaler(), RobustScaler(), MinMaxScaler()],
         "m__C":     np.logspace(-2, 3, 30),
         "m__gamma": np.logspace(-4, 1, 30),
         "m__epsilon": np.logspace(-3, 0, 15)}
    )
    zoo["KNN"] = (
        Pipeline([("scaler", StandardScaler()), ("m", KNeighborsRegressor())]),
        {"scaler":[StandardScaler(), RobustScaler(), MinMaxScaler()],
         "m__n_neighbors": list(range(3, 41, 2)),
         "m__weights": ["uniform", "distance"],
         "m__p": [1, 2]}
    )

    # --- Trees / ensembles ---
    zoo["RF"] = (
        RandomForestRegressor(n_estimators=500, random_state=random_state, n_jobs=-1),
        {"max_depth":[None]+list(range(3,31,3)),
         "min_samples_split":[2,5,10],
         "min_samples_leaf":[1,2,4],
         "max_features":["sqrt","log2",None]}
    )
    zoo["ExtraTrees"] = (
        ExtraTreesRegressor(n_estimators=600, random_state=random_state, n_jobs=-1),
        {"max_depth":[None]+list(range(3,31,3)),
         "min_samples_split":[2,5,10],
         "min_samples_leaf":[1,2,4],
         "max_features":["sqrt","log2",None]}
    )
    zoo["GBDT"] = (
        GradientBoostingRegressor(random_state=random_state),
        {"n_estimators":[200,300,500,800],
         "learning_rate": np.logspace(-3, -0.3, 10),
         "max_depth":[2,3,4,5],
         "subsample":[0.6,0.8,1.0],
         "min_samples_leaf":[1,2,4]}
    )
    zoo["HGBDT"] = (
        HistGradientBoostingRegressor(random_state=random_state),
        {"learning_rate": np.logspace(-3, -0.3, 10),
         "max_depth":[None,4,6,8,12],
         "l2_regularization": np.logspace(-6, 1, 10)}
    )

    # --- MLP ---
    zoo["MLP"] = (
        Pipeline([("scaler", StandardScaler()),
                  ("m", MLPRegressor(max_iter=800, early_stopping=True, random_state=random_state))]),
        {"scaler":[StandardScaler(), RobustScaler()],
         "m__hidden_layer_sizes":[(64,64),(128,64),(128,128),(256,128),(64,64,32)],
         "m__activation":["relu","tanh"],
         "m__alpha": np.logspace(-6, -2, 10),
         "m__learning_rate_init": np.logspace(-4, -2, 6),
         "m__batch_size":[32,64,128]}
    )

    # --- Optional XGBoost ---
    try:
        from xgboost import XGBRegressor
        zoo["XGB"] = (
            XGBRegressor(n_estimators=1200, random_state=random_state, n_jobs=-1,
                         tree_method="hist", reg_lambda=1.0, reg_alpha=0.0),
            {"max_depth":[3,4,5,6,8],
             "learning_rate": np.logspace(-3, -0.2, 12),
             "subsample":[0.6,0.8,1.0],
             "colsample_bytree":[0.6,0.8,1.0],
             "min_child_weight":[1,5,10],
             "gamma":[0.0,0.1,0.2]}
        )
    except Exception:
        pass

   # --- Symbolic Regression (gplearn) ---
    try:
        from gplearn.genetic import SymbolicRegressor

        # No scaler; it works on raw features. Ensure float64 upstream.
        zoo["SymReg"] = (
            SymbolicRegressor(
                population_size=1200,
                generations=20,
                tournament_size=20,
                stopping_criteria=0.0,
                function_set=("add","sub","mul","div","sin","cos","sqrt","log"),
                metric="mean absolute error",
                p_crossover=0.6,
                p_subtree_mutation=0.2,
                p_hoist_mutation=0.05,
                p_point_mutation=0.15,
                parsimony_coefficient=1e-4,
                max_samples=0.9,
                n_jobs=-1,
                verbose=0,
                random_state=random_state,
            ),
            {
                "population_size": [600, 900, 1200, 1500],
                "generations": [15, 20, 30],
                "tournament_size": [10, 20, 30],
                "p_crossover": np.linspace(0.5, 0.9, 5),
                "p_subtree_mutation": np.linspace(0.1, 0.4, 4),
                "p_hoist_mutation": [0.02, 0.05, 0.1],
                "p_point_mutation": np.linspace(0.05, 0.25, 5),
                "parsimony_coefficient": np.logspace(-6, -2, 9),
                "function_set": [
                    ("add","sub","mul","div","sin","cos","sqrt","log"),
                    ("add","sub","mul","div","sin","cos","sqrt"),
                    ("add","sub","mul","div","sqrt","log"),
                ],
            },
        )
    except ImportError:
        # gplearn is not installed; skip symbolic regression
        pass

    return zoo

def run_model_search(df, *, feature_prefix="x", ycol="y",
                     test_size=0.2, random_state=RNG,
                     cv_splits=5, n_iter=40, per_model_n_iter=None,
                     n_jobs=-1, verbose=1,
                     model_names=None):
    """
    per_model_n_iter: optional dict like {"SymReg": 15, "MLP": 50} to override n_iter per model.
    model_names: optional iterable of model names (keys from build_model_zoo())
                 to restrict which models are evaluated. If None, all models
                 in the zoo are used.
    """
    X, y, xcols = make_Xy(df, feature_prefix, ycol)
    # gplearn prefers float64
    X = X.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)

    from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=random_state)

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    zoo = build_model_zoo(random_state=random_state)

    # 🔹 NEW: optionally restrict to a subset of models
    if model_names is not None:
        model_names = list(model_names)
        zoo = {name: zoo[name] for name in model_names if name in zoo}

    results, best_models = [], {}

    for name, (estimator, param_dist) in zoo.items():
        this_iter = per_model_n_iter.get(name, n_iter) if per_model_n_iter else n_iter

        rs = RandomizedSearchCV(
            estimator,
            param_distributions=param_dist,
            n_iter=this_iter,
            cv=cv,
            scoring="r2",
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            refit=True,
        )
        rs.fit(Xtr, ytr)

        yhat = rs.best_estimator_.predict(Xte)
        s = _scores(yte, yhat)
        s.update(dict(model=name, best_params=rs.best_params_))
        results.append(s)
        best_models[name] = rs.best_estimator_

        print(
            f"\n{name}: best CV R2={rs.best_score_:.3f} | TEST "
            f"{{'R2':{s['R2']:.3f}, 'RMSE':{s['RMSE']:.3f}, 'MAE':{s['MAE']:.3f}}}"
        )

    leaderboard = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
    return leaderboard, best_models, (Xtr, Xte, ytr, yte), xcols


def _scores(y_true, y_pred):
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    return dict(
        R2=float(r2_score(y_true, y_pred)),
        RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        MAE=float(mean_absolute_error(y_true, y_pred)),
    )

def plot_best(model, Xte, yte, title="Best model: Predictions vs Truth"):
    yhat = model.predict(Xte)
    s = _scores(yte, yhat)
    plt.figure(figsize=(5.4, 4.6))
    plt.scatter(yte, yhat, s=14, alpha=0.8)
    lims = [min(yte.min(), yhat.min()), max(yte.max(), yhat.max())]
    plt.plot(lims, lims, lw=1)
    plt.xlabel("True y"); plt.ylabel("Predicted y")
    plt.title(f"{title}\nR²={s['R2']:.3f}  RMSE={s['RMSE']:.3f}  MAE={s['MAE']:.3f}")
    plt.tight_layout(); plt.show()

def make_Xy(df, feature_prefix="x", ycol="y"):
    """
    Split a DataFrame into (X, y, feature_names).

    feature_prefix can be:
    - a string prefix (default): "x" selects columns starting with "x"
    - a list or tuple of specific feature names: ["x0","x3"]
    """
    # Case 1: string prefix (existing behavior)
    if isinstance(feature_prefix, str):
        xcols = [c for c in df.columns if c.startswith(feature_prefix)]

    # Case 2: explicit list/tuple of feature names
    elif isinstance(feature_prefix, (list, tuple)):
        xcols = list(feature_prefix)

    else:
        raise TypeError(
            "feature_prefix must be a string prefix or a list/tuple of column names."
        )

    X = df[xcols].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)
    return X, y, xcols


# ==== SHAP for best architecture of each model ====
import numpy as np, shap, matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

# --- Helpers to unwrap pipelines and wrap predict so SHAP sees RAW X ---
def _unwrap_estimator(est):
    """Return (model, scaler or None). Works for plain estimators or Pipelines with ('scaler','m')."""
    if isinstance(est, Pipeline):
        scaler = est.named_steps.get("scaler", None)
        model  = est.named_steps.get("m", None)
        # allow pipelines without 'm' key (rare)
        if model is None:
            # last step as model
            last_key = list(est.named_steps.keys())[-1]
            model = est.named_steps[last_key]
            # assume first step is scaler if it looks like one
            first_key = list(est.named_steps.keys())[0]
            if "scale" in first_key:
                scaler = est.named_steps[first_key]
        return model, scaler
    return est, None

def _predict_fn(model, scaler=None):
    """Predict on RAW X; if scaler is present, apply it inside the lambda."""
    if scaler is None:
        return model.predict
    return lambda XR: model.predict(scaler.transform(XR))

# --- Correlation-based feature clustering for clustered bars ---
def _build_feature_clustering(X_train, y_train=None):
    # SHAP supports hclust(X[, y]); fallback to correlation-only if needed
    try:
        return shap.utils.hclust(X_train, y_train)
    except TypeError:
        return shap.utils.hclust(X_train, metric="correlation")

def _to_explanation(shap_values, X_for_plot, feature_names):
    sv = shap_values[0] if isinstance(shap_values, list) else shap_values
    sv = np.asarray(sv)                               # (n_samples, n_features)
    base = np.zeros(sv.shape[0], dtype=float)         # bar() ignores base_values
    return shap.Explanation(values=sv, base_values=base, data=X_for_plot,
                            feature_names=feature_names)

def _clustered_bar(explanation, linkage, title, cutoff=None):
    """Try SHAP's clustered bar; robust manual fallback that groups mean|SHAP| by clusters."""
    drew = False
    try:
        fig, ax = plt.subplots(figsize=(6,4))
        if cutoff is None:
            shap.plots.bar(explanation, max_display=12, clustering=linkage)
        else:
            shap.plots.bar(explanation, max_display=12, clustering=linkage, clustering_cutoff=float(cutoff))
        drew = (len(ax.patches) > 0) or (len(ax.containers) > 0)
        if drew:
            ax.set_title(title); plt.tight_layout(); plt.show()
        else:
            plt.close(fig)
    except Exception:
        try: plt.close(fig)
        except Exception: pass
        drew = False

    if not drew:
        # manual fallback
        from scipy.cluster.hierarchy import fcluster, leaves_list
        sv = explanation.values
        mean_abs = np.mean(np.abs(sv), axis=0)
        if cutoff is None:
            order = leaves_list(linkage)
            vals  = mean_abs[order]
            labels= [explanation.feature_names[i] for i in order]
        else:
            clusters = fcluster(linkage, t=float(cutoff), criterion="distance")
            groups = {}
            for j, cid in enumerate(clusters):
                groups.setdefault(int(cid), []).append(j)
            vals, labels = [], []
            for cid, idxs in groups.items():
                vals.append(float(mean_abs[idxs].sum()))
                labels.append(explanation.feature_names[idxs[0]] if len(idxs)==1
                              else explanation.feature_names[idxs[0]]+f" (+{len(idxs)-1})")
            order = np.argsort(vals)[::-1]
            vals  = np.array(vals)[order]
            labels= [labels[i] for i in order]

        plt.figure(figsize=(6, max(3, 0.35*len(labels))))
        y = np.arange(len(labels))
        plt.barh(y, vals)
        plt.yticks(y, labels)
        plt.gca().invert_yaxis()
        plt.xlabel("mean |SHAP| (grouped)")
        plt.title(title)
        plt.tight_layout(); plt.show()

def shap_best_for_all(best_models, splits, xcols, *, nsamples_kernel=100, do_clustered_bar=True):
    """
    For each best model:
      - choose the right SHAP explainer
      - draw summary plot (vs RAW X)
      - optionally draw clustered bar (correlation-aware grouping)
    """
    Xtr, Xte, ytr, yte = splits

    # Build one linkage from the training data (RAW X) for consistent grouping
    linkage = _build_feature_clustering(Xtr, ytr) if do_clustered_bar else None

    for name, est in best_models.items():
        model, scaler = _unwrap_estimator(est)
        cls = model.__class__.__name__.lower()

        print(f"\n=== {name} — {model.__class__.__name__} ===")

        # Choose explainer
        try:
            if any(k in cls for k in ["linearregression","ridge","lasso","elasticnet"]):
                # Work in scaled space for the model, plot vs RAW X
                Xtr_s = scaler.fit_transform(Xtr) if scaler is not None else Xtr
                Xte_s = scaler.transform(Xte)     if scaler is not None else Xte
                explainer = shap.LinearExplainer(model, Xtr_s, feature_names=xcols)
                sv = explainer.shap_values(Xte_s)
                shap.summary_plot(sv, Xte, feature_names=xcols, show=False)
                plt.title(f"{name} — SHAP summary (LinearExplainer)"); plt.tight_layout(); plt.show()

            elif any(k in cls for k in ["randomforest","extratrees","gradientboosting","histgradientboosting","xgbregressor"]):
                # Trees: TreeExplainer, interventional (independent) by default
                fe_pert = "interventional"
                explainer = shap.TreeExplainer(model, feature_names=xcols, feature_perturbation=fe_pert, model_output="raw")
                sv = explainer.shap_values(Xte)
                shap.summary_plot(sv, Xte, feature_names=xcols, show=False)
                plt.title(f"{name} — SHAP summary (TreeExplainer)"); plt.tight_layout(); plt.show()

            else:
                # SVR, KNN, MLP, SymReg → KernelExplainer (black-box, independent assumption)
                f = _predict_fn(model, scaler=scaler)
                bg = Xtr if len(Xtr) <= 200 else Xtr[np.random.choice(len(Xtr), 200, replace=False)]
                explainer = shap.KernelExplainer(f, bg)
                sv = explainer.shap_values(Xte, nsamples=nsamples_kernel)
                shap.summary_plot(sv, Xte, feature_names=xcols, show=False)
                plt.title(f"{name} — SHAP summary (KernelExplainer)"); plt.tight_layout(); plt.show()

            # Clustered bar (optional)
            if do_clustered_bar:
                ex_bar = _to_explanation(sv, Xte, xcols)
                _clustered_bar(ex_bar, linkage, title=f"{name} — Clustered SHAP bar", cutoff=1.0)

        except Exception as e:
            print(f"SHAP failed for {name}: {e}")




# ==== SCALING ABLATION: predictions & SHAP (one model) ====
import numpy as np, pandas as pd, matplotlib.pyplot as plt, shap
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import RandomizedSearchCV, KFold

# --- utilities ---
def _scores(y_true, y_pred):
    return dict(
        R2=float(r2_score(y_true, y_pred)),
        RMSE=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        MAE=float(mean_absolute_error(y_true, y_pred)),
    )

def _predict_fn(pipe):
    # KernelExplainer will pass RAW X; pipeline handles scaling internally.
    return lambda XR: pipe.predict(XR)

def _build_linkage(X_train, y_train=None):
    try:
        return shap.utils.hclust(X_train, y_train)
    except TypeError:
        return shap.utils.hclust(X_train, metric="correlation")

def _to_explanation(sv, X_for_plot, feature_names):
    sv = sv[0] if isinstance(sv, list) else sv
    return shap.Explanation(
        values=np.asarray(sv),
        base_values=np.zeros(len(X_for_plot)),
        data=X_for_plot,
        feature_names=feature_names
    )

def _clustered_bar(explanation, linkage, title, cutoff=1.0):
    drew = False
    try:
        fig, ax = plt.subplots(figsize=(6,4))
        shap.plots.bar(explanation, max_display=12, clustering=linkage, clustering_cutoff=float(cutoff))
        drew = len(ax.patches) > 0 or len(ax.containers) > 0
        if drew:
            ax.set_title(title); plt.tight_layout(); plt.show()
        else:
            plt.close(fig)
    except Exception:
        try: plt.close(fig)
        except Exception: pass
        drew = False

    if not drew:
        # robust manual fallback (group mean|SHAP| by clusters)
        from scipy.cluster.hierarchy import fcluster, leaves_list
        sv = explanation.values
        mean_abs = np.mean(np.abs(sv), axis=0)
        clusters = fcluster(linkage, t=float(cutoff), criterion="distance")
        groups = {}
        for j, cid in enumerate(clusters): groups.setdefault(int(cid), []).append(j)
        vals, labels = [], []
        for cid, idxs in groups.items():
            vals.append(float(mean_abs[idxs].sum()))
            labels.append(explanation.feature_names[idxs[0]] if len(idxs)==1
                          else explanation.feature_names[idxs[0]]+f" (+{len(idxs)-1})")
        order = np.argsort(vals)[::-1]; vals = np.array(vals)[order]; labels = [labels[i] for i in order]
        y = np.arange(len(labels))
        plt.figure(figsize=(6, max(3, 0.35*len(labels))))
        plt.barh(y, vals); plt.yticks(y, labels); plt.gca().invert_yaxis()
        plt.xlabel("mean |SHAP| (grouped)"); plt.title(title); plt.tight_layout(); plt.show()

# ============================================================
# SHAP correlation clustering vs. ground-truth groups
# ============================================================

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import adjusted_rand_score
import numpy as np
import matplotlib.pyplot as plt

def shap_corr_clustering_vs_truth(
    shap_values,
    feature_names,
    groups,
    title_prefix="",
    corr_method="pearson",
    n_thresholds=30,
):
    """
    Compare SHAP-based feature clustering to ground-truth feature groups.

    This function:
    1. Builds a ground-truth |corr| matrix from known groups.
    2. Computes a SHAP-based correlation matrix across SHAP columns.
    3. Converts both to distances D = 1 - |corr|.
    4. Runs hierarchical clustering on both distance matrices.
    5. Scans cluster thresholds to compute best Adjusted Rand Index (ARI).
    6. Plots dendrograms + correlation matrices.

    Parameters
    ----------
    shap_values : shap.Explanation or ndarray (n_samples, n_features)
    feature_names : list of str
    groups : list of lists of int
        Example: [[0,1,2], [3,4]]
    title_prefix : str
        Optional prefix for dendrogram titles
    corr_method : {"pearson", "spearman"}
    n_thresholds : int
        Number of thresholds to scan for ARI

    Returns
    -------
    best_ari : float
    best_thresh : float
    """

    # Extract SHAP matrix
    if hasattr(shap_values, "values"):
        sv = np.asarray(shap_values.values)
    else:
        sv = np.asarray(shap_values)

    sv = np.asarray(sv)
    assert sv.ndim == 2, "shap_values must be 2D"
    n_features = sv.shape[1]

    feature_names = list(feature_names)
    assert len(feature_names) == n_features, "feature_names mismatch"

    # ---- Ground-truth correlation matrix ----
    C_true = np.eye(n_features)
    for g in groups:
        for i in g:
            for j in g:
                C_true[i, j] = 1.0

    D_true = 1 - np.abs(C_true)
    np.fill_diagonal(D_true, 0)

    # ---- SHAP-based correlation matrix ----
    if corr_method == "spearman":
        from scipy.stats import spearmanr
        C_shap, _ = spearmanr(sv, axis=0)
    else:
        C_shap = np.corrcoef(sv, rowvar=False)

    C_shap = C_shap[:n_features, :n_features]
    D_shap = 1 - np.abs(C_shap)
    np.fill_diagonal(D_shap, 0)

    # ---- Convert to condensed form ----
    cond_true = squareform(D_true, checks=False)
    cond_shap = squareform(D_shap, checks=False)

    Z_true = linkage(cond_true, method="average")
    Z_shap = linkage(cond_shap, method="average")

    # ---- Scan thresholds for best ARI ----
    max_height = max(Z_true[:, 2].max(), Z_shap[:, 2].max())
    thresholds = np.linspace(0, max_height, n_thresholds)[1:]  # skip 0

    best_ari = -1
    best_thresh = None

    for t in thresholds:
        labels_true = fcluster(Z_true, t, criterion="distance")
        labels_shap = fcluster(Z_shap, t, criterion="distance")
        ari = adjusted_rand_score(labels_true, labels_shap)
        if ari > best_ari:
            best_ari = ari
            best_thresh = t

    print(f"Best ARI between ground-truth and SHAP clustering: {best_ari:.3f}")
    print(f"Threshold at best ARI: {best_thresh:.3f}")

    # ---- Plot dendrograms ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    dendrogram(Z_true, labels=feature_names, ax=axes[0], orientation="top")
    axes[0].set_title(f"{title_prefix}Ground-truth clustering")

    dendrogram(Z_shap, labels=feature_names, ax=axes[1], orientation="top")
    axes[1].set_title(f"{title_prefix}SHAP-based clustering")

    plt.tight_layout()
    plt.show()

    # ---- Plot correlation matrices ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    im0 = axes[0].imshow(C_true, vmin=-1, vmax=1, cmap="coolwarm")
    axes[0].set_title("Ground-truth |corr|")
    axes[0].set_xticks(range(n_features))
    axes[0].set_yticks(range(n_features))
    axes[0].set_xticklabels(feature_names, rotation=90)
    axes[0].set_yticklabels(feature_names)

    im1 = axes[1].imshow(C_shap, vmin=-1, vmax=1, cmap="coolwarm")
    axes[1].set_title("SHAP-derived corr")
    axes[1].set_xticks(range(n_features))
    axes[1].set_yticks(range(n_features))
    axes[1].set_xticklabels(feature_names, rotation=90)
    axes[1].set_yticklabels(feature_names)

    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

    return best_ari, best_thresh


def scaling_ablation_on_model(
    splits, xcols, *,
    model_kind="SVR-RBF",
    use_best_params_from=None,   # pass best_models dict if you want Mode A (fixed params)
    refit_hyperparams=False,     # set True for Mode B (search per scaler)
    nsamples_kernel=80,
    random_state=7
):
    """
    Compare 3 pipelines of the SAME model:
      1) Standardize (mean=0, std=1), 2) MinMax [0,1], 3) Raw (no scaling).
    Plots: metrics table, SHAP summaries, and clustered bars.

    Args:
      splits: (Xtr, Xte, ytr, yte) from your earlier run.
      xcols : feature names list.
      use_best_params_from: dict of {model_name: best_estimator} to pull tuned params (Mode A).
      refit_hyperparams: if True, RandomizedSearchCV per scaler (Mode B).
    """
    assert model_kind in ["SVR-RBF", "Ridge", "MLP"], "Demo supports 'SVR-RBF', 'Ridge', or 'MLP'."

    Xtr, Xte, ytr, yte = splits
    results = []
    variants = []

    # --- Build three pipelines for the chosen model ---
    if model_kind == "SVR-RBF":
        base = SVR(kernel="rbf")
        pipes = {
            "Standardize(0,1)": Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)), ("m", base)]),
            "MinMax[0,1]":      Pipeline([("scaler", MinMaxScaler()), ("m", base)]),
            "Raw":              Pipeline([("m", SVR(kernel="rbf"))])
        }
        # default/fallback hyperparams
        default_params = {"m__C": 10.0, "m__gamma": 0.01, "m__epsilon": 0.1}
        search_space = {
            "m__C": np.logspace(-2, 3, 30),
            "m__gamma": np.logspace(-4, 1, 30),
            "m__epsilon": np.logspace(-3, 0, 15),
        }

    elif model_kind == "Ridge":
        from sklearn.linear_model import Ridge
        base = Ridge(random_state=random_state)
        pipes = {
            "Standardize(0,1)": Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)), ("m", base)]),
            "MinMax[0,1]":      Pipeline([("scaler", MinMaxScaler()), ("m", base)]),
            "Raw":              Pipeline([("m", Ridge(random_state=random_state))])
        }
        default_params = {"m__alpha": 1.0}
        search_space = {"m__alpha": np.logspace(-4, 3, 50)}

    else:  # MLP
        from sklearn.neural_network import MLPRegressor
        base = MLPRegressor(max_iter=800, early_stopping=True, random_state=random_state)
        pipes = {
            "Standardize(0,1)": Pipeline([("scaler", StandardScaler(with_mean=True, with_std=True)), ("m", base)]),
            "MinMax[0,1]":      Pipeline([("scaler", MinMaxScaler()), ("m", base)]),
            "Raw":              Pipeline([("m", MLPRegressor(max_iter=800, early_stopping=True, random_state=random_state))])
        }
        default_params = {"m__hidden_layer_sizes": (128,64), "m__alpha": 1e-4, "m__learning_rate_init": 1e-3}
        search_space = {
            "m__hidden_layer_sizes": [(64,64),(128,64),(128,128)],
            "m__alpha": np.logspace(-6, -2, 7),
            "m__learning_rate_init": np.logspace(-4, -2, 5)
        }

    # Pull tuned params for the chosen model if provided (Mode A)
    tuned = None
    if use_best_params_from is not None:
        # try to find the exact key (e.g., "SVR-RBF", "Ridge", "MLP")
        est = use_best_params_from.get(model_kind)
        if est is None:
            # Some leaderboards keyed like {"SVR-RBF": Pipeline(...)}; otherwise search by class
            for k,v in use_best_params_from.items():
                if model_kind.split("-")[0].lower() in v.__class__.__name__.lower() or model_kind in k:
                    est = v; break
        if est is not None:
            # extract only the "m__*" params so we can set them on each pipeline
            if isinstance(est, Pipeline) and "m" in est.named_steps:
                tuned = {f"m__{k}": v for k,v in est.named_steps["m"].get_params().items()
                         if not isinstance(v, (np.ndarray,))}  # keep simple types
            else:
                tuned = {f"m__{k}": v for k,v in est.get_params().items()
                         if not isinstance(v, (np.ndarray,))}
    # Fallback defaults if no tuned params were found
    if tuned is None:
        tuned = default_params

    # Optional per-scaler refit (Mode B)
    if refit_hyperparams:
        cv = KFold(n_splits=4, shuffle=True, random_state=random_state)
        for name, pipe in pipes.items():
            rs = RandomizedSearchCV(pipe, param_distributions=search_space, n_iter=25, cv=cv,
                                    scoring="r2", n_jobs=-1, random_state=random_state, verbose=0, refit=True)
            rs.fit(Xtr, ytr)
            pipes[name] = rs.best_estimator_
            print(f"{model_kind} @ {name}: re-tuned best CV R2={rs.best_score_:.3f}")
    else:
        # Just set the same model hyperparams on each pipeline (scaler effect only)
        for name, pipe in pipes.items():
            pipe.set_params(**tuned)
            pipe.fit(Xtr, ytr)

    # Build one correlation tree (on full data for stability; change to Xtr if you want strict train-only)
    linkage = _build_linkage(np.vstack([Xtr, Xte]))

    # Evaluate + SHAP
    for name, pipe in pipes.items():
        if refit_hyperparams and not hasattr(pipe, "predict"):
            # if pipe was replaced by best_estimator_, ensure it's fitted
            pipe.fit(Xtr, ytr)

        yhat = pipe.predict(Xte)
        s = _scores(yte, yhat); s["variant"] = name
        results.append(s)
        variants.append((name, pipe))

        # KernelExplainer for SVR/MLP; for Ridge we’ll use LinearExplainer
        model_cls = pipe.named_steps["m"].__class__.__name__.lower() if isinstance(pipe, Pipeline) else pipe.__class__.__name__.lower()
        print(f"\n[{model_kind} | {name}] TEST: { {k:round(s[k],4) for k in ['R2','RMSE','MAE']} }")

        try:
            if "ridge" in model_cls:
                # Ridge: work in pipeline space (scaler inside), plot vs RAW X
                # Build arrays pipeline expects
                scaler = pipe.named_steps.get("scaler", None)
                if scaler is not None:
                    Xtr_s = scaler.transform(Xtr)   # NO re-fit here
                    Xte_s = scaler.transform(Xte)
                else:
                    Xtr_s, Xte_s = Xtr, Xte

                explainer = shap.LinearExplainer(pipe.named_steps["m"], Xtr_s, feature_names=xcols)
                sv = explainer.shap_values(Xte_s)
                shap.summary_plot(sv, Xte, feature_names=xcols, show=False)
                plt.title(f"{model_kind} | {name} — SHAP summary (LinearExplainer)"); plt.tight_layout(); plt.show()
                ex = _to_explanation(sv, Xte, xcols)
            else:
                # SVR / MLP → KernelExplainer
                f = _predict_fn(pipes[name])
                bg = Xtr if len(Xtr) <= 200 else Xtr[np.random.choice(len(Xtr), 200, replace=False)]
                explainer = shap.KernelExplainer(f, bg)
                sv = explainer.shap_values(Xte, nsamples=nsamples_kernel)
                shap.summary_plot(sv, Xte, feature_names=xcols, show=False)
                plt.title(f"{model_kind} | {name} — SHAP summary (KernelExplainer)"); plt.tight_layout(); plt.show()
                ex = _to_explanation(sv, Xte, xcols)

            _clustered_bar(ex, linkage, title=f"{model_kind} | {name} — Clustered SHAP bar", cutoff=1.0)

        except Exception as e:
            print(f"SHAP failed for {model_kind} | {name}: {e}")
    
    # Metrics table
    df_res = pd.DataFrame(results).sort_values("R2", ascending=False)
    print("\n=== Scaling ablation leaderboard (TEST) ===")
    display(df_res[["variant","R2","RMSE","MAE"]])

