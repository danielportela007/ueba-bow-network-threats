"""
Modelado: clasificacion supervisada.

Implementa:
- Division train/test a nivel de usuario (sin filtracion).
- Tres representaciones: BoW cruda, BoW + TF-IDF, features originales (baseline).
- Manejo de desbalance: downsampling (como en el paper RBD24) y class weights.
- Clasificadores: XGBoost, Random Forest, Gradient Boosting, Logistic Regression, SVM.
- Validacion cruzada estratificada por grupo (usuario).
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, confusion_matrix, average_precision_score,
)
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
from . import config


def user_level_split(X, y, user_ids, test_size=None, random_state=None):
    """Division train/test garantizando separacion a nivel usuario."""
    ts = test_size or config.TEST_SIZE
    rs = random_state or config.RANDOM_STATE

    splitter = GroupShuffleSplit(n_splits=1, test_size=ts, random_state=rs)
    train_idx, test_idx = next(splitter.split(X, y, groups=user_ids))

    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    u_tr, u_te = user_ids[train_idx], user_ids[test_idx]

    print(f"\n--- Division train/test (nivel usuario) ---")
    print(f"  Train: {len(train_idx):,} muestras, {len(set(u_tr))} usuarios "
          f"({(y_tr==1).sum():,} positivos)")
    print(f"  Test:  {len(test_idx):,} muestras, {len(set(u_te))} usuarios "
          f"({(y_te==1).sum():,} positivos)")

    assert not (set(u_tr) & set(u_te)), "Filtracion de usuarios detectada"
    print("  Sin filtracion de usuarios entre train/test")

    return X_tr, X_te, y_tr, y_te, u_tr, u_te


def downsample_balance(X, y, ratio=1.0, random_state=None):
    """Submuestreo de la clase mayoritaria (como en el paper RBD24).

    Parameters
    ----------
    ratio : float
        Proporcion deseada neg/pos. 1.0 = balanceado. 2.0 = 2:1.
    """
    rs = random_state or config.RANDOM_STATE
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    target_neg = int(n_pos * ratio)
    target_neg = min(target_neg, n_neg)

    sampler = RandomUnderSampler(
        sampling_strategy={0: target_neg, 1: n_pos},
        random_state=rs,
    )
    X_bal, y_bal = sampler.fit_resample(X, y)
    print(f"  Downsampling: {n_neg:,}neg/{n_pos:,}pos -> "
          f"{(y_bal==0).sum():,}neg/{(y_bal==1).sum():,}pos")
    return X_bal, y_bal


def get_classifiers(random_state=None, scale_pos_weight=1.0):
    """Diccionario de clasificadores supervisados a evaluar."""
    rs = random_state or config.RANDOM_STATE
    return {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.01,
            min_child_weight=5, reg_alpha=0.1, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", random_state=rs, n_jobs=-1,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=7, min_samples_split=10,
            min_samples_leaf=5, class_weight="balanced",
            random_state=rs, n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.01,
            min_samples_split=10, min_samples_leaf=5,
            subsample=0.8, random_state=rs,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", C=1.0,
            random_state=rs, solver="saga", n_jobs=-1,
        ),
        "SVM (RBF)": SVC(
            kernel="rbf", class_weight="balanced", probability=True,
            random_state=rs, gamma="scale", C=10.0,
        ),
    }


def compute_metrics(y_true, y_pred, y_proba):
    """Calcula el conjunto completo de metricas de evaluacion."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "F1": f1_score(y_true, y_pred, pos_label=1),
        "F1_macro": f1_score(y_true, y_pred, average="macro"),
        "Precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall (TPR)": recall_score(y_true, y_pred, pos_label=1),
        "Accuracy": accuracy_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
        "PR-AUC": average_precision_score(y_true, y_proba),
        "TPR": tp / max(tp + fn, 1),
        "FPR": fp / max(fp + tn, 1),
        "Confusion Matrix": cm,
    }


def train_evaluate_representation(X_train, X_test, y_train, y_test, rep_name,
                                  use_scaling=False, downsample_ratio=1.0):
    """Entrena y evalua todos los clasificadores sobre una representacion.

    Siguiendo la metodologia del paper RBD24:
    - Downsampling del training set para balancear.
    - Evaluacion en test set balanceado (downsampled) para metricas comparables
      con el paper (F1, Accuracy, TPR, FPR).
    - Evaluacion adicional en test completo para ROC-AUC y PR-AUC (metricas
      que no dependen del umbral de decision y son validas en datos desbalanceados).

    Parameters
    ----------
    rep_name : str
        Nombre de la representacion (para reportes).
    use_scaling : bool
        Si True, aplica StandardScaler (necesario para SVM/LR con features raw).
    downsample_ratio : float
        Ratio neg/pos despues del downsampling.
    """
    print(f"\n{'='*60}")
    print(f"EVALUACION: {rep_name}")
    print(f"{'='*60}")

    # Downsampling del training set (como el paper RBD24)
    X_bal, y_bal = downsample_balance(X_train, y_train, ratio=downsample_ratio)

    # Tambien balancear el test set (metodologia RBD24)
    n_pos_test = int((y_test == 1).sum())
    if n_pos_test > 0:
        test_sampler = RandomUnderSampler(
            sampling_strategy={0: n_pos_test, 1: n_pos_test},
            random_state=config.RANDOM_STATE,
        )
        X_test_bal, y_test_bal = test_sampler.fit_resample(X_test, y_test)
        print(f"  Test balanceado: {(y_test_bal==0).sum()}neg/{(y_test_bal==1).sum()}pos")
    else:
        X_test_bal, y_test_bal = X_test, y_test

    if use_scaling:
        scaler = StandardScaler()
        X_bal_s = scaler.fit_transform(X_bal)
        X_test_s = scaler.transform(X_test)
        X_test_bal_s = scaler.transform(X_test_bal)
    else:
        scaler = None
        X_bal_s = X_bal
        X_test_s = X_test
        X_test_bal_s = X_test_bal

    spw = (y_bal == 0).sum() / max((y_bal == 1).sum(), 1)
    classifiers = get_classifiers(scale_pos_weight=spw)

    results = {}
    predictions = {}
    best_f1 = -1
    best_info = None

    for name, clf in classifiers.items():
        print(f"  {name}...", end=" ", flush=True)

        # SVM y LR necesitan scaling
        if name in ["SVM (RBF)", "Logistic Regression"]:
            if scaler is None:
                sc = StandardScaler()
                X_fit = sc.fit_transform(X_bal)
                X_eval_bal = sc.transform(X_test_bal)
                X_eval_full = sc.transform(X_test)
            else:
                X_fit = X_bal_s
                X_eval_bal = X_test_bal_s
                X_eval_full = X_test_s
        else:
            X_fit = X_bal
            X_eval_bal = X_test_bal
            X_eval_full = X_test

        clf.fit(X_fit, y_bal)

        # Metricas en test BALANCEADO (comparable con paper RBD24)
        y_pred_bal = clf.predict(X_eval_bal)
        y_proba_bal = clf.predict_proba(X_eval_bal)[:, 1]
        m = compute_metrics(y_test_bal, y_pred_bal, y_proba_bal)

        # ROC-AUC y PR-AUC en test COMPLETO (mas informativo)
        y_proba_full = clf.predict_proba(X_eval_full)[:, 1]
        m["ROC-AUC (full)"] = roc_auc_score(y_test, y_proba_full)
        m["PR-AUC (full)"] = average_precision_score(y_test, y_proba_full)

        results[name] = m
        predictions[name] = {
            "y_pred": y_pred_bal,
            "y_proba": y_proba_bal,
            "y_proba_full": y_proba_full,
        }

        print(f"F1={m['F1']:.4f}  Prec={m['Precision']:.4f}  "
              f"Rec={m['Recall (TPR)']:.4f}  "
              f"ROC-AUC={m['ROC-AUC']:.4f}  "
              f"ROC-AUC(full)={m['ROC-AUC (full)']:.4f}")

        if m["F1"] > best_f1:
            best_f1 = m["F1"]
            best_info = (name, clf)

    print(f"\n  >> Mejor: {best_info[0]} (F1={best_f1:.4f})")
    return results, best_info, predictions



def cross_validate_best(X, y, user_ids, best_name, n_splits=5):
    """Validacion cruzada estratificada por grupo del mejor modelo.

    Usa StratifiedGroupKFold para garantizar que:
    - Usuarios no se repiten entre folds
    - Proporcion de clases se mantiene
    """
    print(f"\n--- Validacion cruzada ({n_splits}-fold) para {best_name} ---")

    # Crear labels a nivel usuario para estratificacion
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                random_state=config.RANDOM_STATE)

    fold_metrics = []
    for fold, (tr_idx, te_idx) in enumerate(sgkf.split(X, y, groups=user_ids)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # Skip fold si no hay positivos en test
        if (y_te == 1).sum() == 0 or (y_tr == 1).sum() == 0:
            continue

        # Downsampling del training
        n_pos = int((y_tr == 1).sum())
        n_neg = int((y_tr == 0).sum())
        target_neg = min(n_neg, n_pos)
        sampler = RandomUnderSampler(
            sampling_strategy={0: target_neg, 1: n_pos},
            random_state=config.RANDOM_STATE,
        )
        X_bal, y_bal = sampler.fit_resample(X_tr, y_tr)

        # Downsampling del test (metodologia RBD24)
        n_pos_te = int((y_te == 1).sum())
        n_neg_te = int((y_te == 0).sum())
        target_neg_te = min(n_neg_te, n_pos_te)
        sampler_te = RandomUnderSampler(
            sampling_strategy={0: target_neg_te, 1: n_pos_te},
            random_state=config.RANDOM_STATE,
        )
        X_te_bal, y_te_bal = sampler_te.fit_resample(X_te, y_te)

        spw = (y_bal == 0).sum() / max((y_bal == 1).sum(), 1)

        if best_name == "XGBoost":
            clf = xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.01,
                min_child_weight=5, reg_alpha=0.1, subsample=0.8,
                colsample_bytree=0.8, scale_pos_weight=spw,
                eval_metric="logloss", random_state=config.RANDOM_STATE, n_jobs=-1,
            )
        elif best_name == "Random Forest":
            clf = RandomForestClassifier(
                n_estimators=300, max_depth=7, min_samples_split=10,
                min_samples_leaf=5, class_weight="balanced",
                random_state=config.RANDOM_STATE, n_jobs=-1,
            )
        elif best_name == "Gradient Boosting":
            clf = GradientBoostingClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.01,
                min_samples_split=10, min_samples_leaf=5, subsample=0.8,
                random_state=config.RANDOM_STATE,
            )
        else:
            clf = xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.01,
                min_child_weight=5, eval_metric="logloss",
                random_state=config.RANDOM_STATE, n_jobs=-1,
            )

        clf.fit(X_bal, y_bal)
        y_pred = clf.predict(X_te_bal)
        y_proba = clf.predict_proba(X_te_bal)[:, 1]

        m = compute_metrics(y_te_bal, y_pred, y_proba)
        fold_metrics.append(m)
        print(f"  Fold {fold+1}: F1={m['F1']:.4f}  ROC-AUC={m['ROC-AUC']:.4f}  "
              f"TPR={m['TPR']:.4f}  FPR={m['FPR']:.4f}")

    if fold_metrics:
        print(f"\n  Promedio CV:")
        for key in ["F1", "F1_macro", "Precision", "Recall (TPR)",
                     "ROC-AUC", "PR-AUC", "TPR", "FPR"]:
            vals = [m[key] for m in fold_metrics]
            print(f"    {key}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    return fold_metrics
