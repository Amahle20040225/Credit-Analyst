# model_engine.py
"""
Robust model engine with a pure-Python fallback so the app can run even when
`scikit-learn` is not available or fails to build on the target machine.
"""

from typing import Tuple
import numpy as np

HAS_SKLEARN = True
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, roc_curve
except Exception:
    HAS_SKLEARN = False


if not HAS_SKLEARN:
    # Minimal StandardScaler fallback
    class StandardScaler:
        def __init__(self):
            self.mean_ = None
            self.scale_ = None

        def fit(self, X: np.ndarray):
            self.mean_ = np.mean(X, axis=0)
            self.scale_ = np.std(X, axis=0)
            self.scale_[self.scale_ == 0] = 1.0
            return self

        def transform(self, X: np.ndarray) -> np.ndarray:
            return (X - self.mean_) / self.scale_

        def fit_transform(self, X: np.ndarray) -> np.ndarray:
            return self.fit(X).transform(X)

    # Simple logistic regression using gradient descent
    class LogisticRegression:
        def __init__(self, max_iter=1000, random_state=42, C=1.0):
            self.max_iter = max_iter
            self.random_state = random_state
            self.C = C
            self.coef_ = None
            self.intercept_ = 0.0

        def _sigmoid(self, z: np.ndarray) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

        def fit(self, X: np.ndarray, y: np.ndarray):
            rng = np.random.RandomState(self.random_state)
            n_samples, n_features = X.shape
            X_b = np.hstack([np.ones((n_samples, 1)), X])
            w = np.zeros(n_features + 1)

            lr = 0.1
            reg = 1.0 / max(self.C, 1e-12)

            for _ in range(self.max_iter):
                preds = self._sigmoid(X_b.dot(w))
                error = preds - y
                grad = (X_b.T.dot(error)) / n_samples
                # L2 regularization (skip bias term)
                grad[1:] += reg * w[1:]
                w -= lr * grad

            self.intercept_ = w[0]
            self.coef_ = w[1:][np.newaxis, :]
            return self

        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            if X.ndim == 1:
                X = X.reshape(1, -1)
            z = X.dot(self.coef_.T).ravel() + self.intercept_
            probs_pos = self._sigmoid(z)
            probs_neg = 1.0 - probs_pos
            return np.vstack([probs_neg, probs_pos]).T

    def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
        # Fast implementation using the Mann-Whitney U statistic
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        pos = y_score[y_true == 1]
        neg = y_score[y_true == 0]
        if pos.size == 0 or neg.size == 0:
            return 0.5
        # compute ranks
        combined = np.concatenate([pos, neg])
        ranks = np.argsort(np.argsort(combined)) + 1
        # sum of ranks for positives
        rank_pos = np.sum(ranks[: pos.size])
        u = rank_pos - (pos.size * (pos.size + 1)) / 2.0
        auc = u / (pos.size * neg.size)
        return float(auc)

    def roc_curve(y_true: np.ndarray, y_score: np.ndarray):
        # Simple ROC points computation
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        desc_score_indices = np.argsort(-y_score)
        y_true = y_true[desc_score_indices]
        distinct_value_indices = np.where(np.diff(y_score[desc_score_indices]))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
        tps = np.cumsum(y_true == 1)[threshold_idxs]
        fps = 1 + threshold_idxs - tps
        tps = tps.astype(float)
        fps = fps.astype(float)
        fn = np.sum(y_true == 1) - tps
        tn = np.sum(y_true == 0) - fps
        tpr = tps / (tps + fn)
        fpr = fps / (fps + tn)
        thresholds = y_score[desc_score_indices][threshold_idxs]
        return fpr, tpr, thresholds


class CreditRiskModelEngine:

    def __init__(self, target_features: list):

        self.features = target_features

        self.scaler = StandardScaler()

        # logistic model (sklearn or fallback)
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )

        self.test_auc_score = 0.0

        self.fpr = None
        self.tpr = None

    def execute_training_pipeline(self, clean_df) -> Tuple:

        # -----------------------------------------
        # TRAIN / TEST SPLIT
        # -----------------------------------------

        train_matrix = (
            clean_df[clean_df['set'] == 'train']
        )

        test_matrix = (
            clean_df[clean_df['set'] == 'test']
        )

        # -----------------------------------------
        # FEATURE ARRAYS
        # -----------------------------------------

        X_train = train_matrix[self.features]
        y_train = train_matrix['default_flag']

        X_test = test_matrix[self.features]
        y_test = test_matrix['default_flag']

        # Convert to numpy for fallback implementations
        X_train_np = X_train.values if hasattr(X_train, 'values') else np.asarray(X_train)
        X_test_np = X_test.values if hasattr(X_test, 'values') else np.asarray(X_test)
        y_train_np = y_train.values if hasattr(y_train, 'values') else np.asarray(y_train)
        y_test_np = y_test.values if hasattr(y_test, 'values') else np.asarray(y_test)

        # -----------------------------------------
        # SCALE FEATURES
        # -----------------------------------------

        X_train_scaled = self.scaler.fit_transform(X_train_np)
        X_test_scaled = self.scaler.transform(X_test_np)

        # -----------------------------------------
        # MODEL TRAINING
        # -----------------------------------------

        self.model.fit(X_train_scaled, y_train_np)

        # -----------------------------------------
        # PREDICTIONS
        # -----------------------------------------

        predicted_probabilities = self.model.predict_proba(X_test_scaled)[:, 1]

        # -----------------------------------------
        # AUC SCORE
        # -----------------------------------------

        self.test_auc_score = float(
            roc_auc_score(
                y_test_np,
                predicted_probabilities
            )
        )

        # -----------------------------------------
        # ROC CURVE
        # -----------------------------------------

        self.fpr, self.tpr, _ = roc_curve(
            y_test_np,
            predicted_probabilities
        )

        return (
            X_test,
            y_test,
            predicted_probabilities
        )