"""Factory for creating unbalanced learners with the selected models and techniques.

For leveraging GPU acceleration, this module uses cuML estimators when available. Using GPU requires that cuML is installed and that the `use_gpu` flag is set to True

It implements the `ClassifierFactory` protocol.
"""

from typing import cast

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from experiments.core.modeling.classifiers import (
    HAS_CUML,
    Classifier,
    MetaCostClassifier,
    ModelType,
    Technique,
)
from experiments.core.modeling.classifiers import RobustSVC as SVC
from experiments.core.modeling.classifiers import RobustXGBClassifier as XGBClassifier

if HAS_CUML:
    from cuml.ensemble import RandomForestClassifier as CuRF

    from experiments.core.modeling.classifiers import CuSVC


class UnbalancedLearnerFactory:
    """Factory for creating estimators that handle unbalanced datasets."""

    def __init__(self, use_gpu: bool = False, sampler_k_neighbors: int = 3) -> None:
        self._use_gpu = use_gpu
        self._sampler_k_neighbors = sampler_k_neighbors

        if self._use_gpu and not HAS_CUML:
            raise ImportError("cuML is not available, cannot use GPU-based estimators.")

    def _get_model_instance(
        self,
        model_type: ModelType,
        random_state: int,
        use_gpu: bool,
        n_jobs: int = 1,
    ) -> BaseEstimator:
        """Internal helper to dispatch to CPU or GPU classes."""
        use_gpu = self._use_gpu if use_gpu is None else use_gpu

        if model_type == ModelType.SVM:
            if use_gpu and HAS_CUML:
                return CuSVC(random_state=random_state, probability=True)
            return SVC(random_state=random_state)

        elif model_type == ModelType.RANDOM_FOREST:
            if use_gpu and HAS_CUML:
                return CuRF(random_state=random_state)
            return RandomForestClassifier(random_state=random_state, n_jobs=n_jobs)

        elif model_type == ModelType.XGBOOST:
            return XGBClassifier(
                random_state=random_state,
                device="cuda" if use_gpu else "cpu",
                eval_metric="logloss",
                n_jobs=n_jobs,
                objective="binary:logistic",
                tree_method="hist",
            )

        elif model_type == ModelType.MLP:
            return MLPClassifier(random_state=random_state)

        raise ValueError(f"Unknown model type: {model_type}")

    def create_model(
        self,
        model_type: ModelType,
        technique: Technique,
        seed: int,
        use_gpu: bool | None = None,
        n_jobs: int = 1,
    ) -> Classifier:
        """Create a pipeline for the given model type and technique.

        Args:
            model_type: Type of model to create.
            technique: Technique for handling class imbalance.
            seed: Random seed for reproducibility.
            use_gpu: Whether to use GPU for training. Defaults to `None`, which means using the factory's setting.
            n_jobs: Number of parallel jobs for training. Defaults to `1`.

        Returns:
            The configured pipeline.
        """
        use_gpu = self._use_gpu if use_gpu is None else use_gpu

        steps: list[tuple[str, BaseEstimator]] = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]

        # Add sampling technique if applicable
        if technique == Technique.RANDOM_UNDER_SAMPLING:
            steps.append(("sampler", RandomUnderSampler(random_state=seed)))
        elif technique == Technique.SMOTE:
            steps.append(("sampler", self._create_smote_sampler(seed)))
        elif technique == Technique.SMOTE_TOMEK:
            steps.append(
                (
                    "sampler",
                    SMOTETomek(
                        random_state=seed, smote=self._create_smote_sampler(seed), n_jobs=n_jobs
                    ),
                )
            )
        # Add classifier
        clf = self._get_model_instance(model_type, seed, use_gpu=use_gpu, n_jobs=n_jobs)

        # Wrap with MetaCost if applicable
        if technique == Technique.META_COST:
            clf = MetaCostClassifier(base_estimator=clf, random_state=seed, n_jobs=n_jobs)

        steps.append(("clf", clf))
        return cast(Classifier, ImbPipeline(steps))

    def _create_smote_sampler(self, seed: int) -> SMOTE:
        """Helper to create a SMOTE sampler with appropriate parameters."""
        return SMOTE(random_state=seed, k_neighbors=self._sampler_k_neighbors)
