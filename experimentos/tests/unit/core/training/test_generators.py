"""Tests for experiments.core.training.generators module."""

from experiments.core.data import Dataset
from experiments.core.modeling.types import ModelType, Technique
from experiments.core.training.generators import (
    ExperimentTaskGenerator,
    TaskGeneratorConfig,
)
from experiments.core.training.protocols import ExperimentTask


class DescribeTaskGeneratorConfig:
    """Tests for TaskGeneratorConfig dataclass."""

    def it_has_default_num_seeds(self) -> None:
        """Verify default num_seeds is 30."""
        config = TaskGeneratorConfig()

        assert config.num_seeds == 30

    def it_has_empty_excluded_models_by_default(self) -> None:
        """Verify excluded_models defaults to empty set."""
        config = TaskGeneratorConfig()

        assert config.excluded_models == set()

    def it_accepts_custom_values(self) -> None:
        """Verify custom values are stored."""
        config = TaskGeneratorConfig(
            num_seeds=10,
            excluded_models={ModelType.SVM, ModelType.MLP},
        )

        assert config.num_seeds == 10
        assert config.excluded_models == {ModelType.SVM, ModelType.MLP}


class DescribeExperimentTaskGenerator:
    """Tests for ExperimentTaskGenerator class."""

    def it_initializes_with_config(self) -> None:
        """Verify generator stores config."""
        config = TaskGeneratorConfig(num_seeds=5)
        generator = ExperimentTaskGenerator(config)

        assert generator._config == config


class DescribeExperimentTaskGeneratorGenerate:
    """Tests for ExperimentTaskGenerator.generate() method."""

    def it_generates_tasks_for_all_combinations(self) -> None:
        """Verify tasks are generated for all valid combinations."""
        config = TaskGeneratorConfig(num_seeds=2)
        generator = ExperimentTaskGenerator(config)

        tasks = generator.generate([Dataset.TAIWAN_CREDIT])

        assert len(tasks) > 0
        # Should have tasks for each seed, model, and technique combination
        assert all(isinstance(t, ExperimentTask) for t in tasks)

    def it_respects_num_seeds(self) -> None:
        """Verify correct number of seeds are generated."""
        config = TaskGeneratorConfig(num_seeds=3)
        generator = ExperimentTaskGenerator(config)

        tasks = generator.generate([Dataset.TAIWAN_CREDIT])

        # Count unique seeds
        seeds = {t.seed for t in tasks}
        assert seeds == {0, 1, 2}

    def it_excludes_models_from_config(self) -> None:
        """Verify models from config exclusion are excluded."""
        config = TaskGeneratorConfig(
            num_seeds=1,
            excluded_models={ModelType.SVM, ModelType.MLP},
        )
        generator = ExperimentTaskGenerator(config)

        tasks = generator.generate([Dataset.TAIWAN_CREDIT])

        model_types = {t.model_type for t in tasks}
        assert ModelType.SVM not in model_types
        assert ModelType.MLP not in model_types

    def it_excludes_models_from_runtime_param(self) -> None:
        """Verify models from runtime parameter are excluded."""
        config = TaskGeneratorConfig(num_seeds=1)
        generator = ExperimentTaskGenerator(config)

        tasks = generator.generate(
            [Dataset.TAIWAN_CREDIT],
            excluded_models={ModelType.RANDOM_FOREST},
        )

        model_types = {t.model_type for t in tasks}
        assert ModelType.RANDOM_FOREST not in model_types

    def it_combines_config_and_runtime_exclusions(self) -> None:
        """Verify config and runtime exclusions are combined."""
        config = TaskGeneratorConfig(
            num_seeds=1,
            excluded_models={ModelType.SVM},
        )
        generator = ExperimentTaskGenerator(config)

        tasks = generator.generate(
            [Dataset.TAIWAN_CREDIT],
            excluded_models={ModelType.MLP},
        )

        model_types = {t.model_type for t in tasks}
        assert ModelType.SVM not in model_types
        assert ModelType.MLP not in model_types

    def it_returns_empty_list_when_all_models_excluded(self) -> None:
        """Verify empty list when all models are excluded."""
        all_models = set(ModelType)
        config = TaskGeneratorConfig(num_seeds=1, excluded_models=all_models)
        generator = ExperimentTaskGenerator(config)

        tasks = generator.generate([Dataset.TAIWAN_CREDIT])

        assert tasks == []

    def it_generates_tasks_for_multiple_datasets(self) -> None:
        """Verify tasks are generated for all datasets."""
        config = TaskGeneratorConfig(num_seeds=1)
        generator = ExperimentTaskGenerator(config)

        tasks = generator.generate([Dataset.TAIWAN_CREDIT, Dataset.LENDING_CLUB])

        datasets = {t.dataset for t in tasks}
        assert Dataset.TAIWAN_CREDIT in datasets
        assert Dataset.LENDING_CLUB in datasets

    def it_excludes_invalid_model_technique_combinations(self) -> None:
        """Verify CS_SVM is only used with SVM model."""
        config = TaskGeneratorConfig(num_seeds=1)
        generator = ExperimentTaskGenerator(config)

        tasks = generator.generate([Dataset.TAIWAN_CREDIT])

        # CS_SVM should only appear with SVM model
        cs_svm_tasks = [t for t in tasks if t.technique == Technique.CS_SVM]
        for task in cs_svm_tasks:
            assert task.model_type == ModelType.SVM

    def it_includes_all_valid_techniques(self) -> None:
        """Verify all valid techniques are included."""
        config = TaskGeneratorConfig(num_seeds=1)
        generator = ExperimentTaskGenerator(config)

        tasks = generator.generate([Dataset.TAIWAN_CREDIT])

        techniques = {t.technique for t in tasks}
        # All non-CS_SVM techniques should be present for all models
        assert Technique.BASELINE in techniques
        assert Technique.SMOTE in techniques


class DescribeExperimentTaskGeneratorGenerateSingle:
    """Tests for ExperimentTaskGenerator.generate_single() method."""

    def it_generates_single_task(self) -> None:
        """Verify single task is generated correctly."""
        config = TaskGeneratorConfig()
        generator = ExperimentTaskGenerator(config)

        task = generator.generate_single(
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.BASELINE,
            seed=42,
        )

        assert task is not None
        assert task.dataset == Dataset.TAIWAN_CREDIT
        assert task.model_type == ModelType.RANDOM_FOREST
        assert task.technique == Technique.BASELINE
        assert task.seed == 42

    def it_returns_none_for_invalid_combination(self) -> None:
        """Verify None is returned for invalid model/technique combination."""
        config = TaskGeneratorConfig()
        generator = ExperimentTaskGenerator(config)

        # CS_SVM is only valid for SVM
        task = generator.generate_single(
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.RANDOM_FOREST,
            technique=Technique.CS_SVM,
            seed=42,
        )

        assert task is None

    def it_allows_cs_svm_with_svm_model(self) -> None:
        """Verify CS_SVM is allowed with SVM model."""
        config = TaskGeneratorConfig()
        generator = ExperimentTaskGenerator(config)

        task = generator.generate_single(
            dataset=Dataset.TAIWAN_CREDIT,
            model_type=ModelType.SVM,
            technique=Technique.CS_SVM,
            seed=42,
        )

        assert task is not None
        assert task.technique == Technique.CS_SVM
        assert task.model_type == ModelType.SVM


class DescribeExperimentTaskGeneratorIsValidCombination:
    """Tests for ExperimentTaskGenerator._is_valid_combination() method."""

    def it_returns_true_for_baseline_with_any_model(self) -> None:
        """Verify BASELINE is valid with any model."""
        config = TaskGeneratorConfig()
        generator = ExperimentTaskGenerator(config)

        for model_type in ModelType:
            assert generator._is_valid_combination(model_type, Technique.BASELINE)

    def it_returns_true_for_smote_with_any_model(self) -> None:
        """Verify SMOTE is valid with any model."""
        config = TaskGeneratorConfig()
        generator = ExperimentTaskGenerator(config)

        for model_type in ModelType:
            assert generator._is_valid_combination(model_type, Technique.SMOTE)

    def it_returns_false_for_cs_svm_with_non_svm_model(self) -> None:
        """Verify CS_SVM is invalid with non-SVM models."""
        config = TaskGeneratorConfig()
        generator = ExperimentTaskGenerator(config)

        non_svm_models = [m for m in ModelType if m != ModelType.SVM]
        for model_type in non_svm_models:
            assert not generator._is_valid_combination(model_type, Technique.CS_SVM)

    def it_returns_true_for_cs_svm_with_svm_model(self) -> None:
        """Verify CS_SVM is valid with SVM model."""
        config = TaskGeneratorConfig()
        generator = ExperimentTaskGenerator(config)

        assert generator._is_valid_combination(ModelType.SVM, Technique.CS_SVM)
