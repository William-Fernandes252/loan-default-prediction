# Plan: Refactor Context Class with Dependency Injection

The goal is to decompose the monolithic `Context` class into focused, single-responsibility services and wire them through a proper DI container, while fixing open-closed principle violations in the pipeline factory setup.

## Steps

1. **Implement `PathManager`** — Extract all 8 path-related methods from `Context` into this service, implementing existing protocols like `CheckpointPathProvider`, `ConsolidatedResultsPathProvider`, and `ResultsPathProvider`.

2. **Implement `ResourceCalculator`** — Extract `compute_safe_jobs` logic from `Context`, encapsulating RAM-based job calculation with configurable `safety_factor`.

3. **Remove the `_configure_environment()`** — The environment variables are now expected to be set externally (e.g., via `.env` or system configuration).

4. **Create `ModelVersioningServiceFactory`** — Extract `get_model_versioning_service()` into a factory class that creates `ModelVersioningServiceImpl` instances, replacing direct Context-based instantiation.

5. **Complete the DI `Container`** — Wire all new services as `Singleton` or `Factory` providers, expanding `wiring_config` to include all CLI modules (`train`, `data`, `features`, `analysis`, `predict`).

6. **Refactor CLI commands to use `@inject`** — Replace manual `Context()` instantiation in `train.py`, `analysis.py`, `data.py`, `features.py`, and `predict.py` with dependency-injected services.

## Further Considerations

1. **Dataset transformer registry in `ExperimentDataManager`** — Refactor the hard-coded `_transformer_registry` dict by overriding the container `init_resources` method?

2. **Deprecation strategy for `Context`** — Try to remove it entirely. Some methods may need to be temporarily duplicated in new services during the transition. But the plan is to fully phase it out. It is passed to many places currently, so careful refactoring is needed.

3. **Configuration loading** — Refactor `AppConfig` into a `ExperimentsSettings`, using Pydantic Settings to be loaded via the container's `config` provider (e.g., from YAML/env).