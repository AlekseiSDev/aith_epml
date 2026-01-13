"""Luigi task for running multiple trainings in parallel."""

from __future__ import annotations

import luigi

from wine_quality_epml.pipeline.test_model_task import TestModelTask


class MultiTrainTask(luigi.WrapperTask):
    """Wrapper task to run multiple model trainings in parallel."""

    params_list = luigi.ListParameter(
        default=[
            "configs/parallel_rf.yaml",
            "configs/parallel_ridge.yaml",
            "configs/parallel_gbr.yaml",
        ]
    )

    def requires(self) -> list[TestModelTask]:
        """Depends on multiple TestModelTask instances."""
        return [TestModelTask(params_path=p) for p in self.params_list]
