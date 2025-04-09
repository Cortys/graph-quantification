from datetime import datetime
import os
from typing import Any, Callable, Dict, List, Optional, Union

from filelock import SoftFileLock
from matplotlib.pylab import f
from sacred import Experiment
from tinydb import Query, TinyDB
from tinydb_serialization import SerializationMiddleware, Serializer

from gq.utils.config import (
    DataConfiguration,
    ModelConfiguration,
    RunConfiguration,
    TrainingConfiguration,
)


class ModelExistsError(Exception):
    """exception raised when existing model found with the same configuration at a certain path"""

    def __init__(self, message="Model already exists!"):
        self.message = message
        super().__init__(self.message)


class ModelNotFoundError(Exception):
    """exception raised when model with desired configuration not found at a certain path"""

    def __init__(self, message="No matching model found!"):
        self.message = message
        super().__init__(self.message)


class DateTimeSerializer(Serializer):
    """helper-class serializing date and time"""

    OBJ_CLASS = datetime  # The class this serializer handles

    def encode(self, obj):
        return obj.strftime("%Y-%m-%dT%H:%M:%S")

    def decode(self, s):
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")


class Storage:
    """helper-class used for loading and storing models using a small JSON-based TinyDB"""

    def __init__(
        self,
        base_directory: str,
        experiment_name: str = "",
        eval_experiment_name: str = "",
        experiment: Optional[Experiment] = None,
        lock_timeout: int = 10,
        allow_override: bool = False,
    ):
        cache_dir = os.path.join(base_directory, eval_experiment_name)
        os.makedirs(cache_dir, exist_ok=True)

        self.experiment_name = experiment_name
        self.eval_experiment_name = eval_experiment_name
        self.experiment = experiment
        self.cache_dir = cache_dir
        self.lock_timeout = lock_timeout
        self.allow_override = allow_override

        self.dbs: Dict[str, TinyDB] = {}

    @staticmethod
    def locked_call(
        callable_: Callable[[], Any], lock_file: str, lock_timeout: int
    ) -> Any:
        lock = SoftFileLock(lock_file, timeout=lock_timeout)
        with lock.acquire(timeout=lock_timeout):
            return callable_()

    def _get_index_path(self, table: str) -> str:
        return os.path.join(self.cache_dir, f"{table}.json")

    def _get_lock_path(self, table: str) -> str:
        return f"{self._get_index_path(table)}.lock"

    def _get_db(self, table: str) -> TinyDB:
        if table == "index":
            raise ValueError("The table must not be `index`!")
        serialization = SerializationMiddleware()
        serialization.register_serializer(DateTimeSerializer(), "DateTime")
        return TinyDB(self._get_index_path(table), storage=serialization)

    def _upsert_meta(
        self, table: str, params: Dict[str, Any], experiment_id: Optional[int] = None
    ) -> List[int]:
        meta = (
            {}
            if self.experiment is None or self.experiment.mainfile is None
            else {
                "commit": self.experiment.mainfile.commit,
                "is_dirty": self.experiment.mainfile.is_dirty,
                "filename": os.path.basename(self.experiment.mainfile.filename),
            }
        )
        data = {
            "params": params,
            "meta": meta,
            "time": datetime.utcnow(),
            "experiment_id": experiment_id,
        }

        table_db = self._get_db(table)
        doc_id = table_db.upsert(data, Query().params == params)
        return doc_id

    def _remove_meta(self, table: str, params: Dict[str, Any]) -> List[int]:
        return self._get_db(table).remove(Query().params == params)

    def _find_meta_by_exact_params(
        self, table: str, params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        return self._get_db(table).search(Query().params == params)  # type: ignore

    def _find_meta(
        self, table: str, match_condition: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        query = Query()
        composite_condition = None
        for key, value in match_condition.items():
            current_condition = query["params"][key] == value

            composite_condition = (
                current_condition
                if composite_condition is None
                else composite_condition & current_condition
            )

        if composite_condition is None:
            return self._get_db(table).all()  # type: ignore

        return self._get_db(table).search(composite_condition)  # type: ignore

    def build_model_dir_path(self, artifact_type: str, dir_id: Union[int, str]) -> str:
        path = os.path.join(self.cache_dir, artifact_type, str(dir_id))
        os.makedirs(path, exist_ok=True)

        return path

    def build_model_file_path(
        self, path: str, init_no: int = 1, round_no: int | None = None
    ) -> str:
        # final model path, file-name is simply based on init_no and round_no
        suffix = str(init_no)
        if round_no is not None:
            suffix += f"_{round_no}"

        model_file_name = f"model_{suffix}.pt"
        model_file_path = os.path.join(path, model_file_name)

        return model_file_path

    def build_results_file_path(
        self,
        path: str,
        init_no: int = 1,
        round_no: int | None = None,
        quantification_test_skew_strategy: str | None = None,
        quantification_test_skew_repeats: int | None = None,
        quantification_test_skew_tuplet_size: int | None = None,
        quantification_test_skew_depth_limit: int | None = None,
        quantification_test_skew_noise: float | None = None,
        quantification_test_tuplet_oversampling_factor: float | None = None,
        quantification_test_tuplet_size: int | float | None = None,
    ) -> str:
        # final results path, file-name is simply based on init_no and round_no
        if self.experiment_name != self.eval_experiment_name:
            prefix = f"_{self.experiment_name}"
        else:
            prefix = ""

        suffix = str(init_no)

        if quantification_test_tuplet_oversampling_factor is not None:
            suffix += f"_tupf{quantification_test_tuplet_oversampling_factor}"

        if quantification_test_tuplet_size is not None:
            suffix += f"_tups{quantification_test_tuplet_size}"

        if quantification_test_skew_repeats is not None:
            suffix += f"_skew_{quantification_test_skew_strategy}_{quantification_test_skew_repeats}"
            if quantification_test_skew_tuplet_size is not None:
                suffix += f"_{quantification_test_skew_tuplet_size}"
            if quantification_test_skew_strategy == "neighbor":
                if quantification_test_skew_depth_limit is not None:
                    suffix += f"_d{quantification_test_skew_depth_limit}"
                if (
                    quantification_test_skew_noise is not None
                    and quantification_test_skew_noise > 0
                ):
                    suffix += f"_n{quantification_test_skew_noise}"

        if round_no is not None:
            suffix += f"_{round_no}"

        results_file_name = f"results{prefix}_{suffix}.json"
        results_file_path = os.path.join(path, results_file_name)

        return results_file_path

    def build_al_subset_file_path(
        self, path: str, init_no: int = 1, round_no: int | None = None
    ) -> str:
        # final al subset path, file-name is simply based on init_no and round_no
        if self.experiment_name != self.eval_experiment_name:
            prefix = f"_{self.experiment_name}"
        else:
            prefix = ""

        suffix = str(init_no)
        if round_no is not None:
            suffix += f"_{round_no}"

        al_subset_file_name = f"al_subset{prefix}_{suffix}.json"
        al_subset_file_path = os.path.join(path, al_subset_file_name)

        return al_subset_file_path

    def create_model_file_path(
        self,
        artifact_type: str,
        params: Dict[str, Any],
        init_no: int = 1,
        round_no: int | None = None,
    ) -> str:
        documents = self.find_artifacts(artifact_type, params)

        # more than one matching document: search ambiguous, raise Error
        if len(documents) > 1:
            document_ids = [d["id"] for d in documents]
            raise RuntimeError(
                f"Found more than one matching entry (artficat_type={artifact_type}, params={params}): {document_ids}"
            )

        # exactly 1 document found, i.e. some models with same configuration
        # might exists: check if same init_no already exists, if so raise Error
        # else return proper model_file_path
        if len(documents) == 1:
            # check if init_no already exists
            path = self.build_model_dir_path(artifact_type, documents[0]["id"])
            model_file_path = self.build_model_file_path(
                path, init_no=init_no, round_no=round_no
            )

            if os.path.exists(model_file_path) and not self.allow_override:
                raise ModelExistsError(
                    f"model already exists (artifact_type={artifact_type}, \
                    params={params}, init_no={init_no}, round_no={round_no})"
                )

            return model_file_path

        # else: default case, no previous models with same config exist
        # return model_file_path
        ids = Storage.locked_call(
            lambda: self._upsert_meta(artifact_type, params),
            self._get_lock_path(artifact_type),
            self.lock_timeout,
        )

        if len(ids) != 1:
            raise RuntimeError(
                f"Something unexpected happened. The index contains duplicates \
                (artifact_type={artifact_type}, params={params})"
            )

        path = self.build_model_dir_path(artifact_type, ids[0])
        model_file_path = self.build_model_file_path(
            path, init_no=init_no, round_no=round_no
        )

        return model_file_path

    def create_results_file_path(
        self,
        artifact_type: str,
        params: Dict[str, Any],
        init_no: int = 1,
        round_no: int | None = None,
        quantification_test_skew_strategy: str | None = None,
        quantification_test_skew_repeats: int | None = None,
        quantification_test_skew_tuplet_size: int | None = None,
        quantification_test_skew_depth_limit: int | None = None,
        quantification_test_skew_noise: float | None = None,
        quantification_test_tuplet_oversampling_factor: float | None = None,
        quantification_test_tuplet_size: int | float | None = None,
    ) -> str:
        documents = self.find_artifacts(artifact_type, params)

        # more than one matching document: search ambiguous, raise Error
        if len(documents) > 1:
            document_ids = [d["id"] for d in documents]
            raise RuntimeError(
                f"Found more than one matching entry (artficat_type={artifact_type}, params={params}): {document_ids}"
            )

        # exactly 1 document found, i.e. some models with same configuration
        # might exists: check if same init_no already exists, if so raise Error
        # else return proper results_file_path
        if len(documents) == 1:
            # check if init_no already exists
            path = self.build_model_dir_path(artifact_type, documents[0]["id"])
            results_file_path = self.build_results_file_path(
                path,
                init_no=init_no,
                round_no=round_no,
                quantification_test_skew_strategy=quantification_test_skew_strategy,
                quantification_test_skew_repeats=quantification_test_skew_repeats,
                quantification_test_skew_tuplet_size=quantification_test_skew_tuplet_size,
                quantification_test_skew_depth_limit=quantification_test_skew_depth_limit,
                quantification_test_skew_noise=quantification_test_skew_noise,
                quantification_test_tuplet_oversampling_factor=quantification_test_tuplet_oversampling_factor,
                quantification_test_tuplet_size=quantification_test_tuplet_size,
            )

            return results_file_path

        # else: default case, no previous models with same config exist
        # return results_file_path
        ids = Storage.locked_call(
            lambda: self._upsert_meta(artifact_type, params),
            self._get_lock_path(artifact_type),
            self.lock_timeout,
        )

        if len(ids) != 1:
            raise RuntimeError(
                f"Something unexpected happened. The index contains duplicates \
                (artifact_type={artifact_type}, params={params})"
            )

        path = self.build_model_dir_path(artifact_type, ids[0])
        results_file_path = self.build_results_file_path(
            path,
            init_no=init_no,
            round_no=round_no,
            quantification_test_skew_strategy=quantification_test_skew_strategy,
            quantification_test_skew_repeats=quantification_test_skew_repeats,
            quantification_test_skew_tuplet_size=quantification_test_skew_tuplet_size,
            quantification_test_skew_depth_limit=quantification_test_skew_depth_limit,
            quantification_test_skew_noise=quantification_test_skew_noise,
            quantification_test_tuplet_oversampling_factor=quantification_test_tuplet_oversampling_factor,
            quantification_test_tuplet_size=quantification_test_tuplet_size,
        )

        return results_file_path

    def create_al_subset_file_path(
        self,
        artifact_type: str,
        params: Dict[str, Any],
        init_no: int = 1,
        round_no: int = 0,
    ) -> str:
        documents = self.find_artifacts(artifact_type, params)

        # more than one matching document: search ambiguous, raise Error
        if len(documents) > 1:
            document_ids = [d["id"] for d in documents]
            raise RuntimeError(
                f"Found more than one matching entry (artficat_type={artifact_type}, params={params}): {document_ids}"
            )

        # exactly 1 document found, i.e. some models with same configuration
        # might exists: check if same init_no already exists, if so raise Error
        # else return proper results_file_path
        if len(documents) == 1:
            # check if init_no already exists
            path = self.build_model_dir_path(artifact_type, documents[0]["id"])
            results_file_path = self.build_al_subset_file_path(
                path, init_no=init_no, round_no=round_no
            )

            return results_file_path

        # else: default case, no previous models with same config exist
        # return results_file_path
        ids = Storage.locked_call(
            lambda: self._upsert_meta(artifact_type, params),
            self._get_lock_path(artifact_type),
            self.lock_timeout,
        )

        if len(ids) != 1:
            raise RuntimeError(
                f"Something unexpected happened. The index contains duplicates \
                (artifact_type={artifact_type}, params={params})"
            )

        path = self.build_model_dir_path(artifact_type, ids[0])
        results_file_path = self.build_al_subset_file_path(
            path, init_no=init_no, round_no=round_no
        )

        return results_file_path

    def retrieve_model_dir_path(
        self, artifact_type: str, match_condition: Dict[str, Any]
    ) -> str:
        documents = self.find_artifacts(artifact_type, match_condition)

        try:
            document = documents[0]

        except IndexError:
            raise ModelNotFoundError(
                f"""No matching model found with (artficat_type={artifact_type},
                match_condition={match_condition}"""
            )

        return self.build_model_dir_path(artifact_type, document["id"])

    def retrieve_model_file_path(
        self,
        artifact_type: str,
        params: Dict[str, Any],
        init_no: int = 1,
        round_no: int | None = None,
    ) -> str:
        path = self.retrieve_model_dir_path(artifact_type, params)
        model_file_path = self.build_model_file_path(
            path, init_no=init_no, round_no=round_no
        )

        if not os.path.exists(model_file_path):
            raise ModelNotFoundError(
                f"""No matching model found with (artficat_type={artifact_type},
                match_condition={params}"""
            )

        return model_file_path

    def find_artifacts(
        self,
        artifact_type: str,
        match_condition: Dict[str, Any],
        exact_params: bool = False,
    ) -> List[Dict[str, Any]]:
        if exact_params:
            search_lambda = lambda: self._find_meta_by_exact_params(
                artifact_type, match_condition
            )
        else:
            search_lambda = lambda: self._find_meta(artifact_type, match_condition)

        raw_documents = Storage.locked_call(
            search_lambda,
            self._get_lock_path(artifact_type),
            self.lock_timeout,
        )

        documents = []
        default_ignore = set(ModelConfiguration.default_ignore())
        match_keyset = set(match_condition.keys())
        for document in raw_documents:
            document_id = document.doc_id
            document = dict(document)
            document["id"] = document_id
            doc_keyset = set(document["params"].keys()) - default_ignore
            diff = doc_keyset - match_keyset
            for key in list(diff):
                if key.startswith("ood_"):
                    diff.remove(key)
            if len(diff) == 0:
                documents.append(document)

        return documents


def create_storage(
    run_cfg: RunConfiguration,
    data_cfg: DataConfiguration,
    model_cfg: ModelConfiguration,
    train_cfg: TrainingConfiguration,
    ex: Optional[Experiment] = None,
):
    if run_cfg.job == "train" or (
        run_cfg.job in ("evaluate", "predict") and run_cfg.eval_experiment_name is None
    ):
        run_cfg.set_values(eval_experiment_name=run_cfg.experiment_name)

    storage = Storage(
        run_cfg.experiment_directory,
        experiment_name=run_cfg.experiment_name,
        eval_experiment_name=run_cfg.eval_experiment_name,
        experiment=ex,
        allow_override=run_cfg.retrain,
    )

    storage_params = {
        **model_cfg.to_dict(ignore=model_cfg.default_ignore()),
        **data_cfg.to_dict(ignore=data_cfg.default_ignore()),
        **train_cfg.to_dict(ignore=train_cfg.default_ignore()),
    }

    # ignore ood parameters for matching in an evasion setting
    if (
        run_cfg.job in ("evaluate", "predict")
        and data_cfg.ood_flag
        and data_cfg.ood_setting == "evasion"
    ):
        storage_params = {
            k: v for k, v in storage_params.items() if not k.startswith("ood_")
        }

    return storage, storage_params
