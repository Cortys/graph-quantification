from gpn.data.dataset_manager import DatasetManager
from gpn.data.dataset_provider import InMemoryDatasetProvider


def load_dataset(dataset_name):
    dataset_provider = InMemoryDatasetProvider(
        DatasetManager(
            dataset=dataset_name,
            split_no=3,
            root="./data",
            ood_flag=False,
            train_samples_per_class=0.05,
            val_samples_per_class=0.15,
            test_samples_per_class=0.8,
            split="public" if dataset_name == "ogbn-arxiv" else "random",
            # ood_setting="poisoning",
            # ood_type="leave_out_classes",
            # ood_num_left_out_classes=-1,
            # ood_leave_out_last_classes=True,
        )
    )

    return dataset_provider


if __name__ == "__main__":
    dataset_name = "ogbn-arxiv"
    dataset_provider = load_dataset(dataset_name)
    print(f"Loaded dataset: {dataset_name}")
    dataset_provider.get_artifact("apsp")
