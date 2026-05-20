import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import torch

from keras.src.trainers.data_adapters.torch_data_loader_adapter import (
    TorchDataLoaderAdapter,
)


class ListDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, idx):
        return [torch.ones((3,)), torch.zeros((1,))]


def test_list_dataloader():
    ds = ListDataset()
    dataloader = torch.utils.data.DataLoader(ds, batch_size=4)
    adapter = TorchDataLoaderAdapter(dataloader)

    print("Testing get_numpy_iterator...")
    it = adapter.get_numpy_iterator()
    batch = next(it)
    print(f"Batch type: {type(batch)}")
    print(f"Batch len: {len(batch)}")

    print("\nTesting get_tf_dataset...")
    try:
        ds_tf = adapter.get_tf_dataset()
        batch_tf = next(iter(ds_tf))
        print(f"TF Batch type: {type(batch_tf)}")
        print(f"TF Batch len: {len(batch_tf)}")
    except Exception as e:
        print(f"FAILED to get TF dataset: {e}")
        # import traceback
        # traceback.print_exc()


if __name__ == "__main__":
    test_list_dataloader()
