"""
Upload to GCS:

```
gcloud config set project tensorflow
gcloud storage cp /Users/fchollet/Downloads/CaliforniaHousing/california_housing.npz gs://tensorflow/tf-keras-datasets/california_housing.npz
```
"""
import numpy as np

fpath = "/Users/fchollet/Downloads/CaliforniaHousing/cal_housing.data"
target_fpath = (
    "/Users/fchollet/Downloads/CaliforniaHousing/california_housing.npz"
)
header = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
    "MedHouseVal",
]


def process_dataset():
    f = open(fpath)
    contents = f.read()
    f.close()
    lines = contents.split("\n")
    samples = []
    targets = []
    for line in lines[:-1]:
        values = line.split(",")
        values = [float(v) for v in values]
        values, target = values[:-1], values[-1]
        samples.append(values)
        targets.append(target)
    samples = np.array(samples, dtype="float32")
    targets = np.array(targets, dtype="float32")
    print(samples.shape)
    print(targets.shape)
    np.savez(target_fpath, x=samples, y=targets)


def test_dataset():
    with np.load(target_fpath, allow_pickle=True) as f:
        x = f["x"]
        y = f["y"]
    print(x.shape)
    print(y.shape)

    print("Lowest y", sorted(y)[:10])
    print("Highest y", sorted(y, reverse=True)[:10])

    from keras_core.utils.file_utils import hash_file

    print(hash_file(target_fpath))


# process_dataset()
test_dataset()
