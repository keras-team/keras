import pytest

from keras.src import backend
from keras.src import layers
from keras.src import testing
from keras.src.utils.cuda_graph_utils import cuda_graph


def _torch_cuda_available():
    if backend.backend() != "torch":
        return False
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


class CudaGraphUtilsTest(testing.TestCase):
    def test_non_torch_backend_raises(self):
        if backend.backend() == "torch":
            self.skipTest("Test is for non-torch backends.")
        with self.assertRaisesRegex(ValueError, "torch backend"):
            cuda_graph(lambda x: x, sample_input=None)

    @pytest.mark.skipif(
        backend.backend() != "torch",
        reason="Remaining tests require the torch backend.",
    )
    def test_non_tensor_input_raises(self):
        with self.assertRaisesRegex(ValueError, "torch.Tensor"):
            cuda_graph(lambda x: x, sample_input=[1.0, 2.0, 3.0])

    @pytest.mark.skipif(
        backend.backend() != "torch",
        reason="Requires the torch backend.",
    )
    def test_cpu_tensor_raises(self):
        # A CPU tensor fails the device check whether or not CUDA is
        # present, so this runs on CPU-only torch as well.
        import torch

        with self.assertRaisesRegex(ValueError, "CUDA device"):
            cuda_graph(lambda x: x, sample_input=torch.zeros(2, 3))

    @pytest.mark.skipif(
        backend.backend() != "torch",
        reason="Requires the torch backend.",
    )
    def test_empty_sample_input_raises(self):
        for empty in ([], {}, ()):
            with self.assertRaisesRegex(ValueError, "at least one"):
                cuda_graph(lambda x: x, sample_input=empty)

    @pytest.mark.skipif(not _torch_cuda_available(), reason="Requires CUDA.")
    def test_replay_matches_eager(self):
        import torch

        model = layers.Dense(8)
        model.build((None, 4))
        x = torch.randn(3, 4, device="cuda")
        with torch.no_grad():
            eager = model(x)
        graphed = cuda_graph(model, x)
        replayed = graphed(x)
        # The capture buffer aliases the returned tensor. Convert to numpy
        # first so the comparison doesn't read stale data after a future
        # replay.
        self.assertAllClose(
            eager.detach().cpu().numpy(),
            replayed.detach().cpu().numpy(),
            atol=1e-5,
        )

    @pytest.mark.skipif(not _torch_cuda_available(), reason="Requires CUDA.")
    def test_wrong_shape_raises(self):
        import torch

        model = layers.Dense(8)
        model.build((None, 4))
        x = torch.randn(3, 4, device="cuda")
        graphed = cuda_graph(model, x)
        with self.assertRaisesRegex(ValueError, "shape"):
            graphed(torch.randn(5, 4, device="cuda"))

    @pytest.mark.skipif(not _torch_cuda_available(), reason="Requires CUDA.")
    def test_wrong_dtype_raises(self):
        import torch

        model = layers.Dense(8)
        model.build((None, 4))
        x = torch.randn(3, 4, device="cuda", dtype=torch.float32)
        graphed = cuda_graph(model, x)
        with self.assertRaisesRegex(ValueError, "dtype"):
            graphed(torch.randn(3, 4, device="cuda", dtype=torch.float16))

    @pytest.mark.skipif(not _torch_cuda_available(), reason="Requires CUDA.")
    def test_replay_reflects_new_input(self):
        import torch

        model = layers.Dense(4)
        model.build((None, 4))
        x1 = torch.randn(2, 4, device="cuda")
        x2 = torch.randn(2, 4, device="cuda")
        graphed = cuda_graph(model, x1)
        y1 = graphed(x1).detach().clone()
        y2 = graphed(x2).detach().clone()
        # Different inputs through the same captured graph must produce
        # different outputs.
        self.assertGreater(float((y1 - y2).abs().max()), 0.0)

    @pytest.mark.skipif(not _torch_cuda_available(), reason="Requires CUDA.")
    def test_nested_list_input(self):
        import torch

        def model(inputs):
            a, b = inputs
            return a + b

        a = torch.randn(2, 4, device="cuda")
        b = torch.randn(2, 4, device="cuda")
        graphed = cuda_graph(model, [a, b])
        with torch.no_grad():
            eager = model([a, b])
        replayed = graphed([a, b])
        self.assertAllClose(
            eager.detach().cpu().numpy(),
            replayed.detach().cpu().numpy(),
            atol=1e-5,
        )

    @pytest.mark.skipif(not _torch_cuda_available(), reason="Requires CUDA.")
    def test_nested_dict_input(self):
        import torch

        def model(inputs):
            return inputs["a"] * inputs["b"]

        a = torch.randn(3, 5, device="cuda")
        b = torch.randn(3, 5, device="cuda")
        graphed = cuda_graph(model, {"a": a, "b": b})
        with torch.no_grad():
            eager = model({"a": a, "b": b})
        replayed = graphed({"a": a, "b": b})
        self.assertAllClose(
            eager.detach().cpu().numpy(),
            replayed.detach().cpu().numpy(),
            atol=1e-5,
        )

    @pytest.mark.skipif(not _torch_cuda_available(), reason="Requires CUDA.")
    def test_wrong_structure_raises(self):
        import torch

        def model(inputs):
            a, b = inputs
            return a + b

        a = torch.randn(2, 4, device="cuda")
        b = torch.randn(2, 4, device="cuda")
        graphed = cuda_graph(model, [a, b])
        with self.assertRaisesRegex(ValueError, "structure"):
            graphed({"a": a, "b": b})
