# Torchvision to OpenVINO preprocessing converter

The Torchvision to OpenVINO preprocessing converter enables the use of an existing `torchvision.transforms` object to automatically translate it to OpenVINO preprocessing. It is then being embedded into the model, resulting in better inference performance.


## Supported transforms

Currently, the torchvision to OpenVINO preprocessing converter does not support all torchvision transforms.

Supported operations:
- `transforms.Compose`
- `transforms.Normalize`
- `transforms.ConvertImageDtype`
- `transforms.Grayscale`
- `transforms.Pad`
- `transforms.ToTensor`
- `transforms.CenterCrop`
- `transforms.Resize`

## Example usage

```python
preprocess_pipeline = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(256, interpolation=transforms.InterpolationMode.NEAREST),
        torchvision.transforms.CenterCrop((216, 218)),
        torchvision.transforms.Pad((2, 3, 4, 5), fill=3),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.ConvertImageDtype(torch.float32),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

torch_model = SimpleConvnet(input_channels=3)

torch.onnx.export(torch_model, torch.randn(1, 3, 224, 224), "test_convnet.onnx", verbose=False, input_names=["input"], output_names=["output"])
core = Core()
ov_model = core.read_model(model="test_convnet.onnx")

test_input = np.random.randint(255, size=(260, 260, 3), dtype=np.uint16)
ov_model = PreprocessConverter.from_torchvision(
    model=ov_model, transform=preprocess_pipeline, input_example=Image.fromarray(test_input.astype("uint8"), "RGB")
)
ov_model = core.compile_model(ov_model, "CPU")
ov_input = np.expand_dims(test_input, axis=0)
output = ov_model.output(0)
ov_result = ov_model(ov_input)[output]
```

## Key contacts

If you have any questions, feature requests or want us to review your PRs, send us a message or ping us on GitHub via [openvino-ie-python-api-maintainers](https://github.com/orgs/openvinotoolkit/teams/openvino-ie-python-api-maintainers). You can always directly contact everyone from this group.

## See also

* [OpenVINO™ README](../../../README.md)
* [OpenVINO™ Core Components](../../README.md)
* [OpenVINO™ Python API Reference](https://docs.openvino.ai/2025/api/ie_python_api/api.html)
* [OpenVINO™ Python API Advanced Inference](https://docs.openvino.ai/2025/openvino-workflow/running-inference/integrate-openvino-with-your-application/python-api-advanced-inference.html)
* [OpenVINO™ Python API Exclusives](https://docs.openvino.ai/2025/openvino-workflow/running-inference/integrate-openvino-with-your-application/python-api-exclusives.html)
