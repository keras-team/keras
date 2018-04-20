"""Model config
Credit:
Script modified from TensorFlow Benchmark repo:
https://github.com/tensorflow/benchmarks/blob/keras-benchmarks/scripts/keras_benchmarks/models/model_config.py
"""
from models import resnet50_benchmark
from models import resnet50_benchmark_tf_keras
from models import lstm_synthetic
from models import lstm_text_generation


def get_model_config(model_name):
    if model_name == 'resnet50':
        return resnet50_benchmark.Resnet50Benchmark()

    if model_name == 'resnet50_tf_keras':
        return resnet50_benchmark_tf_keras.Resnet50Benchmark()

    if model_name == 'lstm_synthetic':
        return lstm_synthetic.LstmBenchmark()

    if model_name == 'lstm_nietzsche':
        return lstm_text_generation.LstmBenchmark('nietzsche')

    if model_name == 'lstm_wikitext2':
        return lstm_text_generation.LstmBenchmark('wikitext2')
