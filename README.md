# Keras: 인간을 위한 딥 러닝

![Keras logo](https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png)

이 저장소에서는 Keras 라이브러리 개발을 다룹니다. 설명서는 [keras.io](https://keras.io/)에서 읽어보시면 됩니다.

## keras에 대해

Keras는 Python으로 작성된 딥 러닝 API로, 
머신러닝 플랫폼 -[TensorFlow](https://github.com/tensorflow/tensorflow)에서 실행됩니다. 
그것은 빠른 실험을 가능하게 하는 데 초점을 두고 개발되었습니다. 
가능한 한 빨리 아이디어에서 결과로 옮겨갈 수 있는 것이 좋은 연구를 하는 열쇠입니다. 

Keras는:

Simple - 단순하지는 않습니다. 케라스는 개발자의 인지 부담을 줄여줌으로써 여러분이 정말로 중요한 부분에 집중할 수 있게 해줍니다.

Flexible - Keras는 복잡성의 점진적 공개 원칙을 채택합니다. 간단한 워크플로우는 빠르고 쉬워야 하며, 임의로 고급 워크플로우는 이미
           학습한 내용을 기반으로 하는 명확한 경로를 통해 가능해야 합니다.

Powerful - Keras는 업계 최고의 성능과 확장성을 제공합니다: 그것은 NASA, YouTube 또는 Waymo를 포함한 조직과 회사에서 사용됩니다.

## Keras & TensorFlow 2

텐서플로우2(https://www.tensorflow.org/)는 end-to-end 오픈 소스 머신러닝 플랫폼입니다. 
차별화 가능한 프로그래밍(https://en.wikipedia.org/wiki/Differentiable_programming)을 위한 인프라 계층이라고 생각할 수 있습니다. 
여기에는 다음 네 가지 핵심 기능이 결합되어 있습니다: 

- CPU, GPU 또는 TPU에서 낮은 수준의 텐서 작업을 효율적으로 실행합니다.
- 임의의 미분 가능한 식의 그라데이션 계산.
- 수백 개의 GPU 클러스터와 같은 여러 장치로 연산 확장.
- 서버, 브라우저, 모바일 및 임베디드 장치와 같은 외부 런타임으로 프로그램("그래프") 내보내기.

Keras는 TensorFlow 2의 고급 API로, 현대 딥 러닝에 중점을 두고 머신러닝 문제를 해결하기 위한 접근 가능하고 매우 생산적인 인터페이스입니다. 
반복 속도가 빠른 머신러닝 솔루션을 개발 및 전달하기 위한 필수 추상화 및 구성 요소를 제공합니다.

Keras는 엔지니어와 연구자가 TensorFlow 2의 확장성 및 교차 플랫폼 기능을 최대한 활용할 수 있도록 지원합니다. 
사용자는 TPU 또는 GPU의 대규모 클러스터에서 Keras를 실행할 수 있으며 브라우저 또는 모바일 장치에서 실행할 수 있도록 Keras 모델을 내보낼 수 있습니다.

---

## Keras 첫 번째 시도

Keras의 핵심 데이터 구조는  __layers__ 와 __models__ 입니다.
가장 간단한 모형 유형은 [`Sequential` model](/guides/sequential_model/), 선형 레이어 스택 입니다. 
보다 복잡한 아키텍처의 경우, 임의의 계층 그래프를 작성하거나 하위 클래싱 [write models entirely from scratch via subclasssing](/guides/making_new_layers_and_models_via_subclassing/)
을 통해 완전히 처음부터 모델을 작성할 수 있는 [Keras functional API](/guides/functional_api/)를 사용해야 합니다.

'Sequential' 모델:

```python
from tensorflow.keras.models import Sequential

model = Sequential()
```

레이어를 쌓는 것은 '.add()'만큼 쉽습니다:

```python
from tensorflow.keras.layers import Dense

model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=10, activation='softmax'))
```

모델이 정상으로 보이면 .compile()을 사용하여 학습 프로세스를 구성합니다:

```python
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```

필요한 경우 최적화 프로그램을 추가로 구성할 수 있습니다.
Keras의 원칙은 간단한 것을 유지하면서 사용자가 필요할 때 완전히 제어할 수 있도록 하는 것입니다(최종 제어는 하위 분류를 통한 소스 코드의 쉬운 확장성입니다).

```python
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.SGD(
                  learning_rate=0.01, momentum=0.9, nesterov=True))
```

이제 훈련 데이터를 일괄적으로 반복할 수 있습니다.

```python
# x_train and y_train are Numpy arrays.
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

테스트 손실과 메트릭을 한 줄로 평가합니다:

```python
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```
또는 새 데이터에 대한 예측을 생성합니다.

```python
classes = model.predict(x_test, batch_size=128)
```

방금 보신 것은 케라스를 이용하는 가장 기본적인 방법입니다.

그러나 Keras는 최첨단 연구 아이디어를 반복하기에 적합한 매우 유연한 프레임워크이기도 합니다. 
Keras는 복잡성의 점진적 공개의 원칙을 따릅니다: 
그것은 시작하기 쉽지만 임의적으로 앞선 사용 사례를 처리할 수 있게 하고 각 단계마다 점진적인 학습만 필요로 한다.

위의 간단한 신경망을 몇 줄로 훈련하고 평가할 수 있었던 것과 거의 같은 방법으로 Keras를 사용하여 새로운 훈련 절차나 이국적인 모델 아키텍처를 신속하게 개발할 수 있습니다. 
다음은 Keras 기능과 TensorFlow GradientTape를 결합한 로우 레벨 교육 루프 예입니다:

```python
import tensorflow as tf

# Prepare an optimizer.
optimizer = tf.keras.optimizers.Adam()
# Prepare a loss function.
loss_fn = tf.keras.losses.kl_divergence

# Iterate over the batches of a dataset.
for inputs, targets in dataset:
    # Open a GradientTape.
    with tf.GradientTape() as tape:
        # Forward pass.
        predictions = model(inputs)
        # Compute the loss value for this batch.
        loss_value = loss_fn(targets, predictions)

    # Get gradients of loss wrt the weights.
    gradients = tape.gradient(loss_value, model.trainable_weights)
    # Update the weights of the model.
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
```

Keras에 대한 자세한 튜토리얼을 보려면 다음 링크를 확인하십시오:

- [Introduction to Keras for engineers](https://keras.io/getting_started/intro_to_keras_for_engineers/)
- [Introduction to Keras for researchers](https://keras.io/getting_started/intro_to_keras_for_researchers/)
- [Developer guides](https://keras.io/guides/)

---

## 설치

Keras는 텐서플로 2와 함께 패키지로 제공됩니다. 
케라스 사용을 시작하려면 TensorFlow 2[install TensorFlow 2](https://www.tensorflow.org/install)를 설치하기만 하면 됩니다.

---

## 릴리스 및 호환성

Keras에는 **nightly releases** (`keras-nightly` PyPI)와 **stable releases**(keras` PyPI)가 있습니다. 
nightly Keras releases는 대개 해당 버전의 `tf-nightly` 릴리스와 호환됩니다(예: `keras-nightly==2.7.0.dev2021100607`은 `tf-nightly==2.7.0.dev2021100607`과 함께 사용해야 합니다). 
야간 릴리스의 이전 버전 호환성은 유지되지 않습니다. 
안정적인 릴리스를 위해 각 Keras 버전은 TensorFlow의 특정 안정 버전에 매핑됩니다.

아래 표에는 TensorFlow 버전과 Keras 버전 간의 호환성 버전 매핑이 나와 있습니다.

모든 브랜치들은 [Github](https://github.com/keras-team/keras/releases)에서 찾을 수 있습니다.

모든 릴리스 바이너리는 [Pypi](https://pypi.org/project/keras/#history)에서 찾을 수 있습니다.

| Keras release | Note      | Compatible Tensorflow version |
| -----------   | ----------- | -----------        |
| [2.4](https://github.com/keras-team/keras/releases/tag/2.4.0)  | Last stable release of multi-backend Keras | < 2.5
| 2.5-pre| Pre-release (not formal) for standalone Keras repo | >= 2.5 < 2.6
| [2.6](https://github.com/keras-team/keras/releases/tag/v2.6.0)    | First formal release of standalone Keras.  | >= 2.6 < 2.7
| [2.7](https://github.com/keras-team/keras/releases/tag/v2.7.0-rc0)    | (Upcoming release) | >= 2.7 < 2.8
| nightly|                                            | tf-nightly

---
## 지원

질문을 하고 개발 토론에 참여할 수 있습니다:

- [TensorFlow forum](https://discuss.tensorflow.org/).
- [Keras Google group](https://groups.google.com/forum/#!forum/keras-users).
- [Keras Slack channel](https://kerasteam.slack.com). 그 채널에 초대를 요청하기 위해 이 링크를[this link](https://keras-slack-autojoin.herokuapp.com/)사용하세요.

---

## 이슈 열기
[GitHub issues](https://github.com/keras-team/keras/issues)에서 버그 보고서 및 기능 요청(전용)을 게시할 수도 있습니다.


---

## PR 

기여를 환영합니다! PR을 시작하기 전에 기여자 가이드와 API 설계 가이드라인을 읽어보시기 바랍니다:
[기여 가이드](https://github.com/keras-team/keras/blob/master/CONTRIBUTING.md),
[API ](https://github.com/keras-team/governance/blob/master/keras_api_design_guidelines.md).
