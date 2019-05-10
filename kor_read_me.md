# Keras: Deep Learning for humans

![Keras logo](https://s3.amazonaws.com/keras.io/img/keras-logo-2018-large-1200.png)

[![Build Status](https://travis-ci.org/keras-team/keras.svg?branch=master)](https://travis-ci.org/keras-team/keras)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/keras-team/keras/blob/master/LICENSE)

## You have just found Keras.

케라스는 파이썬으로 작성된 높은 수준의 인공신경망 API 이며, [TensorFlow](https://github.com/tensorflow/tensorflow), [CNTK](https://github.com/Microsoft/cntk), [Theano](https://github.com/Theano/Theano)에서 까지 구동 될 수 있습니다. 이러한 것은 빠른 실행을 가능하게 만드는 것을 목적으로 삼아 개발되었습니다. 좋은 연구를 해내는 대에 있어 가장 중요한 것은 가능한 적은 시간으로 아이디어에서 결과를 도출 해내는 것을 가능하게 만드는 것입니다.
아래와 같은 기능을 가진 딥러닝 라이브러리를 필요로 한다면 케라스를 사용하십시오.

* 유저 친숙성, 모듈성, 확장성을 통하여 쉽고 빠른 프로토 타이핑을 가능하도록 해줍니다. 
* 합성곱 신경망(Convolutional network) 과 현존하는 네트워크를 지원할 뿐만 아니라, 이 둘의 조합 또한 같이 지원합니다.
*CPU 와 GPU에서 완벽하게 구동됩니다.

자세한 내용은 [Keras.io](https://keras.io).에서 참고하십시오.
케라스는 __Python 2.7-3.6__ 과 호환됩니다.

------------------


## Guiding principles


- **유저친숙성** 케라스는 기계가 아닌 사람을 위해 디자인된 API 입니다. 이러한 특징은 유저를 최우선 이자 중심으로 생각합니다. 케라스는 인지적인 부담을 줄이는데에 있어 다음과 같은 최선의 업무를 수행합니다 : 케라스는 일관적이고 간단한 API를 제공하며, 평범하게 사용하는 데에 있어 필요한 유저의 행동을 최소한으로 줄이고, 유저의 에러에 대한 명확하고 활동적인 피드백을 제공합니다.

- **모듈성** 이 모델은 가능한 적은 제한으로 연결된, 완전히 구성 가능한 모듈의 독립 실행형 그래프 혹은 순서도 로 받아들여지고 있습니다. 특히나, neural layers, cost functions, optimizers, initialization schemes, activation functions and regularization schemes 들은 모두 새로운 모델과 결합될 수 있는 독립 실행형 모듈들 입니다.

- **쉬운 확장성** 새로운 모듈들은 새로운 클래스나 기능으로써 추가하기 쉬우며, 현존하는 모듈들은 충분한 예제를 제공해줍니다. 새로운 모듈 만들기를 쉽게 하는 것은 케라스를 심화된 연구에 적합하게 만들며 완전한 표현성을 보장합니다.

- **파이썬과 함께 작동** 선언적 형태의 분리된 모델 구성 파일은 존재하지 않습니다. 모델은 간결하고, 디버그 하기 쉬우며 확장의 안정성을 보장하는 파이썬 코드로 표시되어 있습니다. 

------------------


## Getting started: 30 seconds to Keras
케라스의 중요 데이터 구조는 레이어를 정돈하는 방법인 모델 입니다. 가장 간단한 타입의 모델은 선형 스택 구조의 레이어를 가진 순차적 모델입니다.  더 복잡한 구조를 위해서는 임의의 그래프나 레이어를 만들 수 있도록 해주는 케라스의 기능적 API를 사용해야만 합니다.

여기 순차적 모델이 있습니다.

```python
from keras.models import Sequential

model = Sequential()
```

스택 레이어는 .add() 만큼이나 쉽습니다.

```python
from keras.layers import Dense

model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))
```

당신의 모델이 괜찮게 보인다면, compile() 을 통해 확인해 보십시오.

```python
model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])
```

필요로만 한다면, 당신의 optimizer 을 통해 더 깊게 확인해 볼 수 있습니다. 케라스의 핵심적인 요소는 유저가 필요로 할 때 유저가 직접 컨트롤 할 수 있도록 하며 여러 것들을 합리적으로 간단하게 만드는 것입니다. 

```python
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True))
```

당신은 이제 당신의 연습용 데이터의 일괄 처리를 반복할 수 있습니다.

```python
# x_train and y_train are Numpy arrays --just like in the Scikit-Learn API.
model.fit(x_train, y_train, epochs=5, batch_size=32)
```

동시에, 당신은 당신의 모델에 일괄처리를 수동적으로 시행할 수 있습니다.

```python
model.train_on_batch(x_batch, y_batch)
```

당신의 수행결과를 한 줄로 평가하십시오.

```python
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)
```

또는 새로운 데이터에 대한 예상도를 만들어내 보십시오.

```python
classes = model.predict(x_test, batch_size=128)
```

시스템이나 이미지 분류 모델, Neural Turing Machine 이나 다른 모델에 답하며 질문을 만들어내는 것은 충분히 빠릅니다. 딥러닝의 뒤에 숨어있는 아이디어는 간단한데, 그것들을 실행하는 절차가 까다로워야만 할까요?

케라스의 듀토리얼에 대해 더 깊이 파고들고 싶다면, 이 곳을 참고해 보십시오

- [Getting started with the Sequential model](https://keras.io/getting-started/sequential-model-guide)
- [Getting started with the functional API](https://keras.io/getting-started/functional-api-guide)


레퍼지토리의 예시 파일 안에는, 쌓인 LSTMS를 이용한 텍스트 생성기, 메모리 네트워크를 이용한 질문 응답 기능 등 과 같은 심화된 모델들이 있습니다.

------------------


## Installation

케라스를 설치하기 이전에, TensorFlow, Theano, CNTK 중 하나를 기본 엔진으로서 설치해주시기 바랍니다. (TensorFlow추천)

- [TensorFlow installation instructions](https://www.tensorflow.org/install/).
- [Theano installation instructions](http://deeplearning.net/software/theano/install.html#install).
- [CNTK installation instructions](https://docs.microsoft.com/en-us/cognitive-toolkit/setup-cntk-on-your-machine).

다음 선택사항들을 설치하는 것도 고려하시기 바립니다.

- [cuDNN](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/) (GPU에서 케라스를 돌릴 예정이라면 추천).
- HDF5 and [h5py](http://docs.h5py.org/en/latest/build.html) (디스크에 케라스 모델을 저장할 예정이라면 필수).
- [graphviz](https://graphviz.gitlab.io/download/) and [pydot](https://github.com/erocarrera/pydot) (모델 그래프를 만들어내기 위해 [visualization utilities](https://keras.io/visualization/) 에 의해 사용됨).

그리고나서, 케라스를 직접적으로 설치하십시오. 케라스를 설치하는 데엔 두가지 방법이 있습니다.


- **케라스를 PyPl로부터 설치하기 (추천)**


이 설치 방식은 리눅스와 맥 환경에 친화되어 있으므로, 윈도우 환경에서 구동하길 원한다면 아래 코드에서 sudo 부분을 제거해야만 합니다.

```sh
sudo pip install keras
```

가상환경을 사용하고 있다면, sudo 사용을 원치 않을 수도 있습니다.

```sh
pip install keras
```

- **대체 방법:  케라스를 GitHub의 소스를 이용해 설치하기**

첫번째로, git를 사용하여 Keras를 clone하십시오.

```sh
git clone https://github.com/keras-team/keras.git
```

그리고나서, 케라스 폴더에 cd를 입력한 뒤 설치 명령을 실행하십시오.

```sh
cd keras
sudo python setup.py install
```

------------------


## Configuring your Keras backend

기본적으로, 케라스는 tensor 조종 라이브러리로써 텐서플로어를 사용할 것입니다. 케라스의 백엔드를 구성하고 싶다면 [이 지시](https://keras.io/backend/)를 따라주십시오.

------------------


## Support


질문을 할 수도 있고 개발 토론에 참여할 수도 있습니다.

*[케라스 구글 그룹](https://groups.google.com/forum/#!forum/keras-users)
*[케라스 Slack channel ](https://kerasteam.slack.com)을 이용하고 싶으시다면 이 [링크](https://keras-slack-autojoin.herokuapp.com/)를 사용해 초청장을 요청하십시오.
또한 **버그 신고나 개발 요청**은 Github issues에서만 다룰 수 있습니다. [가이드라인](https://github.com/keras-team/keras/blob/master/CONTRIBUTING.md)을 참고하여 주십시오.

------------------


## Why this name, Keras?

케라스는 그리스어로 뿔을 의미합니다. 이러한 것은 꿈의 영혼들이 인간들을 거짓된 환영으로 속이고, 상아의관문을 통해 지구에 도달하였으며, 언젠가 도달하게 될 미래를 알리고, 뿔의 관문을 통해 도착한 이들에 의해 나뉘어진 오디세이에서 처음으로 발견된 고대 그리스와 라틴의 문화로부터 만들어진 문학적 이미지를 참고하고 있습니다. 위의 것들은 모두 뿔과 충족, 그리고 상아와 속임 같은 단어들의 연극입니다.
케라스는 초반에 ONEIROS 프로젝트의 연구의 일환으로서 개발되었습니다. (Open-ended Neuro-Electronic Intelligent Robot Operating System).

>_"Oneiroi are beyond our unravelling --who can be sure what tale they tell? Not all that men look for comes to pass. Two gates there are that give passage to fleeting Oneiroi; one is made of horn, one of ivory. The Oneiroi that pass through sawn ivory are deceitful, bearing a message that will not be fulfilled; those that come out through polished horn have truth behind them, to be accomplished for men who see them."_ Homer, Odyssey 19. 562 ff (Shewring translation).

------------------