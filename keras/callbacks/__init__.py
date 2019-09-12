from __future__ import absolute_import

from .callbacks import Callback
from .callbacks import CallbackList
from .callbacks import BaseLogger
from .callbacks import TerminateOnNaN
from .callbacks import ProgbarLogger
from .callbacks import History
from .callbacks import ModelCheckpoint
from .callbacks import EarlyStopping
from .callbacks import RemoteMonitor
from .callbacks import LearningRateScheduler
from .callbacks import ReduceLROnPlateau
from .callbacks import CSVLogger
from .callbacks import LambdaCallback

from .. import backend as K

if K.backend() == 'tensorflow' and not K.tensorflow_backend._is_tf_1():
    from .tensorboard_v2 import TensorBoard
else:
    from .tensorboard_v1 import TensorBoard
