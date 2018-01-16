from keras import backend as K
from keras.engine.topology import Layer
from keras.engine.topology import InputSpec
import numpy as np
'''
以下是定义自主类的demo
实现一个自主定义的layer首先要让这个类继承于Layer类。每个自主类主要写__init__()、build()、和call()这三个部分。
__init__():用来进行初始化
build(input_shape)是用来定义层的权重（weight）
compute_output_shape(self, input_shape)这个层是用来指定输出的shape的比如一个全连接层，输入为（1024，16）输出shape为（1024，108），
我们就通过这个函数来进行修改。
call函数主要进行逻辑的计算，这是定义层功能的方法，如果你写的层不需要支持masking，你就只需要考虑的第一个参数x（输入张量）
'''
class MyLayer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim #初始化相应的参数
        #以下的调用是用来解决多继承的问题
        super(MyLayer, self).__init__(**kwargs)#确保写这句话

    def build(self, input_shape):
        # 为这个层创建训练用的权重

        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        '''这是add_weight的说明（由于语文水平有限，所以将原文档对于该函数参数的说明放在下面）
        # Arguments
            name: String, the name for the weight variable.
            shape: The shape tuple of the weight.
            dtype: The dtype of the weight.
            initializer: An Initializer instance (callable).
            regularizer: An optional Regularizer instance.
            trainable: A boolean, whether the weight should
                be trained via backprop or not (assuming
                that the layer itself is also trainable).
            constraint: An optional Constraint instance.
        # Returns
            The created weight variable.
        需要特殊说明的是trainable这个参数决定这个weight是否需要回传。
        
        
        '''


        super(MyLayer, self).build(input_shape)  # 确认调用这句话,这句话使得self.built = true

    def call(self, x):
        #x是上一个layer的结果，用来当作现在这个layer的输入
        return K.dot(x, self.kernel)
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


#以上是一个比较粗略的例子，以下是一个实际的例子，首先说一下这个层是用来做什么的。就是对于每一个通道进行归一化，不过通道使用的是不同的归一化参数，也就是说这个参数是需要进行学习的，因此需要通过 自定义层来完成。

class L2Normalization(Layer):
    '''
      Performs L2 normalization on the input tensor with a learnable scaling parameter
      as described in the paper "Parsenet: Looking Wider to See Better" (see references)
    and as used in the original SSD model.

     Arguments:
         gamma_init (int): The initial scaling parameter. Defaults to 20 following the
             SSD paper.

     Input shape:
         4D tensor of shape `(batch, channels, height, width)` if `dim_ordering = 'th'`
         or `(batch, height, width, channels)` if `dim_ordering = 'tf'`.

     Returns:
         The scaled tensor. Same shape as the input tensor.
    '''

    def __init__(self, gamma_init=20, **kwargs):
        #根据后端是什么来判断
        if K.image_dim_ordering() == 'tf':
            self.axis = 3
        else:
            self.axis = 1
        self.gamma_init = gamma_init
        super(L2Normalization, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)] #这句话是用来进行正确性检查的可以不用写
        gamma = self.gamma_init * np.ones((input_shape[self.axis],))
        self.gamma = K.variable(gamma, name='{}_gamma'.format(self.name)) #根据tf这个gamma必须先变成Variable才能进行运算
        self.trainable_weights = [self.gamma]  #设定训练的需要回传的weight，这个weight必须为list
        super(L2Normalization, self).build(input_shape)

    def call(self, x, mask=None):
        output = K.l2_normalize(x, self.axis) #此时调用了keras的backend的封装函数
        output *= self.gamma
        return output

    #由于这一层输入输出的shape是不变的所以不需要重写compute_output_shape这个函数
    #特殊说明这里的layer是对backend里的各种keras已经封装的tf组件的再一次封装，有什么需要的底层组件可以查看backend文件夹里头有详尽的对于tf各类底层代码封装函数的介绍