from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf

from tensorflow.python.platform import flags
import sys
sys.path.append("..")
from utils.utils import conv_block, fc, max_pool, lrn, dropout
from utils.utils import xent, kd, linear, dice_coefficient##, masked_dice_coefficient
from tensorflow.python.framework.ops import convert_to_tensor

FLAGS = flags.FLAGS

class MASF:
    def __init__(self,task='class'):
        """ Call construct_model_*() after initializing MASF"""
        # self.inner_lr = FLAGS.inner_lr
        # self.outer_lr = FLAGS.outer_lr
        # self.metric_lr = FLAGS.metric_lr
        self.SKIP_LAYER = ['fc8', 'fc9']
        if task=='class':
            self.forward = self.forward_alexnet
            # self.forward_metric_net = self.forward_metric_net
            self.construct_weights = self.construct_alexnet_weights
            # self.inner_lr = FLAGS.inner_lr
            # self.outer_lr = FLAGS.outer_lr
            # self.metric_lr = FLAGS.metric_lr
        elif task=='seg':
            self.forward = self.forward_unet
            # self.forward_metric_net = self.forward_metric_net
            self.construct_weights = self.construct_unet_weights

        self.loss_func = xent
        self.global_loss_func = kd
        self.WEIGHTS_PATH = './pretrained_weights/bvlc_alexnet.npy'
        self.task = task

    def construct_model_train(self, prefix='metatrain_'):
        # a: meta-train for inner update, b: meta-test for meta loss
        if self.task=='class':
            self.inputa = tf.placeholder(tf.float32)
            self.labela = tf.placeholder(tf.float32)
            self.inputa1= tf.placeholder(tf.float32)
            self.labela1= tf.placeholder(tf.float32)
            self.inputb = tf.placeholder(tf.float32)
            self.labelb = tf.placeholder(tf.float32)
        elif self.task=='seg':
            self.inputa = tf.placeholder(tf.float32,shape=[None,FLAGS.resize,FLAGS.resize,1])
            self.labela = tf.placeholder(tf.float32,shape=[None,FLAGS.resize,FLAGS.resize,2])
            self.maska = tf.placeholder(tf.float32,shape=[None,FLAGS.resize,FLAGS.resize])

            self.inputa1= tf.placeholder(tf.float32,shape=[None,FLAGS.resize,FLAGS.resize,1])
            self.labela1= tf.placeholder(tf.float32,shape=[None,FLAGS.resize,FLAGS.resize,2])
            self.maska1 = tf.placeholder(tf.float32,shape=[None,FLAGS.resize,FLAGS.resize])

            self.inputb = tf.placeholder(tf.float32,shape=[None,FLAGS.resize,FLAGS.resize,1])
            self.labelb = tf.placeholder(tf.float32,shape=[None,FLAGS.resize,FLAGS.resize,2])
            self.maskb = tf.placeholder(tf.float32,shape=[None,FLAGS.resize,FLAGS.resize])
        else:
            raise RuntimeError('check task type, class or seg.')
        self.inner_lr = tf.placeholder(tf.float32,shape=())
        self.outer_lr = tf.placeholder(tf.float32,shape=())
        # self.metric_lr = tf.placeholder(tf.float32,shape=())


        meta_sample_num = (FLAGS.meta_batch_size /3) * 3

        self.clip_value = FLAGS.gradients_clip_value
        self.margin = FLAGS.margin
        self.KEEP_PROB = tf.placeholder(tf.float32)

        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                print('weights already defined')
                training_scope.reuse_variables()
                weights = self.weights
            else:
                self.weights = weights = self.construct_weights()

            def task_metalearn(inp, reuse=True):
                # Function to perform meta learning update """
                inputa, inputa1, inputb, labela, labela1, labelb= inp


                # Obtaining the conventional task loss on meta-train
                task_outputa = self.forward(inputa, weights, reuse=reuse)
                task_lossa = self.loss_func(task_outputa, labela)
                task_outputa1 = self.forward(inputa1, weights, reuse=reuse)
                task_lossa1 = self.loss_func(task_outputa1, labela1)
                task_outputb = self.forward(inputb, weights,reuse= reuse)
                task_lossb = self.loss_func (task_outputb, labelb)

                ## perform inner update with plain gradient descent on meta-train

                task_output = [ task_lossa, task_lossa1, task_lossb]
                task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), 1), tf.argmax(labela, 1)) #this accuracy already gathers batch size
                task_accuracya1 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa1), 1), tf.argmax(labela1, 1))
                task_accuracyb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputb), 1), tf.argmax(labelb, 1))
                task_output.extend([task_accuracya, task_accuracya1, task_accuracyb])

                return task_output

            self.global_step = tf.Variable(0, trainable=False)
            # self.inner_lr = tf.train.exponential_decay(learning_rate=FLAGS.inner_lr, global_step=self.global_step, decay_steps=FLAGS.decay_steps, decay_rate=FLAGS.decay_rate)

            input_tensors = (self.inputa, self.inputa1, self.inputb,  self.labela, self.labela1, self.labelb)

            result = task_metalearn(inp=input_tensors)
            self.lossa_raw, self.lossa1_raw, self.lossb_raw, accuracya, accuracya1, accuracyb = result


        ## Performance & Optimization
        if 'train' in prefix:
            # if self.task=='class':
            #     self.lossa = avg_lossa = tf.reduce_mean(self.lossa_raw)
            #     self.lossa1 = avg_lossa1 = tf.reduce_mean(self.lossa1_raw)
            #     self.lossb = avg_lossb = tf.reduce_mean(self.lossb_raw)
            # elif self.task=='seg':
            #     self.lossa = avg_lossa = tf.reduce_sum(self.lossa_raw*self.maska)/tf.reduce_sum(self.maska)
            #     self.lossa1 = avg_lossa1 = tf.reduce_sum(self.lossa1_raw*self.maska1)/tf.reduce_sum(self.maska1)
            #     self.lossb = avg_lossb = tf.reduce_sum(self.lossb_raw*self.maskb)/tf.reduce_sum(self.maskb)

            self.lossa = avg_lossa = tf.reduce_mean(self.lossa_raw)
            self.lossa1 = avg_lossa1 = tf.reduce_mean(self.lossa1_raw)
            self.lossb = avg_lossb = tf.reduce_mean(self.lossb_raw)

            self.source_loss = (avg_lossa + avg_lossa1 + avg_lossb) / 3.0
            self.task_train_op = tf.train.AdamOptimizer(learning_rate=self.outer_lr).minimize(self.source_loss, global_step=self.global_step)

            self.accuracya = accuracya * 100.
            self.accuracya1 = accuracya1 * 100.
            self.accuracyb = accuracyb * 100.
            self.source_accuracy = (self.accuracya + self.accuracya1 + self.accuracyb) / 3.0


        ## Summaries
        tf.summary.scalar(prefix+'source_1 loss', self.lossa)
        tf.summary.scalar(prefix+'source_2 loss', self.lossa1)
        tf.summary.scalar(prefix + 'source_3 loss', self.lossb)

        tf.summary.scalar(prefix+'source_1 accuracy', self.accuracya)
        tf.summary.scalar(prefix+'source_2 accuracy', self.accuracya1)
        tf.summary.scalar(prefix + 'source_2 accuracy', self.accuracyb)

    def construct_model_test(self, prefix='test'):
        if self.task == 'class':
            self.test_input = tf.placeholder(tf.float32)
            self.test_label = tf.placeholder(tf.float32)
        elif self.task == 'seg':
            self.test_input = tf.placeholder(tf.float32,[None,FLAGS.resize,FLAGS.resize,1])
            self.test_label = tf.placeholder(tf.float32,[None,FLAGS.resize,FLAGS.resize,2])
            self.test_mask = tf.placeholder(tf.float32,[None,FLAGS.resize,FLAGS.resize])
        else:
            raise RuntimeError('check task type, class or seg.')

        with tf.variable_scope('model', reuse=None) as testing_scope:
            if 'weights' in dir(self):
                testing_scope.reuse_variables()
                weights = self.weights
            else:
                raise ValueError('Weights not initilized. Create training model before testing model')

            # self.semantic_feature, outputs = self.forward(self.test_input, weights)
            outputs = self.forward(self.test_input, weights)
            losses = self.loss_func(outputs, self.test_label)
            accuracies = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputs), -1), tf.argmax(self.test_label, -1))
            self.pred_prob = tf.nn.softmax(outputs)
            self.outputs = outputs
            if self.task == 'seg':
                self.dice_slice = dice_coefficient(y_true=tf.argmax(self.test_label, -1),y_pred=tf.argmax(tf.nn.softmax(outputs), -1))
                # self.masked_dice_slice = masked_dice_coefficient(y_true=tf.argmax(self.test_label, -1),y_pred=tf.argmax(tf.nn.softmax(outputs), -1),mask=self.test_mask)
                self.y_true = tf.argmax(self.test_label, -1)
                self.y_prob = tf.nn.softmax(outputs,-1)[:,:,:,1]
                self.y_pred = tf.argmax(tf.nn.softmax(outputs), -1)
            elif self.task=='class':
                self.y_true = tf.argmax(self.test_label, -1)
                self.y_pred = tf.argmax(tf.nn.softmax(outputs), -1)
                self.y_conf = tf.reduce_max(tf.nn.softmax(outputs), -1)
        self.test_loss = tf.reduce_mean(losses)
        self.test_acc = accuracies

    def load_initial_weights(self, session):
        """Load weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
        The weights come as a dict of lists (e.g. weights['conv1'] is a list)
        Load the weights into the model
        """
        weights_dict = np.load(self.WEIGHTS_PATH, allow_pickle= True, encoding='bytes').item()

        # Loop over all layer names stored in the weights dict
        for op_name in weights_dict:

            # Check if layer should be trained from scratch
            if op_name not in self.SKIP_LAYER:

                with tf.variable_scope('model', reuse=True):
                    with tf.variable_scope(op_name, reuse=True):

                        for data in weights_dict[op_name]:
                            # Biases
                            if len(data.shape) == 1:
                                var = tf.get_variable('biases', trainable=True)
                                session.run(var.assign(data))
                            # Weights
                            else:
                                var = tf.get_variable('weights', trainable=True)
                                session.run(var.assign(data))

    def construct_alexnet_weights(self):

        weights = {}
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
        fc_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)

        with tf.variable_scope('conv1') as scope:
            weights['conv1_weights'] = tf.get_variable('weights', shape=[11, 11, 3, 96], initializer=conv_initializer)
            weights['conv1_biases'] = tf.get_variable('biases', [96])

        with tf.variable_scope('conv2') as scope:
            weights['conv2_weights'] = tf.get_variable('weights', shape=[5, 5, 48, 256], initializer=conv_initializer)
            weights['conv2_biases'] = tf.get_variable('biases', [256])

        with tf.variable_scope('conv3') as scope:
            weights['conv3_weights'] = tf.get_variable('weights', shape=[3, 3, 256, 384], initializer=conv_initializer)
            weights['conv3_biases'] = tf.get_variable('biases', [384])

        with tf.variable_scope('conv4') as scope:
            weights['conv4_weights'] = tf.get_variable('weights', shape=[3, 3, 192, 384], initializer=conv_initializer)
            weights['conv4_biases'] = tf.get_variable('biases', [384])

        with tf.variable_scope('conv5') as scope:
            weights['conv5_weights'] = tf.get_variable('weights', shape=[3, 3, 192, 256], initializer=conv_initializer)
            weights['conv5_biases'] = tf.get_variable('biases', [256])

        with tf.variable_scope('fc6') as scope:
            weights['fc6_weights'] = tf.get_variable('weights', shape=[9216, 4096], initializer=conv_initializer)
            weights['fc6_biases'] = tf.get_variable('biases', [4096])

        with tf.variable_scope('fc7') as scope:
            weights['fc7_weights'] = tf.get_variable('weights', shape=[4096, 4096], initializer=conv_initializer)
            weights['fc7_biases'] = tf.get_variable('biases', [4096])

        with tf.variable_scope('fc8') as scope:
            weights['fc8_weights'] = tf.get_variable('weights', shape=[4096, 512], initializer=fc_initializer)
            weights['fc8_biases'] = tf.get_variable('biases', [512])

        with tf.variable_scope('fc9') as scope:
            weights['fc9_weights'] = tf.get_variable('weights', shape=[512, 2], initializer=fc_initializer)

        return weights

    def forward_alexnet(self, inp, weights, T=0.05, reuse=False):
        # reuse is for the normalization parameters.

        conv1 = conv_block(inp, weights['conv1_weights'], weights['conv1_biases'], stride_y=4, stride_x=4, groups=1,
                           reuse=reuse, scope='conv1')
        norm1 = lrn(conv1, 2, 1e-05, 0.75)
        pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID')

        # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
        conv2 = conv_block(pool1, weights['conv2_weights'], weights['conv2_biases'], stride_y=1, stride_x=1, groups=2,
                           reuse=reuse, scope='conv2')
        norm2 = lrn(conv2, 2, 1e-05, 0.75)
        pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID')

        # 3rd Layer: Conv (w ReLu)
        conv3 = conv_block(pool2, weights['conv3_weights'], weights['conv3_biases'], stride_y=1, stride_x=1, groups=1,
                           reuse=reuse, scope='conv3')

        # 4th Layer: Conv (w ReLu) splitted into two groups
        conv4 = conv_block(conv3, weights['conv4_weights'], weights['conv4_biases'], stride_y=1, stride_x=1, groups=2,
                           reuse=reuse, scope='conv4')

        # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
        conv5 = conv_block(conv4, weights['conv5_weights'], weights['conv5_biases'], stride_y=1, stride_x=1, groups=2,
                           reuse=reuse, scope='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID')

        # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
        flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
        fc6 = fc(flattened, weights['fc6_weights'], weights['fc6_biases'], activation='relu')
        dropout6 = dropout(fc6, self.KEEP_PROB)

        # 7th Layer: FC (w ReLu) -> Dropout
        fc7 = fc(dropout6, weights['fc7_weights'], weights['fc7_biases'], activation='relu')
        dropout7 = dropout(fc7, self.KEEP_PROB)
        self.semantic_feature = fc7

        # 8th Layer: FC and return unscaled activations
        fc8 = fc(dropout7, weights['fc8_weights'], weights['fc8_biases'])
        fc8 = tf.nn.l2_normalize(fc8, dim=1)
        w = tf.nn.l2_normalize(weights['fc9_weights'], dim=0)
        fc9 = linear(fc8, w)
        fc9 = fc9 / T

        return fc9

    def construct_unet_weights(self):

        weights = {}
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)

        with tf.variable_scope('conv1') as scope:
            weights['conv1_weights'] = tf.get_variable('weights', shape=[5, 5, 1, 16], initializer=conv_initializer)
            weights['conv1_biases'] = tf.get_variable('biases', [16])

        with tf.variable_scope('conv2') as scope:
            weights['conv2_weights'] = tf.get_variable('weights', shape=[5, 5, 16, 32], initializer=conv_initializer)
            weights['conv2_biases'] = tf.get_variable('biases', [32])

        ## Network has downsample here

        with tf.variable_scope('conv3') as scope:
            weights['conv3_weights'] = tf.get_variable('weights', shape=[3, 3, 32, 64], initializer=conv_initializer)
            weights['conv3_biases'] = tf.get_variable('biases', [64])

        with tf.variable_scope('conv4') as scope:
            weights['conv4_weights'] = tf.get_variable('weights', shape=[3, 3, 64, 64], initializer=conv_initializer)
            weights['conv4_biases'] = tf.get_variable('biases', [64])

        ## Network has downsample here

        with tf.variable_scope('conv5') as scope:
            weights['conv5_weights'] = tf.get_variable('weights', shape=[3, 3, 64, 128], initializer=conv_initializer)
            weights['conv5_biases'] = tf.get_variable('biases', [128])

        # with tf.variable_scope('conv6') as scope:
        #     weights['conv6_weights'] = tf.get_variable('weights', shape=[3, 3, 128, 128], initializer=conv_initializer)
        #     weights['conv6_biases'] = tf.get_variable('biases', [128])

        with tf.variable_scope('deconv1') as scope:
            weights['deconv1_weights'] = tf.get_variable('weights', shape=[3, 3, 64, 128], initializer=conv_initializer)

        with tf.variable_scope('conv7') as scope:
            weights['conv7_weights'] = tf.get_variable('weights', shape=[3, 3, 64, 64], initializer=conv_initializer)
            weights['conv7_biases'] = tf.get_variable('biases', [64])

        with tf.variable_scope('conv8') as scope:
            weights['conv8_weights'] = tf.get_variable('weights', shape=[3, 3, 64, 64], initializer=conv_initializer)
            weights['conv8_biases'] = tf.get_variable('biases', [64])

        with tf.variable_scope('deconv2') as scope:
            weights['deconv2_weights'] = tf.get_variable('weights', shape=[3, 3, 32, 64], initializer=conv_initializer)

        # with tf.variable_scope('conv9') as scope:
        #     weights['conv9_weights'] = tf.get_variable('weights', shape=[3, 3, 32, 32], initializer=conv_initializer)
        #     weights['conv9_biases'] = tf.get_variable('biases', [32])

        with tf.variable_scope('conv10') as scope:
            weights['conv10_weights'] = tf.get_variable('weights', shape=[3, 3, 32, 32], initializer=conv_initializer)
            weights['conv10_biases'] = tf.get_variable('biases', [32])

        with tf.variable_scope('output') as scope:
            weights['output_weights'] = tf.get_variable('weights', shape=[3, 3, 32, 2], initializer=conv_initializer)

        return weights


    def forward_unet(self, inp, weights,reuse=False):

        self.conv1 = conv_block1(inp, weights['conv1_weights'], weights['conv1_biases'])
        self.conv2 = conv_block1(self.conv1, weights['conv2_weights'], weights['conv2_biases'])
        self.pool2 = max_pool(self.conv2, 2, 2, 2, 2, padding='VALID')

        self.conv3 = conv_block1(self.pool2, weights['conv3_weights'], weights['conv3_biases'])
        self.conv4 = conv_block1(self.conv3, weights['conv4_weights'], weights['conv4_biases'])
        self.pool4 = max_pool(self.conv4, 2, 2, 2, 2, padding='VALID')

        self.conv5 = conv_block1(self.pool4, weights['conv5_weights'], weights['conv5_biases'])
        # self.conv6 = conv_block(self.conv5, weights['conv6_weights'], weights['conv6_biases'])

        ## add upsampling, meanwhile, channel number is reduced to half
        # self.up1 = deconv_block(self.conv6, weights['deconv1_weights'])
        self.up1 = deconv_block(self.conv5, weights['deconv1_weights'])

        self.conv7 = conv_block1(self.up1, weights['conv7_weights'], weights['conv7_biases'])
        self.conv8 = conv_block1(self.conv7, weights['conv8_weights'], weights['conv8_biases'])

        ## add upsampling, meanwhile, channel number is reduced to half
        self.up2 = deconv_block(self.conv8, weights['deconv2_weights'])

        # self.conv9 = conv_block(self.up2, weights['conv9_weights'], weights['conv9_biases'])
        # self.conv10 = conv_block(self.conv9, weights['conv10_weights'], weights['conv10_biases'])
        self.conv10 = conv_block1(self.up2, weights['conv10_weights'], weights['conv10_biases'])

        self.logits = tf.nn.conv2d(self.conv10, weights['output_weights'], strides=[1, 1, 1, 1], padding='SAME')

        # self.pred_prob = tf.nn.softmax(self.logits) # shape [batch, w, h, num_classes]
        # self.pred_compact = tf.argmax(self.pred_prob, axis=-1) # shape [batch, w, h]

        # return self.conv10, self.logits, self.pred_prob, self.pred_compact
        return self.logits


# # Network blocks
def conv_block1(inp, cweight, bweight):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    conv = tf.nn.conv2d(inp, cweight, strides=[1, 1, 1, 1], padding='SAME')
    conv = tf.nn.bias_add(conv, bweight)
    relu = tf.nn.leaky_relu(conv)
    return relu


def deconv_block(inp, cweight):
    inp_shape = tf.shape(inp)
    x_shape = inp.get_shape().as_list()
    output_shape = tf.stack([inp_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
    #output_shape = convert_to_tensor(np.array([x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2]))
    #x_shape = tf.shape(inp)
    # output_shape = [x_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2] 
    deconv = tf.nn.conv2d_transpose(inp, cweight, output_shape, strides=[1,2,2,1], padding='SAME')

    return deconv
