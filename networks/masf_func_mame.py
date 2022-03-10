from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf
# try:
#     import special_grads
# except KeyError as e:
#     print('WARN: Cannot define MaxPoolGrad, likely already defined for this version of tensorflow: %s' % e, file=sys.stderr)

from tensorflow.python.platform import flags
# from utils import conv_block,fc, max_pool, lrn, dropout, linear, dice_coefficient
# from utils import xent, xent2, Coral_loss, kd_class,  g_loss, get_center_loss, center_loss_global, center_loss, cos_center_loss, kd
from utils import *

FLAGS = flags.FLAGS

class MASF:
    def __init__(self,task='class'):
        """ Call construct_model_*() after initializing MASF"""
        # self.inner_lr = FLAGS.inner_lr
        # self.outer_lr = FLAGS.outer_lr
        # self.metric_lr = FLAGS.metric_lr
        self.SKIP_LAYER = ['fc8']
        if task=='class':
            self.forward = self.forward_alexnet
            # self.forward_metric_net = self.forward_metric_net
            self.construct_weights = self.construct_alexnet_weights
            self.featurelen = 512
            self.global_loss_func3 = kd
            self.global_loss_func1 = g_loss

        elif task=='seg':
            self.forward = self.forward_unet
            # self.forward_metric_net = self.forward_metric_net
            self.construct_weights = self.construct_unet_weights#self.construct_alexnet_weights
            self.featurelen = 16
            self.global_loss_func3 = masked_kd
            self.global_loss_func1 = masked_g_loss

        self.construct_cla_weights = self.construct_classifiar_weights
        self.loss_func = xent
        self.loss_func2 = xent
        self.loss_func1 = get_center_loss
        # self.global_loss_func1 = g_loss
        self.global_loss_func = cos_center_loss
        self.global_loss_func2 = center_loss_global
        
        self.sm_loss = g_loss
        self.num_classes = 2
        self.WEIGHTS_PATH = './pretrained_weights/bvlc_alexnet.npy'

        self.alpha0 = FLAGS.a0 # 
        self.alpha1 = FLAGS.a1 
        self.alpha2 = FLAGS.a2
        self.task = task

    def construct_model_train(self, prefix='metatrain_'):
        if self.task=='class':
            # a: meta-train for inner update, b: meta-test for meta loss
            self.inputa = tf.placeholder(tf.float32)
            self.labela = tf.placeholder(tf.float32)

            self.inputa1= tf.placeholder(tf.float32)
            self.labela1= tf.placeholder(tf.float32)

            self.inputb = tf.placeholder(tf.float32)
            self.labelb = tf.placeholder(tf.float32)

            meta_sample_num = (FLAGS.meta_batch_size) * 1
            # self.input_group = tf.placeholder(tf.float32)
            # self.label_group = tf.placeholder(tf.float32)
            # self.label_groupm = tf.placeholder(tf.int32, shape=(meta_sample_num,))
            # self.label_am = tf.placeholder(tf.int32, shape=(meta_sample_num,))
            # self.label_a1m = tf.placeholder(tf.int32, shape=(meta_sample_num,))
            # self.label_bm = tf.placeholder(tf.int32, shape=(meta_sample_num,))

        elif self.task=='seg':
            self.inputa = tf.placeholder(tf.float32,shape=[None,FLAGS.resize,FLAGS.resize,1])
            self.labela = tf.placeholder(tf.float32,shape=[None,FLAGS.resize,FLAGS.resize,2])
            self.maska = tf.placeholder(tf.float32,shape=[None,FLAGS.resize,FLAGS.resize])

            self.inputa1 = tf.placeholder(tf.float32,shape=[None,FLAGS.resize,FLAGS.resize,1])
            self.labela1 = tf.placeholder(tf.float32,shape=[None,FLAGS.resize,FLAGS.resize,2])
            self.maska1 = tf.placeholder(tf.float32,shape=[None,FLAGS.resize,FLAGS.resize])

            self.inputb = tf.placeholder(tf.float32,shape=[None,FLAGS.resize,FLAGS.resize,1])
            self.labelb = tf.placeholder(tf.float32,shape=[None,FLAGS.resize,FLAGS.resize,2])
            self.maskb = tf.placeholder(tf.float32,shape=[None,FLAGS.resize,FLAGS.resize])

            meta_sample_num = (FLAGS.meta_batch_size) * 1
            # self.input_group = tf.placeholder(tf.float32,shape=[None,FLAGS.resize,FLAGS.resize,1])
            # self.label_group = tf.placeholder(tf.float32,shape=[None,FLAGS.resize,FLAGS.resize,2])
            # self.label_groupm = tf.placeholder(tf.int32, shape=(None,FLAGS.resize,FLAGS.resize))
            # self.label_am = tf.placeholder(tf.int32, shape=(None,FLAGS.resize,FLAGS.resize))
            # self.label_a1m = tf.placeholder(tf.int32, shape=(None,FLAGS.resize,FLAGS.resize))
            # self.label_bm = tf.placeholder(tf.int32, shape=(None,FLAGS.resize,FLAGS.resize))


        # self.domaina = tf.placeholder(tf.int32)
        # self.domaina1= tf.placeholder(tf.int32)
        # self.domainb = tf.placeholder(tf.int32)
        self.bool_indicator_b_a = tf.placeholder(tf.float32, shape=(self.num_classes,))
        self.bool_indicator_b_a1 = tf.placeholder(tf.float32, shape=(self.num_classes,))

    
        self.clip_value = FLAGS.gradients_clip_value
        self.margin = FLAGS.margin
        self.KEEP_PROB = tf.placeholder(tf.float32)
        self.inner_lr = tf.placeholder(tf.float32,shape=())
        self.outer_lr = tf.placeholder(tf.float32,shape=())

        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                print('weights already defined')
                training_scope.reuse_variables()
                weights = self.weights
            else:
                self.weights = weights = self.construct_weights()

            if 'cla_weights' in dir(self):
                print('weights already defined')
                training_scope.reuse_variables()
                cla_weights = self.cla_weights
            else:
                self.cla_weights = cla_weights = self.construct_cla_weights()
            # if 'source_moving_1_centroid' in dir(self):
            #     print('centroids already defined')
            #     training_scope.reuse_variables()
            #     source_moving_1_centroid = self.source_moving_1_centroid
            # else:
            #     self.source_moving_1_centroid = source_moving_1_centroid = tf.zeros((self.num_classes, self.featurelen),
            #                                                                         dtype=tf.float32) + 1e-10
            #
            # if 'source_moving_2_centroid' in dir(self):
            #     print('centroids already defined')
            #     training_scope.reuse_variables()
            #     source_moving_2_centroid = self.source_moving_2_centroid
            # else:
            #     self.source_moving_2_centroid = source_moving_2_centroid = tf.zeros(
            #         (self.num_classes, self.featurelen), dtype=tf.float32) + 1e-10
            #
            # if 'source_moving_3_centroid' in dir(self):
            #     print('centroids already defined')
            #     training_scope.reuse_variables()
            #     source_moving_3_centroid = self.source_moving_3_centroid
            # else:
            #     self.source_moving_3_centroid = source_moving_3_centroid = tf.zeros((self.num_classes, self.featurelen),
            #                                                                         dtype=tf.float32) + 1e-10

            def task_metalearn(inp, global_bool_indicator, reuse=True):
                # Function to perform meta learning update """
                # inputa, inputa1, inputb, input_group, labela, labela1, labelb, label_group, labelam, labela1m, labelbm, label_groupm, domaina, domaina1, domainb = inp
                inputa, inputa1, inputb, labela, labela1, labelb = inp
                global_bool_indicator_b_a, global_bool_indicator_b_a1 = global_bool_indicator

                # Obtaining the conventional task loss on meta-train

                class_centroid = tf.nn.l2_normalize(cla_weights['conv9_weights'], dim=0)
                class_centroid = tf.transpose(class_centroid, [1, 0])
                task_embeddinga = self.forward(inputa, weights, reuse=reuse)
                task_outputa = self.forward_classifier(task_embeddinga, cla_weights)
                task_lossa = self.loss_func2(task_outputa, labela)
                # p_a = self.global_loss_func(task_embeddinga, labela, class_centroid)

                # center_loss_a = self.loss_func1(task_embeddinga, labela, class_centroid)

                # w_a = self.global_loss_func2(task_outputa, labela, class_centroid)

                task_embeddinga1 = self.forward(inputa1, weights, reuse=reuse)
                task_outputa1 = self.forward_classifier(task_embeddinga1, cla_weights)
                task_lossa1 = self.loss_func2(task_outputa1, labela1)

                #center_loss_a1 = self.loss_func1(task_embeddinga1, labela1, class_centroid)
                
                # p_a1 = self.global_loss_func( task_embeddinga1, labela1, class_centroid)
                # w_a1 = self.global_loss_func2(task_outputa1, labela1, class_centroid)

                task_embeddingb = self.forward(inputb, weights, reuse=reuse)
                task_outputb = self.forward_classifier(task_embeddingb, cla_weights)

                # perform inner update with plain gradient descent on meta-train
                fast_weights = weights


                ## compute global loss
                # center_loss_a = self.loss_func1(task_embeddinga, labela, current_source_a_centroid)
                # center_loss_a1 = self.loss_func1(task_embeddinga1, labela1, current_source_a1_centroid)

                # grads = tf.gradients( 1*(task_lossa +  task_lossa1) + 0.*(center_loss_a + center_loss_a1),
                #                      list(cla_weights.values()))

                grads = tf.gradients( 1*(task_lossa +  task_lossa1), list(cla_weights.values()))


                grads = [tf.stop_gradient(grad) for grad in grads] # first-order gradients approximation
                gradients = dict(zip(cla_weights.keys(), grads))

                fast_class = dict(zip(cla_weights.keys(), [cla_weights[key] - self.inner_lr * tf.clip_by_norm(gradients[key], clip_norm=self.clip_value) for key in cla_weights.keys()]))
                # fast_class = cla_weights

                class_centroid = tf.nn.l2_normalize(fast_class['conv9_weights'], dim=0)
                class_centroid = tf.transpose( class_centroid, [1, 0])
                if self.task == 'seg':
                    assert self.featurelen==task_embeddinga.shape[-1]==task_embeddinga1.shape[-1]==task_embeddingb.shape[-1]
                    assert self.num_classes == labela.shape[-1]==labela1.shape[-1]==labelb.shape[-1]
                    task_embeddinga = tf.reshape(task_embeddinga,(-1,self.featurelen))
                    task_embeddinga1 = tf.reshape(task_embeddinga1,(-1,self.featurelen))
                    task_embeddingb = tf.reshape(task_embeddingb,(-1,self.featurelen))
                    labela = tf.reshape(labela,(-1,self.num_classes))
                    labela1 = tf.reshape(labela1,(-1,self.num_classes))
                    labelb = tf.reshape(labelb,(-1,self.num_classes))
                    maska = tf.reshape(self.maska,(-1,))
                    maska1 = tf.reshape(self.maska1,(-1,))
                    maskb = tf.reshape(self.maskb,(-1,))

                # useless
                center_loss_a1 = self.loss_func1(task_embeddinga1, labela1, class_centroid)
                center_loss_a = self.loss_func1(task_embeddinga, labela, class_centroid)
                center_loss_b = self.loss_func1(task_embeddingb, labelb, class_centroid)

                current_source_a_centroid = tf.zeros((self.num_classes, self.featurelen), dtype=tf.float32)
                current_source_a1_centroid = tf.zeros((self.num_classes, self.featurelen), dtype=tf.float32)
                current_source_b_centroid = tf.zeros((self.num_classes, self.featurelen), dtype=tf.float32)
                E = tf.eye(self.num_classes)
                eps = 1e-10
                domain_labela = tf.multiply(1.0, labela)
                domain_labelb = tf.multiply(1.0, labelb)
                domain_labela1 = tf.multiply(1.0, labela1)

                new_task_embeddinga = self.forward(inputa, fast_weights, reuse=reuse)
                new_task_embeddinga1 = self.forward(inputa1, fast_weights, reuse=reuse)
                new_task_embeddingb = self.forward(inputb, fast_weights, reuse=reuse)

                if self.task == 'seg':
                    assert self.featurelen==new_task_embeddinga.shape[-1]==new_task_embeddinga1.shape[-1]==new_task_embeddingb.shape[-1]
                    new_task_embeddinga = tf.reshape(new_task_embeddinga,(-1,self.featurelen))
                    new_task_embeddinga1 = tf.reshape(new_task_embeddinga1,(-1,self.featurelen))
                    new_task_embeddingb = tf.reshape(new_task_embeddingb,(-1,self.featurelen))

                    #计算 域类原型
                    maska_ = tf.expand_dims(maska,-1)
                    maska1_ = tf.expand_dims(maska1,-1)
                    maskb_ = tf.expand_dims(maskb,-1)
                    for i in range(self.num_classes):

                        current_source_a_centroid += tf.matmul(tf.expand_dims(E[:, i], 1), tf.expand_dims(
                            tf.reduce_sum( tf.multiply(new_task_embeddinga, tf.expand_dims(domain_labela[:, i], 1)) * maska_ ,
                                        reduction_indices=0), 0)) / (tf.reduce_sum(domain_labela[:, i]*maska) + eps) 
                        current_source_a1_centroid += tf.matmul(tf.expand_dims(E[:, i], 1), tf.expand_dims(
                            tf.reduce_sum(tf.multiply(new_task_embeddinga1, tf.expand_dims(domain_labela1[:, i], 1)) * maska1_,
                                        reduction_indices=0), 0)) / (tf.reduce_sum(domain_labela1[:, i]*maska1) + eps)
                        current_source_b_centroid += tf.matmul(tf.expand_dims(E[:, i], 1), tf.expand_dims(
                            tf.reduce_sum(tf.multiply(new_task_embeddingb, tf.expand_dims(domain_labelb[:, i], 1)) * maskb_,
                                        reduction_indices=0), 0)) / (tf.reduce_sum(domain_labelb[:, i]*maskb) + eps)
                elif self.task == 'class':
                    for i in range(self.num_classes):
                        current_source_a_centroid += tf.matmul(tf.expand_dims(E[:, i], 1), tf.expand_dims(
                            tf.reduce_sum(tf.multiply(new_task_embeddinga, tf.expand_dims(domain_labela[:, i], 1)),
                                        reduction_indices=0), 0)) / (tf.reduce_sum(domain_labela[:, i]) + eps) 
                        current_source_a1_centroid += tf.matmul(tf.expand_dims(E[:, i], 1), tf.expand_dims(
                            tf.reduce_sum(tf.multiply(new_task_embeddinga1, tf.expand_dims(domain_labela1[:, i], 1)),
                                        reduction_indices=0), 0)) / (tf.reduce_sum(domain_labela1[:, i]) + eps)
                        current_source_b_centroid += tf.matmul(tf.expand_dims(E[:, i], 1), tf.expand_dims(
                            tf.reduce_sum(tf.multiply(new_task_embeddingb, tf.expand_dims(domain_labelb[:, i], 1)),
                                        reduction_indices=0), 0)) / (tf.reduce_sum(domain_labelb[:, i]) + eps)

                outb_a = tf.matmul(new_task_embeddingb, current_source_a_centroid, transpose_b=True) / 0.1 #(?,16)(2,16)_T =>(?,2)
                outb_a1 = tf.matmul(new_task_embeddingb, current_source_a1_centroid, transpose_b=True) / 0.1
                outb_b = tf.matmul(new_task_embeddingb, class_centroid, transpose_b=True) / 0.1

                if self.task=='seg':
                    metric_loss = ( self.global_loss_func3( outb_a1, outb_b, maskb) + self.global_loss_func3( outb_a, outb_b, maskb)) / 2 #L_ca,/apha2
                    metric_loss = metric_loss/(FLAGS.meta_batch_size*FLAGS.resize*FLAGS.resize/120)
                elif self.task=='class':
                    metric_loss = ( self.global_loss_func3( outb_a1, outb_b) + self.global_loss_func3( outb_a, outb_b)) / 2 #L_ca,/apha2
                    metric_loss = metric_loss

                if self.task=='seg':
                    global_loss_a = self.global_loss_func1(task_embeddinga, class_centroid, labela, maska,
                                                            FLAGS.meta_batch_size*FLAGS.resize*FLAGS.resize,self.num_classes, self.featurelen, margin=1)
                    global_loss_a1 = self.global_loss_func1(task_embeddinga1, class_centroid, labela1, maska1,
                                                            FLAGS.meta_batch_size*FLAGS.resize*FLAGS.resize, self.num_classes, self.featurelen, margin=1)
                    global_loss_b = self.global_loss_func1(task_embeddingb, class_centroid, labelb, maskb,
                                                            FLAGS.meta_batch_size*FLAGS.resize*FLAGS.resize, self.num_classes, self.featurelen, margin=1)
                    global_loss = ( global_loss_b + global_loss_a1 + global_loss_a) / 3 #L_fa, /apha1
                    
                    task_outputb = self.forward_classifier(task_embeddingb, fast_class)
                    task_lossb = self.loss_func2(task_outputb, labelb)
                    sm_loss = tf.reduce_mean( task_lossb)

                    task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), -1), tf.reshape(tf.argmax(labela, -1),(-1,FLAGS.resize,FLAGS.resize))) #this accuracy already gathers batch size
                    task_accuracya1 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa1), -1), tf.reshape(tf.argmax(labela1, -1),(-1,FLAGS.resize,FLAGS.resize)))
                    task_accuracyb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputb), -1), tf.argmax(labelb, -1))

                elif self.task=='class':
                    global_loss_a1 = self.global_loss_func1(task_embeddinga, class_centroid, labela,
                                                            FLAGS.meta_batch_size,self.num_classes, self.featurelen, margin=1)
                    global_loss_a = self.global_loss_func1(task_embeddinga1, class_centroid, labela1,
                                                            FLAGS.meta_batch_size, self.num_classes, self.featurelen, margin=1)
                    global_loss_b = self.global_loss_func1(task_embeddingb, class_centroid, labelb,
                                                            FLAGS.meta_batch_size, self.num_classes, self.featurelen, margin=1)
                    global_loss = ( global_loss_b + global_loss_a1 + global_loss_a) / 3

                    task_outputb = self.forward_classifier(task_embeddingb, fast_class)
                    task_lossb = self.loss_func2(task_outputb, labelb)
                    sm_loss = tf.reduce_mean( task_lossb)

                    # task_output = [ global_loss,  metric_loss, sm_loss, task_lossa+0.*center_loss_a , task_lossa1+0.*center_loss_a1, task_lossb]
                    task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), -1), tf.argmax(labela, -1)) #this accuracy already gathers batch size
                    task_accuracya1 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa1), -1), tf.argmax(labela1, -1))
                    task_accuracyb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputb), -1), tf.argmax(labelb, -1))

                task_output = [ global_loss,  metric_loss, sm_loss, task_lossa+0.*center_loss_a , task_lossa1+0.*center_loss_a1, task_lossb]
                task_output.extend([task_accuracya, task_accuracya1, task_accuracyb])

                return task_output

            self.global_step = tf.Variable(0, trainable=False)

            # input_tensors = (self.inputa, self.inputa1, self.inputb, self.input_group, self.labela, self.labela1, self.labelb, self.label_group,self.label_am,self.label_a1m,self.label_bm, self.label_groupm, self.domaina, self.domaina1, self.domainb)
            input_tensors = (self.inputa, self.inputa1, self.inputb, self.labela, self.labela1, self.labelb)
            global_bool_indicator = (self.bool_indicator_b_a, self.bool_indicator_b_a1)

            result = task_metalearn(inp=input_tensors, global_bool_indicator=global_bool_indicator)
            global_loss,  metric_loss, sm_loss, self.lossa_raw, self.lossa1_raw, self.lossb_raw, accuracya, accuracya1, accuracyb = result
            self.sm_loss = sm_loss * self.alpha0
            self.global_loss = global_loss * self.alpha1
            self.metric_loss = metric_loss * self.alpha2

            self.ori_global_loss = global_loss
            self.ori_metric_loss = metric_loss
            self.ori_sm_loss = sm_loss
            # self.source_losses = source_loss * 1

        ## Performance & Optimization
        if 'train' in prefix:

            self.lossa = avg_lossa = tf.reduce_mean(self.lossa_raw)
            self.lossa1 = avg_lossa1 = tf.reduce_mean(self.lossa1_raw)
            self.lossb = avg_lossb = tf.reduce_mean(self.lossb_raw)
            self.source_loss = (avg_lossa + avg_lossa1) / 2.0
            # self.outer_lr = tf.train.exponential_decay(learning_rate=FLAGS.outer_lr, global_step=self.global_step,
            #                                            decay_steps=10000, decay_rate=0.96)
            
            #L_task on D_tr
            self.task_train_op = tf.train.AdamOptimizer(learning_rate=self.outer_lr).minimize( self.source_loss, global_step=self.global_step)


            self.accuracya = accuracya * 100.
            self.accuracya1 = accuracya1 * 100.
            self.accuracyb = accuracyb * 100.
            self.source_accuracy = (self.accuracya + self.accuracya1 + self.accuracyb) / 3.0

            var_list_feature_extractor = [v for v in tf.trainable_variables() if (v.name.split('/')[1] not in ['fc9'])]
            var_list_classifier = [v for v in tf.trainable_variables() if (v.name.split('/')[1] in ['fc9'])]
            # var_list_classifier = [v for v in tf.trainable_variables() if 'classifier' in v.name.split('/')]
            # self.metric_lr = FLAGS.metric_lr


            optimizer = tf.train.AdamOptimizer(self.outer_lr)
            #L_task on D_te
            gvs = optimizer.compute_gradients( self.global_loss + self.metric_loss + self.sm_loss ,
                                              var_list = var_list_feature_extractor + var_list_classifier)  # observe stability of gradients for meta loss
            l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
            for grad, var in gvs:
                tf.summary.histogram("gradients_norm/" + var.name, l2_norm(grad))
                tf.summary.histogram("feature_extractor_var_norm/" + var.name, l2_norm(var))
                tf.summary.histogram('gradients/' + var.name, var)
                tf.summary.histogram("feature_extractor_var/" + var.name, var)

            gvs = [(tf.clip_by_norm(grad, clip_norm=self.clip_value), var) for grad, var in gvs]

            for grad, var in gvs:
                tf.summary.histogram("gradients_norm_clipped/" + var.name, l2_norm(grad))
                tf.summary.histogram('gradients_clipped/' + var.name, var)

            self.meta_train_op = optimizer.apply_gradients(gvs)

        ## Summaries
        tf.summary.scalar(prefix+'source loss', self.source_loss)
        # tf.summary.scalar(prefix+'source_2 loss', self.lossa1)
        tf.summary.scalar(prefix+'source_1 accuracy', self.accuracya)
        tf.summary.scalar(prefix+'source_2 accuracy', self.accuracya1)
        tf.summary.scalar(prefix+'global loss', self.global_loss)
        tf.summary.scalar(prefix+'metric loss', self.metric_loss)
        tf.summary.scalar(prefix+'sm loss', self.sm_loss)

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

            if 'cla_weights' in dir(self):
                testing_scope.reuse_variables()
                cla_weights = self.cla_weights
            else:
                raise ValueError('Weights not initilized. Create training model before testing model')

            # self.semantic_feature = self.forward(self.test_input, weights)
            # outputs = self.forward_classifier(self.semantic_feature, cla_weights)
            # losses = self.loss_func(outputs, self.test_label)
            # accuracies = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputs), 1), tf.argmax(self.test_label, 1))
            # self.pred_prob = tf.nn.softmax(outputs)
            # self.outputs = outputs
            feature = self.forward(self.test_input, weights)
            outputs = self.forward_classifier(feature,cla_weights)
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
            weights['fc8_weights'] = tf.get_variable('weights', shape=[4096, self.featurelen], initializer=conv_initializer)
            weights['fc8_biases'] = tf.get_variable('biases', [self.featurelen])

        return weights

    def construct_classifiar_weights(self):

        weights = {}
        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
        # fc_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)

        with tf.variable_scope('fc9') as scope:
            weights['conv9_weights'] = tf.get_variable('weights', shape=[ self.featurelen, self.num_classes], initializer=conv_initializer)
        return weights
    
    def forward_alexnet(self, inp, weights, reuse=False):
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

        fc8 = fc(dropout7, weights['fc8_weights'], weights['fc8_biases'], activation='relu')

        self.semantic_feature_last = fc8

        fc8 = tf.nn.l2_normalize(fc8, dim=1)

        return fc8

    def forward_classifier(self, inp, weights, T=0.05, reuse=False):

        # inp = tf.nn.l2_normalize(inp, dim=1)
        # inp = dropout(inp, self.KEEP_PROB)
        w = tf.nn.l2_normalize( weights['conv9_weights'], dim=0)
        fc8 = linear(inp,  w)
        fc8 = fc8 / T

        return fc8

    def global_loss(self,source_feature, target_feature, class_num, margin):
        label = tf.to_float(tf.eye(7))
        g_loss = 0
        margin = tf.to_float(tf.constant(margin))
        for cls in range(class_num):
            for row in range(class_num):
            # feature = tf.tile(target_feature[cls, :], [7, 1])
                d = tf.square(tf.subtract(source_feature[cls,:], target_feature[row,:]))
                d_sqrt = tf.sqrt(d+1e-10)

                loss = (1-label[cls,row]) * tf.square(tf.maximum(0., margin - d_sqrt)) + (1*label[cls,row]) * d

                g_loss_cls = 0.5 * tf.reduce_mean(loss)
                g_loss += g_loss_cls
        g_loss = g_loss / (class_num*class_num)
        return g_loss



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
            weights['output_weights'] = tf.get_variable('weights', shape=[3, 3, 32, self.featurelen ], initializer=conv_initializer)

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

        self.logits = tf.nn.l2_normalize( self.logits, dim=-1)


        # self.pred_prob = tf.nn.softmax(self.logits) # shape [batch, w, h, num_classes]
        # self.pred_compact = tf.argmax(self.pred_prob, axis=-1) # shape [batch, w, h]

        return self.logits #self.conv10, self.logits, self.pred_prob, self.pred_compact


# class MASF:
#     def __init__(self,task='class'):
#         """ Call construct_model_*() after initializing MASF"""
#         # self.inner_lr = FLAGS.inner_lr
#         # self.outer_lr = FLAGS.outer_lr
#         # self.metric_lr = FLAGS.metric_lr
#         self.SKIP_LAYER = ['fc8']
#         if task=='class':
#             self.forward = self.forward_alexnet
#             # self.forward_metric_net = self.forward_metric_net
#             self.construct_weights = self.construct_alexnet_weights
#             self.featurelen = 512
#         elif task=='seg':
#             self.forward = self.forward_unet
#             # self.forward_metric_net = self.forward_metric_net
#             self.construct_weights = self.construct_unet_weights#self.construct_alexnet_weights
#             self.featurelen = 16
#         self.construct_cla_weights = self.construct_classifiar_weights
#         self.loss_func = xent
#         self.loss_func2 = xent
#         self.loss_func1 = get_center_loss
#         self.global_loss_func1 = g_loss
#         self.global_loss_func = cos_center_loss
#         self.global_loss_func2 = center_loss_global
#         self.global_loss_func3 = kd
#         self.sm_loss = g_loss
#         self.num_classes = 2
#         self.WEIGHTS_PATH = './pretrained_weights/bvlc_alexnet.npy'

#         self.alpha0 = FLAGS.a0 # 
#         self.alpha1 = FLAGS.a1 
#         self.alpha2 = FLAGS.a2
#         self.task = task

#     def construct_model_train(self, prefix='metatrain_'):
#         if self.task=='class':
#             # a: meta-train for inner update, b: meta-test for meta loss
#             self.inputa = tf.placeholder(tf.float32)
#             self.labela = tf.placeholder(tf.float32)

#             self.inputa1= tf.placeholder(tf.float32)
#             self.labela1= tf.placeholder(tf.float32)

#             self.inputb = tf.placeholder(tf.float32)
#             self.labelb = tf.placeholder(tf.float32)

#             meta_sample_num = (FLAGS.meta_batch_size) * 1
#             # self.input_group = tf.placeholder(tf.float32)
#             # self.label_group = tf.placeholder(tf.float32)
#             # self.label_groupm = tf.placeholder(tf.int32, shape=(meta_sample_num,))
#             # self.label_am = tf.placeholder(tf.int32, shape=(meta_sample_num,))
#             # self.label_a1m = tf.placeholder(tf.int32, shape=(meta_sample_num,))
#             # self.label_bm = tf.placeholder(tf.int32, shape=(meta_sample_num,))

#         elif self.task=='seg':
#             self.inputa = tf.placeholder(tf.float32,shape=[None,FLAGS.resize,FLAGS.resize,1])
#             self.labela = tf.placeholder(tf.float32,shape=[None,FLAGS.resize,FLAGS.resize,2])
#             self.maska = tf.placeholder(tf.float32,shape=[None,FLAGS.resize,FLAGS.resize])

#             self.inputa1 = tf.placeholder(tf.float32,shape=[None,FLAGS.resize,FLAGS.resize,1])
#             self.labela1 = tf.placeholder(tf.float32,shape=[None,FLAGS.resize,FLAGS.resize,2])
#             self.maska1 = tf.placeholder(tf.float32,shape=[None,FLAGS.resize,FLAGS.resize])

#             self.inputb = tf.placeholder(tf.float32,shape=[None,FLAGS.resize,FLAGS.resize,1])
#             self.labelb = tf.placeholder(tf.float32,shape=[None,FLAGS.resize,FLAGS.resize,2])
#             self.maskb = tf.placeholder(tf.float32,shape=[None,FLAGS.resize,FLAGS.resize])

#             meta_sample_num = (FLAGS.meta_batch_size) * 1
#             # self.input_group = tf.placeholder(tf.float32,shape=[None,FLAGS.resize,FLAGS.resize,1])
#             # self.label_group = tf.placeholder(tf.float32,shape=[None,FLAGS.resize,FLAGS.resize,2])
#             # self.label_groupm = tf.placeholder(tf.int32, shape=(None,FLAGS.resize,FLAGS.resize))
#             # self.label_am = tf.placeholder(tf.int32, shape=(None,FLAGS.resize,FLAGS.resize))
#             # self.label_a1m = tf.placeholder(tf.int32, shape=(None,FLAGS.resize,FLAGS.resize))
#             # self.label_bm = tf.placeholder(tf.int32, shape=(None,FLAGS.resize,FLAGS.resize))


#         # self.domaina = tf.placeholder(tf.int32)
#         # self.domaina1= tf.placeholder(tf.int32)
#         # self.domainb = tf.placeholder(tf.int32)
#         self.bool_indicator_b_a = tf.placeholder(tf.float32, shape=(self.num_classes,))
#         self.bool_indicator_b_a1 = tf.placeholder(tf.float32, shape=(self.num_classes,))

        
        

#         self.clip_value = FLAGS.gradients_clip_value
#         self.margin = FLAGS.margin
#         self.KEEP_PROB = tf.placeholder(tf.float32)
#         self.inner_lr = tf.placeholder(tf.float32,shape=())
#         self.outer_lr = tf.placeholder(tf.float32,shape=())

#         with tf.variable_scope('model', reuse=None) as training_scope:
#             if 'weights' in dir(self):
#                 print('weights already defined')
#                 training_scope.reuse_variables()
#                 weights = self.weights
#             else:
#                 self.weights = weights = self.construct_weights()

#             if 'cla_weights' in dir(self):
#                 print('weights already defined')
#                 training_scope.reuse_variables()
#                 cla_weights = self.cla_weights
#             else:
#                 self.cla_weights = cla_weights = self.construct_cla_weights()
#             # if 'source_moving_1_centroid' in dir(self):
#             #     print('centroids already defined')
#             #     training_scope.reuse_variables()
#             #     source_moving_1_centroid = self.source_moving_1_centroid
#             # else:
#             #     self.source_moving_1_centroid = source_moving_1_centroid = tf.zeros((self.num_classes, self.featurelen),
#             #                                                                         dtype=tf.float32) + 1e-10
#             #
#             # if 'source_moving_2_centroid' in dir(self):
#             #     print('centroids already defined')
#             #     training_scope.reuse_variables()
#             #     source_moving_2_centroid = self.source_moving_2_centroid
#             # else:
#             #     self.source_moving_2_centroid = source_moving_2_centroid = tf.zeros(
#             #         (self.num_classes, self.featurelen), dtype=tf.float32) + 1e-10
#             #
#             # if 'source_moving_3_centroid' in dir(self):
#             #     print('centroids already defined')
#             #     training_scope.reuse_variables()
#             #     source_moving_3_centroid = self.source_moving_3_centroid
#             # else:
#             #     self.source_moving_3_centroid = source_moving_3_centroid = tf.zeros((self.num_classes, self.featurelen),
#             #                                                                         dtype=tf.float32) + 1e-10

#             def task_metalearn(inp, global_bool_indicator, reuse=True):
#                 # Function to perform meta learning update """
#                 # inputa, inputa1, inputb, input_group, labela, labela1, labelb, label_group, labelam, labela1m, labelbm, label_groupm, domaina, domaina1, domainb = inp
#                 inputa, inputa1, inputb, labela, labela1, labelb = inp
#                 global_bool_indicator_b_a, global_bool_indicator_b_a1 = global_bool_indicator

#                 # Obtaining the conventional task loss on meta-train

#                 class_centroid = tf.nn.l2_normalize(cla_weights['conv9_weights'], dim=0)
#                 class_centroid = tf.transpose(class_centroid, [1, 0])
#                 task_embeddinga = self.forward(inputa, weights, reuse=reuse)
#                 task_outputa = self.forward_classifier(task_embeddinga, cla_weights)
#                 task_lossa = self.loss_func2(task_outputa, labela)
#                 # p_a = self.global_loss_func(task_embeddinga, labela, class_centroid)

#                 # center_loss_a = self.loss_func1(task_embeddinga, labela, class_centroid)

#                 # w_a = self.global_loss_func2(task_outputa, labela, class_centroid)

#                 task_embeddinga1 = self.forward(inputa1, weights, reuse=reuse)
#                 task_outputa1 = self.forward_classifier(task_embeddinga1, cla_weights)
#                 task_lossa1 = self.loss_func2(task_outputa1, labela1)

#                 #center_loss_a1 = self.loss_func1(task_embeddinga1, labela1, class_centroid)
                
#                 # p_a1 = self.global_loss_func( task_embeddinga1, labela1, class_centroid)
#                 # w_a1 = self.global_loss_func2(task_outputa1, labela1, class_centroid)

#                 task_embeddingb = self.forward(inputb, weights, reuse=reuse)
#                 task_outputb = self.forward_classifier(task_embeddingb, cla_weights)

#                 # perform inner update with plain gradient descent on meta-train
#                 fast_weights = weights


#                 ## compute global loss
#                 # center_loss_a = self.loss_func1(task_embeddinga, labela, current_source_a_centroid)
#                 # center_loss_a1 = self.loss_func1(task_embeddinga1, labela1, current_source_a1_centroid)

#                 # grads = tf.gradients( 1*(task_lossa +  task_lossa1) + 0.*(center_loss_a + center_loss_a1),
#                 #                      list(cla_weights.values()))

#                 grads = tf.gradients( 1*(task_lossa +  task_lossa1), list(cla_weights.values()))


#                 grads = [tf.stop_gradient(grad) for grad in grads] # first-order gradients approximation
#                 gradients = dict(zip(cla_weights.keys(), grads))

#                 fast_class = dict(zip(cla_weights.keys(), [cla_weights[key] - self.inner_lr * tf.clip_by_norm(gradients[key], clip_norm=self.clip_value) for key in cla_weights.keys()]))
#                 # fast_class = cla_weights

#                 class_centroid = tf.nn.l2_normalize(fast_class['conv9_weights'], dim=0)
#                 class_centroid = tf.transpose( class_centroid, [1, 0])
#                 if self.task == 'seg':
#                     assert self.featurelen==task_embeddinga.shape[-1]==task_embeddinga1.shape[-1]==task_embeddingb.shape[-1]
#                     assert self.num_classes == labela.shape[-1]==labela1.shape[-1]==labelb.shape[-1]
#                     task_embeddinga = tf.reshape(task_embeddinga,(-1,self.featurelen))
#                     task_embeddinga1 = tf.reshape(task_embeddinga1,(-1,self.featurelen))
#                     task_embeddingb = tf.reshape(task_embeddingb,(-1,self.featurelen))
#                     labela = tf.reshape(labela,(-1,self.num_classes))
#                     labela1 = tf.reshape(labela1,(-1,self.num_classes))
#                     labelb = tf.reshape(labelb,(-1,self.num_classes))

#                 center_loss_a1 = self.loss_func1(task_embeddinga1, labela1, class_centroid)
#                 center_loss_a = self.loss_func1(task_embeddinga, labela, class_centroid)
#                 center_loss_b = self.loss_func1(task_embeddingb, labelb, class_centroid)

#                 current_source_a_centroid = tf.zeros((self.num_classes, self.featurelen), dtype=tf.float32)
#                 current_source_a1_centroid = tf.zeros((self.num_classes, self.featurelen), dtype=tf.float32)
#                 current_source_b_centroid = tf.zeros((self.num_classes, self.featurelen), dtype=tf.float32)
#                 E = tf.eye(self.num_classes)
#                 eps = 1e-10
#                 domain_labela = tf.multiply(1.0, labela)
#                 domain_labelb = tf.multiply(1.0, labelb)
#                 domain_labela1 = tf.multiply(1.0, labela1)

#                 new_task_embeddinga = self.forward(inputa, fast_weights, reuse=reuse)
#                 new_task_embeddinga1 = self.forward(inputa1, fast_weights, reuse=reuse)
#                 new_task_embeddingb = self.forward(inputb, fast_weights, reuse=reuse)

#                 if self.task == 'seg':
#                     assert self.featurelen==new_task_embeddinga.shape[-1]==new_task_embeddinga1.shape[-1]==new_task_embeddingb.shape[-1]
#                     new_task_embeddinga = tf.reshape(new_task_embeddinga,(-1,self.featurelen))
#                     new_task_embeddinga1 = tf.reshape(new_task_embeddinga1,(-1,self.featurelen))
#                     new_task_embeddingb = tf.reshape(new_task_embeddingb,(-1,self.featurelen))

#                 #计算 域类原型
#                 for i in range(self.num_classes):
#                     current_source_a_centroid += tf.matmul(tf.expand_dims(E[:, i], 1), tf.expand_dims(
#                         tf.reduce_sum(tf.multiply(new_task_embeddinga, tf.expand_dims(domain_labela[:, i], 1)),
#                                       reduction_indices=0), 0)) / (tf.reduce_sum(domain_labela[:, i]) + eps)
#                     current_source_a1_centroid += tf.matmul(tf.expand_dims(E[:, i], 1), tf.expand_dims(
#                         tf.reduce_sum(tf.multiply(new_task_embeddinga1, tf.expand_dims(domain_labela1[:, i], 1)),
#                                       reduction_indices=0), 0)) / (tf.reduce_sum(domain_labela1[:, i]) + eps)
#                     current_source_b_centroid += tf.matmul(tf.expand_dims(E[:, i], 1), tf.expand_dims(
#                         tf.reduce_sum(tf.multiply(new_task_embeddingb, tf.expand_dims(domain_labelb[:, i], 1)),
#                                       reduction_indices=0), 0)) / (tf.reduce_sum(domain_labelb[:, i]) + eps)

#                 outb_a = tf.matmul(new_task_embeddingb, current_source_a_centroid, transpose_b=True) / 0.1
#                 outb_a1 = tf.matmul(new_task_embeddingb, current_source_a1_centroid, transpose_b=True) / 0.1
#                 outb_b = tf.matmul(new_task_embeddingb, class_centroid, transpose_b=True) / 0.1

#                 metric_loss = ( self.global_loss_func3( outb_a1, outb_b) + self.global_loss_func3( outb_a, outb_b)) / 2 #L_ca,/apha2

#                 if self.task=='seg':
#                     global_loss_a1 = self.global_loss_func1(task_embeddinga, class_centroid, labela,
#                                                             FLAGS.meta_batch_size*FLAGS.resize*FLAGS.resize,self.num_classes, self.featurelen, margin=1)
#                     global_loss_a = self.global_loss_func1(task_embeddinga1, class_centroid, labela1,
#                                                             FLAGS.meta_batch_size*FLAGS.resize*FLAGS.resize, self.num_classes, self.featurelen, margin=1)
#                     global_loss_b = self.global_loss_func1(task_embeddingb, class_centroid, labelb,
#                                                             FLAGS.meta_batch_size*FLAGS.resize*FLAGS.resize, self.num_classes, self.featurelen, margin=1)
#                     global_loss = ( global_loss_b + global_loss_a1 + global_loss_a) / 3 #L_fa, /apha1
                    
#                     task_outputb = self.forward_classifier(task_embeddingb, fast_class)
#                     task_lossb = self.loss_func2(task_outputb, labelb)
#                     sm_loss = tf.reduce_mean( task_lossb)

#                     task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), -1), tf.reshape(tf.argmax(labela, -1),(-1,FLAGS.resize,FLAGS.resize))) #this accuracy already gathers batch size
#                     task_accuracya1 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa1), -1), tf.reshape(tf.argmax(labela1, -1),(-1,FLAGS.resize,FLAGS.resize)))
#                     task_accuracyb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputb), -1), tf.argmax(labelb, -1))

#                 else:
#                     global_loss_a1 = self.global_loss_func1(task_embeddinga, class_centroid, labela,
#                                                             FLAGS.meta_batch_size,self.num_classes, self.featurelen, margin=1)
#                     global_loss_a = self.global_loss_func1(task_embeddinga1, class_centroid, labela1,
#                                                             FLAGS.meta_batch_size, self.num_classes, self.featurelen, margin=1)
#                     global_loss_b = self.global_loss_func1(task_embeddingb, class_centroid, labelb,
#                                                             FLAGS.meta_batch_size, self.num_classes, self.featurelen, margin=1)
#                     global_loss = ( global_loss_b + global_loss_a1 + global_loss_a) / 3

#                     task_outputb = self.forward_classifier(task_embeddingb, fast_class)
#                     task_lossb = self.loss_func2(task_outputb, labelb)
#                     sm_loss = tf.reduce_mean( task_lossb)

#                     # task_output = [ global_loss,  metric_loss, sm_loss, task_lossa+0.*center_loss_a , task_lossa1+0.*center_loss_a1, task_lossb]
#                     task_accuracya = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa), -1), tf.argmax(labela, -1)) #this accuracy already gathers batch size
#                     task_accuracya1 = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputa1), -1), tf.argmax(labela1, -1))
#                     task_accuracyb = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(task_outputb), -1), tf.argmax(labelb, -1))

#                 task_output = [ global_loss,  metric_loss, sm_loss, task_lossa+0.*center_loss_a , task_lossa1+0.*center_loss_a1, task_lossb]
#                 task_output.extend([task_accuracya, task_accuracya1, task_accuracyb])

#                 return task_output

#             self.global_step = tf.Variable(0, trainable=False)

#             # input_tensors = (self.inputa, self.inputa1, self.inputb, self.input_group, self.labela, self.labela1, self.labelb, self.label_group,self.label_am,self.label_a1m,self.label_bm, self.label_groupm, self.domaina, self.domaina1, self.domainb)
#             input_tensors = (self.inputa, self.inputa1, self.inputb, self.labela, self.labela1, self.labelb)
#             global_bool_indicator = (self.bool_indicator_b_a, self.bool_indicator_b_a1)

#             result = task_metalearn(inp=input_tensors, global_bool_indicator=global_bool_indicator)
#             global_loss,  metric_loss, sm_loss, self.lossa_raw, self.lossa1_raw, self.lossb_raw, accuracya, accuracya1, accuracyb = result
#             self.sm_loss = sm_loss * self.alpha0
#             self.global_loss = global_loss * self.alpha1
#             self.metric_loss = metric_loss * self.alpha2
#             # self.source_losses = source_loss * 1



#         ## Performance & Optimization
#         if 'train' in prefix:

#             self.lossa = avg_lossa = tf.reduce_mean(self.lossa_raw)
#             self.lossa1 = avg_lossa1 = tf.reduce_mean(self.lossa1_raw)
#             self.lossb = avg_lossb = tf.reduce_mean(self.lossb_raw)
#             self.source_loss = (avg_lossa + avg_lossa1) / 2.0
#             # self.outer_lr = tf.train.exponential_decay(learning_rate=FLAGS.outer_lr, global_step=self.global_step,
#             #                                            decay_steps=10000, decay_rate=0.96)

#             self.task_train_op = tf.train.AdamOptimizer(learning_rate=self.outer_lr).minimize( self.source_loss, global_step=self.global_step)


#             self.accuracya = accuracya * 100.
#             self.accuracya1 = accuracya1 * 100.
#             self.accuracyb = accuracyb * 100.
#             self.source_accuracy = (self.accuracya + self.accuracya1 + self.accuracyb) / 3.0

#             var_list_feature_extractor = [v for v in tf.trainable_variables() if (v.name.split('/')[1] not in ['fc9'])]
#             var_list_classifier = [v for v in tf.trainable_variables() if (v.name.split('/')[1] in ['fc9'])]
#             # var_list_classifier = [v for v in tf.trainable_variables() if 'classifier' in v.name.split('/')]
#             # self.metric_lr = FLAGS.metric_lr


#             optimizer = tf.train.AdamOptimizer(self.outer_lr)
#             gvs = optimizer.compute_gradients( self.global_loss + self.metric_loss + self.sm_loss ,
#                                               var_list = var_list_feature_extractor + var_list_classifier)  # observe stability of gradients for meta loss
#             l2_norm = lambda t: tf.sqrt(tf.reduce_sum(tf.pow(t, 2)))
#             for grad, var in gvs:
#                 tf.summary.histogram("gradients_norm/" + var.name, l2_norm(grad))
#                 tf.summary.histogram("feature_extractor_var_norm/" + var.name, l2_norm(var))
#                 tf.summary.histogram('gradients/' + var.name, var)
#                 tf.summary.histogram("feature_extractor_var/" + var.name, var)

#             gvs = [(tf.clip_by_norm(grad, clip_norm=self.clip_value), var) for grad, var in gvs]

#             for grad, var in gvs:
#                 tf.summary.histogram("gradients_norm_clipped/" + var.name, l2_norm(grad))
#                 tf.summary.histogram('gradients_clipped/' + var.name, var)

#             self.meta_train_op = optimizer.apply_gradients(gvs)


#         ## Summaries
#         tf.summary.scalar(prefix+'source loss', self.source_loss)
#         # tf.summary.scalar(prefix+'source_2 loss', self.lossa1)
#         tf.summary.scalar(prefix+'source_1 accuracy', self.accuracya)
#         tf.summary.scalar(prefix+'source_2 accuracy', self.accuracya1)
#         tf.summary.scalar(prefix+'global loss', self.global_loss)
#         tf.summary.scalar(prefix+'metric loss', self.metric_loss)
#         tf.summary.scalar(prefix+'sm loss', self.sm_loss)

#     def construct_model_test(self, prefix='test'):
#         if self.task == 'class':
#             self.test_input = tf.placeholder(tf.float32)
#             self.test_label = tf.placeholder(tf.float32)
#         elif self.task == 'seg':
#             self.test_input = tf.placeholder(tf.float32,[None,FLAGS.resize,FLAGS.resize,1])
#             self.test_label = tf.placeholder(tf.float32,[None,FLAGS.resize,FLAGS.resize,2])
#             self.test_mask = tf.placeholder(tf.float32,[None,FLAGS.resize,FLAGS.resize])
#         else:
#             raise RuntimeError('check task type, class or seg.')

#         with tf.variable_scope('model', reuse=None) as testing_scope:
#             if 'weights' in dir(self):
#                 testing_scope.reuse_variables()
#                 weights = self.weights
#             else:
#                 raise ValueError('Weights not initilized. Create training model before testing model')

#             if 'cla_weights' in dir(self):
#                 testing_scope.reuse_variables()
#                 cla_weights = self.cla_weights
#             else:
#                 raise ValueError('Weights not initilized. Create training model before testing model')

#             # self.semantic_feature = self.forward(self.test_input, weights)
#             # outputs = self.forward_classifier(self.semantic_feature, cla_weights)
#             # losses = self.loss_func(outputs, self.test_label)
#             # accuracies = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputs), 1), tf.argmax(self.test_label, 1))
#             # self.pred_prob = tf.nn.softmax(outputs)
#             # self.outputs = outputs
#             feature = self.forward(self.test_input, weights)
#             outputs = self.forward_classifier(feature,cla_weights)
#             losses = self.loss_func(outputs, self.test_label)
#             accuracies = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(outputs), -1), tf.argmax(self.test_label, -1))
#             self.pred_prob = tf.nn.softmax(outputs)
#             self.outputs = outputs
#             if self.task == 'seg':
#                 self.dice_slice = dice_coefficient(y_true=tf.argmax(self.test_label, -1),y_pred=tf.argmax(tf.nn.softmax(outputs), -1))
#                 # self.masked_dice_slice = masked_dice_coefficient(y_true=tf.argmax(self.test_label, -1),y_pred=tf.argmax(tf.nn.softmax(outputs), -1),mask=self.test_mask)
#                 self.y_true = tf.argmax(self.test_label, -1)
#                 self.y_prob = tf.nn.softmax(outputs,-1)[:,:,:,1]
#                 self.y_pred = tf.argmax(tf.nn.softmax(outputs), -1)
#         self.test_loss = tf.reduce_mean(losses)
#         self.test_acc = accuracies


#     def load_initial_weights(self, session):
#         """Load weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/
#         The weights come as a dict of lists (e.g. weights['conv1'] is a list)
#         Load the weights into the model
#         """
#         weights_dict = np.load(self.WEIGHTS_PATH, allow_pickle= True, encoding='bytes').item()

#         # Loop over all layer names stored in the weights dict
#         for op_name in weights_dict:

#             # Check if layer should be trained from scratch
#             if op_name not in self.SKIP_LAYER:

#                 with tf.variable_scope('model', reuse=True):
#                     with tf.variable_scope(op_name, reuse=True):

#                         for data in weights_dict[op_name]:
#                             # Biases
#                             if len(data.shape) == 1:
#                                 var = tf.get_variable('biases', trainable=True)
#                                 session.run(var.assign(data))
#                             # Weights
#                             else:
#                                 var = tf.get_variable('weights', trainable=True)
#                                 session.run(var.assign(data))

#     def construct_alexnet_weights(self):

#         weights = {}
#         conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
#         fc_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)

#         with tf.variable_scope('conv1') as scope:
#             weights['conv1_weights'] = tf.get_variable('weights', shape=[11, 11, 3, 96], initializer=conv_initializer)
#             weights['conv1_biases'] = tf.get_variable('biases', [96])

#         with tf.variable_scope('conv2') as scope:
#             weights['conv2_weights'] = tf.get_variable('weights', shape=[5, 5, 48, 256], initializer=conv_initializer)
#             weights['conv2_biases'] = tf.get_variable('biases', [256])

#         with tf.variable_scope('conv3') as scope:
#             weights['conv3_weights'] = tf.get_variable('weights', shape=[3, 3, 256, 384], initializer=conv_initializer)
#             weights['conv3_biases'] = tf.get_variable('biases', [384])

#         with tf.variable_scope('conv4') as scope:
#             weights['conv4_weights'] = tf.get_variable('weights', shape=[3, 3, 192, 384], initializer=conv_initializer)
#             weights['conv4_biases'] = tf.get_variable('biases', [384])

#         with tf.variable_scope('conv5') as scope:
#             weights['conv5_weights'] = tf.get_variable('weights', shape=[3, 3, 192, 256], initializer=conv_initializer)
#             weights['conv5_biases'] = tf.get_variable('biases', [256])

#         with tf.variable_scope('fc6') as scope:
#             weights['fc6_weights'] = tf.get_variable('weights', shape=[9216, 4096], initializer=conv_initializer)
#             weights['fc6_biases'] = tf.get_variable('biases', [4096])

#         with tf.variable_scope('fc7') as scope:
#             weights['fc7_weights'] = tf.get_variable('weights', shape=[4096, 4096], initializer=conv_initializer)
#             weights['fc7_biases'] = tf.get_variable('biases', [4096])

#         with tf.variable_scope('fc8') as scope:
#             weights['fc8_weights'] = tf.get_variable('weights', shape=[4096, self.featurelen], initializer=conv_initializer)
#             weights['fc8_biases'] = tf.get_variable('biases', [self.featurelen])

#         return weights

#     def construct_classifiar_weights(self):

#         weights = {}
#         conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
#         # fc_initializer = tf.contrib.layers.xavier_initializer(dtype=tf.float32)

#         with tf.variable_scope('fc9') as scope:
#             weights['conv9_weights'] = tf.get_variable('weights', shape=[ self.featurelen, self.num_classes], initializer=conv_initializer)
#         return weights
    
#     def forward_alexnet(self, inp, weights, reuse=False):
#         # reuse is for the normalization parameters.

#         conv1 = conv_block(inp, weights['conv1_weights'], weights['conv1_biases'], stride_y=4, stride_x=4, groups=1,
#                            reuse=reuse, scope='conv1')
#         norm1 = lrn(conv1, 2, 1e-05, 0.75)
#         pool1 = max_pool(norm1, 3, 3, 2, 2, padding='VALID')

#         # 2nd Layer: Conv (w ReLu)  -> Lrn -> Pool with 2 groups
#         conv2 = conv_block(pool1, weights['conv2_weights'], weights['conv2_biases'], stride_y=1, stride_x=1, groups=2,
#                            reuse=reuse, scope='conv2')
#         norm2 = lrn(conv2, 2, 1e-05, 0.75)
#         pool2 = max_pool(norm2, 3, 3, 2, 2, padding='VALID')

#         # 3rd Layer: Conv (w ReLu)
#         conv3 = conv_block(pool2, weights['conv3_weights'], weights['conv3_biases'], stride_y=1, stride_x=1, groups=1,
#                            reuse=reuse, scope='conv3')

#         # 4th Layer: Conv (w ReLu) splitted into two groups
#         conv4 = conv_block(conv3, weights['conv4_weights'], weights['conv4_biases'], stride_y=1, stride_x=1, groups=2,
#                            reuse=reuse, scope='conv4')

#         # 5th Layer: Conv (w ReLu) -> Pool splitted into two groups
#         conv5 = conv_block(conv4, weights['conv5_weights'], weights['conv5_biases'], stride_y=1, stride_x=1, groups=2,
#                            reuse=reuse, scope='conv5')
#         pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID')

#         # 6th Layer: Flatten -> FC (w ReLu) -> Dropout
#         flattened = tf.reshape(pool5, [-1, 6 * 6 * 256])
#         fc6 = fc(flattened, weights['fc6_weights'], weights['fc6_biases'], activation='relu')
#         dropout6 = dropout(fc6, self.KEEP_PROB)

#         # 7th Layer: FC (w ReLu) -> Dropout
#         fc7 = fc(dropout6, weights['fc7_weights'], weights['fc7_biases'], activation='relu')
#         dropout7 = dropout(fc7, self.KEEP_PROB)

#         fc8 = fc(dropout7, weights['fc8_weights'], weights['fc8_biases'], activation='relu')

#         fc8 = tf.nn.l2_normalize(fc8, dim=1)

#         return fc8

#     def forward_classifier(self, inp, weights, T=0.05, reuse=False):

#         # inp = tf.nn.l2_normalize(inp, dim=1)
#         # inp = dropout(inp, self.KEEP_PROB)
#         w = tf.nn.l2_normalize( weights['conv9_weights'], dim=0)
#         fc8 = linear(inp,  w)
#         fc8 = fc8 / T

#         return fc8

#     def global_loss(self,source_feature, target_feature, class_num, margin):
#         label = tf.to_float(tf.eye(7))
#         g_loss = 0
#         margin = tf.to_float(tf.constant(margin))
#         for cls in range(class_num):
#             for row in range(class_num):
#             # feature = tf.tile(target_feature[cls, :], [7, 1])
#                 d = tf.square(tf.subtract(source_feature[cls,:], target_feature[row,:]))
#                 d_sqrt = tf.sqrt(d+1e-10)

#                 loss = (1-label[cls,row]) * tf.square(tf.maximum(0., margin - d_sqrt)) + (1*label[cls,row]) * d

#                 g_loss_cls = 0.5 * tf.reduce_mean(loss)
#                 g_loss += g_loss_cls
#         g_loss = g_loss / (class_num*class_num)
#         return g_loss



#     def construct_unet_weights(self):

#         weights = {}
#         conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)

#         with tf.variable_scope('conv1') as scope:
#             weights['conv1_weights'] = tf.get_variable('weights', shape=[5, 5, 1, 16], initializer=conv_initializer)
#             weights['conv1_biases'] = tf.get_variable('biases', [16])

#         with tf.variable_scope('conv2') as scope:
#             weights['conv2_weights'] = tf.get_variable('weights', shape=[5, 5, 16, 32], initializer=conv_initializer)
#             weights['conv2_biases'] = tf.get_variable('biases', [32])

#         ## Network has downsample here

#         with tf.variable_scope('conv3') as scope:
#             weights['conv3_weights'] = tf.get_variable('weights', shape=[3, 3, 32, 64], initializer=conv_initializer)
#             weights['conv3_biases'] = tf.get_variable('biases', [64])

#         with tf.variable_scope('conv4') as scope:
#             weights['conv4_weights'] = tf.get_variable('weights', shape=[3, 3, 64, 64], initializer=conv_initializer)
#             weights['conv4_biases'] = tf.get_variable('biases', [64])

#         ## Network has downsample here

#         with tf.variable_scope('conv5') as scope:
#             weights['conv5_weights'] = tf.get_variable('weights', shape=[3, 3, 64, 128], initializer=conv_initializer)
#             weights['conv5_biases'] = tf.get_variable('biases', [128])

#         # with tf.variable_scope('conv6') as scope:
#         #     weights['conv6_weights'] = tf.get_variable('weights', shape=[3, 3, 128, 128], initializer=conv_initializer)
#         #     weights['conv6_biases'] = tf.get_variable('biases', [128])

#         with tf.variable_scope('deconv1') as scope:
#             weights['deconv1_weights'] = tf.get_variable('weights', shape=[3, 3, 64, 128], initializer=conv_initializer)

#         with tf.variable_scope('conv7') as scope:
#             weights['conv7_weights'] = tf.get_variable('weights', shape=[3, 3, 64, 64], initializer=conv_initializer)
#             weights['conv7_biases'] = tf.get_variable('biases', [64])

#         with tf.variable_scope('conv8') as scope:
#             weights['conv8_weights'] = tf.get_variable('weights', shape=[3, 3, 64, 64], initializer=conv_initializer)
#             weights['conv8_biases'] = tf.get_variable('biases', [64])

#         with tf.variable_scope('deconv2') as scope:
#             weights['deconv2_weights'] = tf.get_variable('weights', shape=[3, 3, 32, 64], initializer=conv_initializer)

#         # with tf.variable_scope('conv9') as scope:
#         #     weights['conv9_weights'] = tf.get_variable('weights', shape=[3, 3, 32, 32], initializer=conv_initializer)
#         #     weights['conv9_biases'] = tf.get_variable('biases', [32])

#         with tf.variable_scope('conv10') as scope:
#             weights['conv10_weights'] = tf.get_variable('weights', shape=[3, 3, 32, 32], initializer=conv_initializer)
#             weights['conv10_biases'] = tf.get_variable('biases', [32])

#         with tf.variable_scope('output') as scope:
#             weights['output_weights'] = tf.get_variable('weights', shape=[3, 3, 32, self.featurelen ], initializer=conv_initializer)

#         return weights
    
#     def forward_unet(self, inp, weights,reuse=False):

#         self.conv1 = conv_block1(inp, weights['conv1_weights'], weights['conv1_biases'])
#         self.conv2 = conv_block1(self.conv1, weights['conv2_weights'], weights['conv2_biases'])
#         self.pool2 = max_pool(self.conv2, 2, 2, 2, 2, padding='VALID')

#         self.conv3 = conv_block1(self.pool2, weights['conv3_weights'], weights['conv3_biases'])
#         self.conv4 = conv_block1(self.conv3, weights['conv4_weights'], weights['conv4_biases'])
#         self.pool4 = max_pool(self.conv4, 2, 2, 2, 2, padding='VALID')

#         self.conv5 = conv_block1(self.pool4, weights['conv5_weights'], weights['conv5_biases'])
#         # self.conv6 = conv_block(self.conv5, weights['conv6_weights'], weights['conv6_biases'])

#         ## add upsampling, meanwhile, channel number is reduced to half
#         # self.up1 = deconv_block(self.conv6, weights['deconv1_weights'])
#         self.up1 = deconv_block(self.conv5, weights['deconv1_weights'])

#         self.conv7 = conv_block1(self.up1, weights['conv7_weights'], weights['conv7_biases'])
#         self.conv8 = conv_block1(self.conv7, weights['conv8_weights'], weights['conv8_biases'])

#         ## add upsampling, meanwhile, channel number is reduced to half
#         self.up2 = deconv_block(self.conv8, weights['deconv2_weights'])

#         # self.conv9 = conv_block(self.up2, weights['conv9_weights'], weights['conv9_biases'])
#         # self.conv10 = conv_block(self.conv9, weights['conv10_weights'], weights['conv10_biases'])
#         self.conv10 = conv_block1(self.up2, weights['conv10_weights'], weights['conv10_biases'])

#         self.logits = tf.nn.conv2d(self.conv10, weights['output_weights'], strides=[1, 1, 1, 1], padding='SAME')

#         self.logits = tf.nn.l2_normalize( self.logits, dim=-1)


#         # self.pred_prob = tf.nn.softmax(self.logits) # shape [batch, w, h, num_classes]
#         # self.pred_compact = tf.argmax(self.pred_prob, axis=-1) # shape [batch, w, h]

#         return self.logits #self.conv10, self.logits, self.pred_prob, self.pred_compact



# # Network blocks
def conv_block1(inp, cweight, bweight):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    conv = tf.nn.conv2d(inp, cweight, strides=[1, 1, 1, 1], padding='SAME')
    conv = tf.nn.bias_add(conv, bweight)
    relu = tf.nn.leaky_relu(conv)
    return relu


def deconv_block(inp, cweight):
    # x_shape = tf.shape(inp)
    inp_shape = tf.shape(inp)
    x_shape = inp.get_shape().as_list()
    output_shape = tf.stack([inp_shape[0], x_shape[1]*2, x_shape[2]*2, x_shape[3]//2])
    deconv = tf.nn.conv2d_transpose(inp, cweight, output_shape, strides=[1,2,2,1], padding='SAME')
    return deconv

