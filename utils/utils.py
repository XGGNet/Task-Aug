""" Utility functions. """ 
import numpy as np
import os
import random
import tensorflow as tf

from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags
# import msda_tensor
import itertools
# from tensorbayes.tfutils import softmax_cross_entropy_with_two_logits as softmax_xent_two

FLAGS = flags.FLAGS

## Image reader
def get_images(paths, labels, nb_samples=None, shuffle=True):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, os.path.join(path, image)) \
        for i, path in zip(labels, paths) \
        for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images)
    return images

## Network blocks
def conv_block(inp, cweight, bweight, stride_y=2, stride_x=2, groups=1, reuse=False, scope=''):
    stride = [1, stride_y, stride_x, 1]
    convolve = lambda i, k: tf.nn.conv2d(i, k, strides=stride, padding='SAME')

    if groups==1:
        conv_output = tf.nn.bias_add(convolve(inp, cweight), bweight)
    else:
        input_groups = tf.split(axis=3, num_or_size_splits=groups, value=inp)
        weight_groups = tf.split(axis=3, num_or_size_splits=groups, value=cweight)
        output_groups = [convolve(i, k) for i, k in zip(input_groups, weight_groups)]

        conv = tf.concat(axis=3, values=output_groups)
        conv_output = tf.nn.bias_add(conv, bweight)

    relu = tf.nn.relu(conv_output)

    return relu

def normalize(inp, activation, reuse, scope):
    return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)

def max_pool(x, filter_height, filter_width, stride_y, stride_x, padding='SAME'):
    """Create a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, filter_height, filter_width, 1], strides=[1, stride_y, stride_x, 1], padding=padding)

def lrn(x, radius, alpha, beta, bias=1.0):
    """Create a local response normalization layer."""
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha, beta=beta, bias=bias)

def dropout(x, keep_prob):
    """Create a dropout layer."""
    return tf.nn.dropout(x, keep_prob)

def fc(x, wweight, bweight, activation=None):
    """Create a fully connected layer."""
    
    act = tf.nn.xw_plus_b(x, wweight, bweight)

    if activation is 'relu':
        return tf.nn.relu(act)
    elif activation is 'leaky_relu':
        return tf.nn.leaky_relu(act)
    elif activation is None:
        return act
    else:
        raise NotImplementedError

## Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))

def xent2(pred, label):
    pred_n = tf.reduce_max(tf.multiply(tf.nn.softmax(pred),(tf.ones((120,7))-label)), axis= 1)
    return tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=label) + pred_n

def xent(pred, label):
    # pred_n = tf.reduce_max(tf.multiply(tf.nn.softmax(pred),(tf.ones((120,7))-label)), axis= 1)
    return tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=label)

def kd(data1, label1, data2, label2, bool_indicator, n_class=7, temperature=2.0):

    kd_loss = 0.0
    eps = 1e-16
    centroid_a = tf.zeros((7,4096))

    prob1s = []
    prob2s = []

    for cls in range(n_class):

        mask1 = tf.tile(tf.expand_dims(label1[:, cls], -1), [1, n_class])
        logits_sum1 = tf.reduce_sum(tf.multiply(data1, mask1), axis=0)
        num1 = tf.reduce_sum(label1[:, cls])
        activations1 = logits_sum1 * 1.0 / (num1 + eps) # add eps for prevent un-sampled class resulting in NAN
        prob1 = tf.nn.softmax(activations1 / temperature)
        prob1 = tf.clip_by_value(prob1, clip_value_min=1e-8, clip_value_max=1.0)  # for preventing prob=0 resulting in NAN

        mask2 = tf.tile(tf.expand_dims(label2[:, cls], -1), [1, n_class])
        logits_sum2 = tf.reduce_sum(tf.multiply(data2, mask2), axis=0)
        num2 = tf.reduce_sum(label2[:, cls])
        activations2 = logits_sum2 * 1.0 / (num2 + eps)
        prob2 = tf.nn.softmax(activations2 / temperature)
        prob2 = tf.clip_by_value(prob2, clip_value_min=1e-8, clip_value_max=1.0)

        KL_div = (tf.reduce_sum(prob1 * tf.log(prob1 / prob2)) + tf.reduce_sum(prob2 * tf.log(prob2 / prob1))) / 2.0
        kd_loss += KL_div * bool_indicator[cls]

        prob1s.append(activations1)
        prob2s.append(activations2)

    kd_loss = kd_loss / n_class

    return kd_loss, prob1s, prob2s

def kd_centroid(data1, label1, centroid1, data2, label2, centroid2, bool_indicator, n_class=7, temperature=2.0):

    kd_loss = 0.0
    eps = 1e-16

    prob1s = []
    prob2s = []
    decay = 0.3
    for cls in range(n_class):
        mask1 = tf.tile(tf.expand_dims(label1[:, cls], -1), [1, 4096])
        logits_sum1 = tf.reduce_sum(tf.multiply(data1, mask1), axis=0)
        num1 = tf.reduce_sum(label1[:, cls])
        activations1 = logits_sum1 * 1.0 / (num1 + eps) # add eps for prevent un-sampled class resulting in NAN
        # prob1 = tf.nn.softmax(activations1 / temperature)
        # prob1 = tf.clip_by_value(prob1, clip_value_min=1e-8, clip_value_max=1.0)  # for preventing prob=0 resulting in NAN
        centroid1[cls,:] = (decay) * activations1 + (1. - decay) * centroid1[cls,:]
        mask2 = tf.tile(tf.expand_dims(label2[:, cls], -1), [1, 4096])
        logits_sum2 = tf.reduce_sum(tf.multiply(data2, mask2), axis=0)
        num2 = tf.reduce_sum(label2[:, cls])
        activations2 = logits_sum2 * 1.0 / (num2 + eps)
        # prob2 = tf.nn.softmax(activations2 / temperature)
        # prob2 = tf.clip_by_value(prob2, clip_value_min=1e-8, clip_value_max=1.0)
        centroid2[cls, :] = (decay) * activations2 + (1. - decay) * centroid2[cls, :]
        prob1 = centroid1[cls]
        prob2 = centroid2[cls]
        KL_div = (tf.reduce_sum(prob1 * tf.log(prob1 / prob2)) + tf.reduce_sum(prob2 * tf.log(prob2 / prob1))) / 2.0
        kd_loss += KL_div * bool_indicator[cls]

        prob1s.append(prob1)
        prob2s.append(prob2)

    kd_loss = kd_loss / n_class

    return kd_loss, centroid1, centroid2
def JS(data1, label1, data2, label2, bool_indicator, n_class=7, temperature=2.0):

    kd_loss = 0.0
    eps = 1e-16

    prob1s = []
    prob2s = []

    for cls in range(n_class):
        mask1 = tf.tile(tf.expand_dims(label1[:, cls], -1), [1, n_class])
        logits_sum1 = tf.reduce_sum(tf.multiply(data1, mask1), axis=0)
        num1 = tf.reduce_sum(label1[:, cls])
        activations1 = logits_sum1 * 1.0 / (num1 + eps) # add eps for prevent un-sampled class resulting in NAN
        prob1 = tf.nn.softmax(activations1 / temperature)
        prob1 = tf.clip_by_value(prob1, clip_value_min=1e-8, clip_value_max=1.0)  # for preventing prob=0 resulting in NAN

        mask2 = tf.tile(tf.expand_dims(label2[:, cls], -1), [1, n_class])
        logits_sum2 = tf.reduce_sum(tf.multiply(data2, mask2), axis=0)
        num2 = tf.reduce_sum(label2[:, cls])
        activations2 = logits_sum2 * 1.0 / (num2 + eps)
        prob2 = tf.nn.softmax(activations2 / temperature)
        prob2 = tf.clip_by_value(prob2, clip_value_min=1e-8, clip_value_max=1.0)

        mean_prob = (prob1 + prob2) / 2

        JS_div = (tf.reduce_sum(prob1 * tf.log(prob1 / mean_prob)) + tf.reduce_sum(prob2 * tf.log(prob2 / mean_prob))) / 2.0
        kd_loss += JS_div * bool_indicator[cls]

        prob1s.append(prob1)
        prob2s.append(prob2)

    kd_loss = kd_loss / n_class

    return kd_loss, prob1s, prob2s

def contrastive(feature1, label1, feature2, label2, bool_indicator=None, margin=50):

    l1 = tf.argmax(label1, axis=1)
    l2 = tf.argmax(label2, axis=1)
    pair = tf.to_float(tf.equal(l1,l2))

    delta = tf.reduce_sum(tf.square(feature1-feature2), 1) + 1e-10
    match_loss = delta

    delta_sqrt = tf.sqrt(delta + 1e-10)
    mismatch_loss = tf.square(tf.nn.relu(margin - delta_sqrt))

    if bool_indicator is None:
        loss = tf.reduce_mean(0.5 * (pair * match_loss + (1-pair) * mismatch_loss))
    else:
        loss = 0.5 * tf.reduce_sum(match_loss*pair)/tf.reduce_sum(pair)

    debug_dist_positive = tf.reduce_sum(delta_sqrt * pair)/tf.reduce_sum(pair)
    debug_dist_negative = tf.reduce_sum(delta_sqrt * (1-pair))/tf.reduce_sum(1-pair)

    return loss, pair, delta, debug_dist_positive, debug_dist_negative

def compute_distance(feature1, label1, feature2, label2):
    l1 = tf.argmax(label1, axis=1)
    l2 = tf.argmax(label2, axis=1)
    pair = tf.to_float(tf.equal(l1,l2))

    delta = tf.reduce_sum(tf.square(feature1-feature2), 1)
    delta_sqrt = tf.sqrt(delta + 1e-16)

    dist_positive_pair = tf.reduce_sum(delta_sqrt * pair)/tf.reduce_sum(pair)
    dist_negative_pair = tf.reduce_sum(delta_sqrt * (1-pair))/tf.reduce_sum(1-pair)

    return dist_positive_pair, dist_negative_pair

def kd(prob1,prob2):
    eps = 1e-16

    prob1 = tf.nn.softmax(prob1)
    prob2 = tf.nn.softmax(prob2)

    prob1 = tf.clip_by_value(prob1, clip_value_min=eps, clip_value_max=1.0)  # for preventing prob=0 resulting in NAN


    prob2 = tf.clip_by_value(prob2, clip_value_min=eps, clip_value_max=1.0)

    KL_div = (tf.reduce_sum(prob1 * tf.log(prob1 / prob2)) + tf.reduce_sum(prob2 * tf.log(prob2 / prob1))) / 2.0
    return KL_div

def masked_kd(prob1,prob2,mask):
    eps = 1e-16

    prob1 = tf.nn.softmax(prob1)
    prob2 = tf.nn.softmax(prob2)

    prob1 = tf.clip_by_value(prob1, clip_value_min=eps, clip_value_max=1.0)  # for preventing prob=0 resulting in NAN

    prob2 = tf.clip_by_value(prob2, clip_value_min=eps, clip_value_max=1.0)

    mask_2 = tf.stack([mask,mask],-1)

    KL_div = (tf.reduce_sum(prob1 * tf.log(prob1 / prob2)*mask_2) + tf.reduce_sum(prob2 * tf.log(prob2 / prob1)*mask_2)) / 2.0 
    return KL_div

def kd_class(activations1,activations2, n_class=7, margin=2.0):

    p_kd_loss = KL_div = 0.0
    eps = 1e-16

    prob1s = []
    prob2s = []

    for cls in range(n_class):
        prob1 = activations1[cls, :]
        prob1 = tf.nn.l2_normalize(prob1)
        prob1 = tf.clip_by_value(prob1, clip_value_min=eps,
                                 clip_value_max=1.0)  # for preventing prob=0 resulting in NAN
        # for row in range(n_class):


        prob2 = activations2[cls,:]
        # prob2 = tf.nn.softmax(prob2 / margin)
        prob2 = tf.nn.l2_normalize(prob2)
        prob2 = tf.clip_by_value(prob2, clip_value_min=eps, clip_value_max=1.0)

        KL_div = (tf.reduce_sum(prob1 * tf.log(prob1 / prob2)) + tf.reduce_sum(prob2 * tf.log(prob2 / prob1))) / 2.0
        prob1s.append(prob1)
        prob2s.append(prob2)

    kd_loss = KL_div / n_class

    return kd_loss



from tensorflow.python.framework import ops

from tensorflow.python.ops import math_ops
def linear(x, weights):
    """Create a fully connected layer."""

    weights = ops.convert_to_tensor(weights, name="weights")

    return math_ops.matmul(x, weights)

def euclidean_distance(a, b):
    # a.shape = N x D
    # b.shape = M x D
    N, D = tf.shape(a)[0], tf.shape(a)[1]
    M = tf.shape(b)[0]
    a = tf.tile(tf.expand_dims(a, axis=1), (1, M, 1))
    b = tf.tile(tf.expand_dims(b, axis=0), (N, 1, 1))
    return tf.reduce_mean(tf.square(a - b), axis=2)

def sample_loss(feature,centroid,y_one_hot, num_classes, num_queries):
    dists = euclidean_distance(feature, centroid)
    # ce_loss = xent(-dists,y_one_hot)
    # print (tf.shape(dists))
    log_p_y = tf.reshape(tf.nn.log_softmax(-dists), [num_classes, num_queries, -1])
    ce_loss = -tf.reduce_mean(tf.reshape(tf.reduce_sum(tf.multiply(y_one_hot, log_p_y), axis=-1), [-1]))
    return ce_loss, log_p_y

# def compute_euclidean_distance(x, y):
#     """
#     Computes the euclidean distance between two tensorflow variables
#     """
#
#     d = tf.reduce_sum(tf.square(x - y), 1)
#     return d

# def compute_contrastive_loss(left_feature, right_feature, label, margin):
#
#     """
#     Compute the contrastive loss as in
#
#
#     L = 0.5 * Y * D^2 + 0.5 * (Y-1) * {max(0, margin - D)}^2
#
#     **Parameters**
#      left_feature: First element of the pair
#      right_feature: Second element of the pair
#      label: Label of the pair (0 or 1)
#      margin: Contrastive margin
#
#     **Returns**
#      Return the loss operation
#
#     """
#
#     label = tf.to_float(label)
#     one = tf.constant(1.0)
#
#     d = compute_euclidean_distance(left_feature, right_feature)
#     d_sqrt = tf.sqrt(compute_euclidean_distance(left_feature, right_feature))
#     first_part = tf.multiply(one-label, d)# (Y-1)*(d)
#
#     max_part = tf.square(tf.maximum(margin-d_sqrt, 0))
#     second_part = tf.multiply(label, max_part)  # (Y) * max(margin - d, 0)
#
#     loss = 0.5 * tf.reduce_mean(first_part + second_part)
#
#     return loss


# def global_loss(source_feature, target_feature, feature_len, class_num, margin):
#     label = tf.eye(7)
#     g_loss = 0
#     for cls in range( class_num):
#         feature = tf.tile(target_feature[cls,:],[7,feature_len])
#         g_loss_cls = compute_contrastive_loss(source_feature, feature, label[:,cls], margin)
#         g_loss += g_loss_cls
#     g_loss = g_loss / class_num
#     return g_loss


def all_diffs(a, b):

    # Returns a tensor of all combinations of a - b

    return tf.expand_dims(a, axis=1) - tf.expand_dims(b, axis=0)









def normalize_perturbation(d, scope=None):
    with tf.name_scope(scope, 'norm_pert'):
        output = tf.nn.l2_normalize(d, axis=range(1, len(d.shape)))
    return output

def py_func(func, inp, Tout, stateful = True, name=None, grad_func=None):
    rand_name = 'PyFuncGrad' + str(np.random.randint(0,1E+8))
    tf.RegisterGradient(rand_name)(grad_func)
    g = tf.get_default_graph()
    with g.gradient_override_map({'PyFunc':rand_name}):
        return tf.py_func(func,inp,Tout,stateful=stateful, name=name)

def coco_forward(xw, y, m, name=None):
    # #pdb.set_trace()    [[2.   3.   7.75]
    #                                        [5.   4.   8.75]
    #                                        [7.   6.   9.75]]
    xw_copy = xw.copy()
    num = len(y)
    orig_ind = range(num)
    xw_copy[orig_ind,y] -= m
    return xw_copy

def coco_help(grad,y):
    grad_copy = grad.copy()
    return grad_copy

def coco_backward(op, grad):

    y = op.inputs[1]
    m = op.inputs[2]
    grad_copy = tf.py_func(coco_help,[grad,y],tf.float32)
    return grad_copy,y,m

def coco_func(xw,y,m, name=None):
    with tf.op_scope([xw,y,m],name,"Coco_func") as name:
        coco_out = py_func(coco_forward,[xw,y,m],tf.float32,name=name,grad_func=coco_backward)
        return coco_out

def cos_loss(x, y,  num_cls=7, reuse=tf.AUTO_REUSE, alpha=0.25, scale=10, name = 'cos_loss'):
    '''
    x: B x D - features
    y: B x 1 - labels
    num_cls: 1 - total class number
    alpah: 1 - margin
    scale: 1 - scaling paramter
    '''
    # define the classifier weights


    # get the scores after normalization

    #implemented by py_func
    #value = tf.identity(xw)
    #substract the marigin and scale it
    y = tf.argmax(y,axis=1,output_type=tf.int32)
    value = coco_func( x,y,alpha) * scale

    #implemented by tf api
    #margin_xw_norm = xw_norm - alpha
    #label_onehot = tf.one_hot(y,num_cls)
    #value = scale*tf.where(tf.equal(label_onehot,1), margin_xw_norm, xw_norm)


    # compute the loss as softmax loss
    cos_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=value))

    return cos_loss

def get_cos_distance(X1, X2, k, m, n):
    # calculate cos distance between two sets
    # more similar more big
    # (k,n) = X1.shape
    # (m,n) = X2.shape
    # k = 120
    # m = 7
    # n = 512

    X1_norm = tf.sqrt(tf.reduce_sum(tf.square(X1+10e-10), axis=1))
    X2_norm = tf.sqrt(tf.reduce_sum(tf.square(X2+10e-10), axis=1))

    X1_X2 = tf.matmul(X1, tf.transpose(X2))
    X1_X2_norm = tf.matmul(tf.reshape(X1_norm,[k,1]),tf.reshape(X2_norm,[1,m]))

    cos = X1_X2/X1_X2_norm
    return cos

def g_loss( X1, X2, label, batch_size, n_class, feature_len, margin=0.5):
    cos_m = get_cos_distance( X1, X2, batch_size, n_class, feature_len) #B*H*W,2
    # same
    p_loss = tf.reduce_mean(tf.reduce_sum(label - tf.multiply(cos_m,label), axis=-1))
    n_loss = tf.reduce_mean(tf.nn.relu( tf.reduce_max( tf.multiply(cos_m, tf.ones(( batch_size, n_class))-label), axis=-1  ) ))
    return  tf.reduce_mean( n_loss) + tf.reduce_mean( p_loss)

def masked_g_loss( X1, X2, label, mask, batch_size, n_class, feature_len, margin=0.5):
    mask_2 = tf.stack([mask,mask],axis=-1)
    cos_m = get_cos_distance( X1, X2, batch_size, n_class, feature_len)
    # same
    p_loss = tf.reduce_sum( (label - tf.multiply(cos_m,label))*mask_2, axis=-1) 
    p_loss = tf.reduce_sum(p_loss) /  tf.reduce_sum(mask)
    n_loss = tf.nn.relu(tf.reduce_max(tf.multiply(cos_m, tf.ones(( batch_size, n_class))-label)*mask_2 , axis=-1) ) 
    n_loss = tf.reduce_sum(n_loss) / tf.reduce_sum(mask)
    return  p_loss + n_loss #tf.reduce_mean( n_loss) + tf.reduce_mean( p_loss)


def multi_category_focal_loss1_fixed(y_true, y_pred,gamma=1.8):
        epsilon = 1.e-7
        # alpha = tf.constant(alpha, dtype=tf.float32)  
        # alpha = tf.constant([[1],[1],[1],[1],[1]], dtype=tf.float32)
        # alpha = tf.constant_initializer(alpha)
        gamma = float(gamma)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1-y_true, 1-y_pred)
        ce = -tf.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.multiply(weight, ce)
        loss = tf.reduce_mean(fl)
        return loss


def get_center_loss( features, labels, centers, class_num=7, margin=2):
    features = tf.nn.l2_normalize(features, dim=-1)
    centers = tf.nn.l2_normalize(centers, dim=-1)

    # labels = tf.to_int32(tf.argmax(labels, axis = -1))

    dist = euclidean_distance(features, centers)

    p_part = tf.reduce_max(tf.multiply(labels, dist), axis=1)

    n_part = tf.maximum(margin-tf.reduce_max(dist-tf.multiply(labels, dist), axis=1),0)


    loss = 1 * tf.reduce_mean(p_part + 0*n_part)


    # loss = compute_euclidean_distance( features, centers_batch)


    return loss


# def center_loss( features, labels, centers, class_num=7, margin=1):
#     features = tf.nn.l2_normalize(features, dim=1)
#     labels = tf.to_int32(tf.argmax(labels, axis = -1))
#
#     centers = tf.nn.l2_normalize(centers, dim=1)
#     centers_batch = tf.gather(centers, labels)
#
#     # diff = (1 - alfa) * (centers_batch - features)
#     # centers = tf.scatter_sub(centers, labels, diff)
#     loss = tf.reduce_sum(tf.square(features - centers_batch),axis=1)
#
#     # loss = compute_euclidean_distance( features, centers_batch)
#
#     p = 1 + loss*0
#
#
#     return p

#
def cos_center_loss( feature, label, centroid, n_class = 7, margin=0.5):
    cos_m = get_cos_distance( feature, centroid, k=120)
    # same
    p_loss = tf.reduce_sum(tf.multiply(cos_m,label), axis= 1)
    n_loss = tf.reduce_max(cos_m-tf.multiply(cos_m,label), axis=1)
    zeros = tf.ones((120,1),tf.float32)* 0.1
    w= tf.exp(tf.nn.l2_normalize(n_loss-p_loss))
    # w = tf.exp(tf.to_float(tf.equal( tf.reduce_max(tf.concat([tf.reshape(p_loss-n_loss,(120,1)), zeros],axis=1),axis=1, keepdims=True),zeros)))
    return  w

def center_loss( output, label, centroid, n_class = 7, margin=0.5):
    output = tf.nn.softmax(output, axis=1)
    p_loss = tf.reduce_sum(tf.multiply( output, label), axis= 1)
    n_loss = tf.reduce_max(output-tf.multiply(output,label), axis=1)
    zeros = tf.ones( (120,1), tf.float32)*0.2
    w = tf.exp(tf.to_float(tf.equal( tf.reduce_max(tf.concat([tf.reshape(p_loss-n_loss,(120,1)), zeros],axis=1),axis=1, keepdims=True),zeros)))
    w = tf.exp(tf.nn.l2_normalize( n_loss - p_loss))
    return  w

def center_loss_global( output, label, centroid, n_class = 7, margin=0.5):
    output = tf.nn.softmax(output, axis=1)
    p_loss = tf.reduce_sum(tf.multiply(output, label), axis=1)
    n_loss = tf.reduce_max(output - tf.multiply(output, label), axis=1)
    zeros = tf.ones((120, 1), tf.float32) * 0.2
    w =  tf.ones((120, 1), tf.float32)- tf.to_float(tf.equal(tf.reduce_max(tf.concat([tf.reshape(p_loss - n_loss, (120, 1)), zeros], axis=1), axis=1, keepdims=True), zeros))
    w = tf.exp(tf.nn.l2_normalize(p_loss - n_loss))
    return  w


def Coral_loss(source_scores, target_scores):
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
      source_scores, target_scores: Logits from inference().
      source_labels: Labels from distorted_inputs or inputs(). 2-D tensor
              of shape [batch_size]
      _lambda: A variable to trade off between coral and classification loss
    Returns:
      Loss tensor of type float.
    """
    # Calculate the average cross entropy loss across the batch.
    # source_labels = tf.cast(source_labels, tf.int64)
    # target_labels = tf.cast(target_labels, tf.int64)
    # classification_loss = tf.reduce_mean(
    #     tf.losses.sparse_softmax_cross_entropy(labels=source_labels, logits=source_scores))

    source_batch_size = tf.cast(tf.shape(source_scores)[0], tf.float32)
    target_batch_size = tf.cast(tf.shape(target_scores)[0], tf.float32)
    d = tf.cast(tf.shape(source_scores)[1], tf.float32)

    # Source covariance
    xm = source_scores - tf.reduce_mean(source_scores, 0, keep_dims=True)
    xc = tf.matmul(tf.transpose(xm), xm) / source_batch_size

    # Target covariance
    xmt = target_scores - tf.reduce_mean(target_scores, 0, keep_dims=True)
    xct = tf.matmul(tf.transpose(xmt), xmt) / target_batch_size

    coral_loss = tf.reduce_sum(tf.multiply((xc - xct), (xc - xct)))
    coral_loss /= 4 * d * d



    # The total loss is defined as the classification loss plus the coral loss.
    return coral_loss


def dice_coefficient(y_true, y_pred,smooth=0.00001):
    # y_pred = tf.argmax(y_pred,-1)
    y_true_f = tf.to_float(tf.layers.flatten(y_true))
    y_pred_f = tf.to_float(tf.layers.flatten(y_pred))
    intersection = tf.to_float(tf.reduce_sum(y_true_f * y_pred_f))
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def np_dice(y_true, y_pred,smooth=0.00001):
    # y_pred = tf.argmax(y_pred,-1)
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)

def masked_np_dice(y_true, y_pred,mask,smooth=0.00001):
    # y_pred = tf.argmax(y_pred,-1)
    mask_f = mask.flatten()
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    masked_gt = (np.delete(y_true_f,np.where(mask_f==0))[0]).astype(np.float32)
    masked_pd = (np.delete(y_pred_f,np.where(mask_f==0))[0]).astype(np.float32)
    assert masked_pd.shape == masked_pd.shape
    intersection = (masked_gt * masked_pd).sum()
    return (2. * intersection + smooth) / (masked_gt.sum() + masked_pd.sum() + smooth)

# def masked_dice_coefficient(y_true, y_pred,mask,smooth=0.00001):
#     # y_pred = tf.argmax(y_pred,-1)
#     mask_f = tf.layers.flatten(mask)
#     y_true_f = tf.layers.flatten(y_true)
#     y_pred_f = tf.layers.flatten(y_pred)
#     masked_gt = tf.to_float(np.delete(y_true_f,np.where(mask_f==0))[0])
#     masked_pd = tf.to_float(np.delete(y_pred_f,np.where(mask_f==0))[0])
#     assert masked_pd.shape == masked_pd.shape
#     intersection = tf.to_float(tf.reduce_sum(masked_gt * masked_pd))
#     return (2. * intersection + smooth) / (tf.reduce_sum(masked_gt) + tf.reduce_sum(masked_pd) + smooth)

from skimage import measure

def post(pred_seg, min_size=20):
    '''
    对输出结果后处理，只保留最大连通域
    pred_seg: 预测结构，numpy类型, shape = [DHW]
    min_size:不保留小于min_size个pixel的最大联通域。
    '''
    # pred_seg = pred.copy()
    [mask, num] = measure.label(pred_seg.astype('bool'), return_num=True)
    region = measure.regionprops(mask)
    box = []
    for i in range(num):
        box.append(region[i].area)
    if box != []:
        label_num = box.index(max(box)) + 1
        mask[mask != label_num] = 0
        mask[mask == label_num] = 1
        if mask.sum() <= min_size:
            mask[mask == 1] = 0
        pred_seg = mask.astype('uint8')

    return pred_seg
