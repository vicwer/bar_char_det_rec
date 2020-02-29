from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

def inference(images, keep_probability, phase_train=True, weight_decay=0.0, reuse=None):

  logits, end_points = mobilenet(images, is_training=phase_train, weight_decay=weight_decay, reuse=None)
  #return end_points['squeeze'],logits #end_points['squeeze']
  return logits, end_points
def mobilenet(inputs,
          num_classes=1000,
          is_training=True,
          width_multiplier=1,
          reuse=None,
          weight_decay=0.0,
          scope='MobileNet'):
  """ MobileNet
  More detail, please refer to Google's paper(https://arxiv.org/abs/1704.04861).
  Args:
    inputs: a tensor of size [batch_size, height, width, channels].
    num_classes: number of predicted classes.
    is_training: whether or not the model is being trained.
    scope: Optional scope for the variables.
  Returns:
    logits: the pre-softmax activations, a tensor of size
      [batch_size, `num_classes`]
    end_points: a dictionary from components of the network to the corresponding
      activation.
  """
  def _depthwise_separable_conv(inputs,
                                num_pwc_filters,
                                width_multiplier,
                                sc,
                                downsample=False):
    """ Helper function to build the depth-wise separable convolution layer.
    """
    num_pwc_filters = round(num_pwc_filters * width_multiplier)
    _stride = 2 if downsample else 1

    # skip pointwise by setting num_outputs=None
    depthwise_conv = slim.separable_convolution2d(inputs,
                                                  num_outputs=None,
                                                  stride=_stride,
                                                  depth_multiplier=1,
                                                  kernel_size=[3, 3],
                                                  normalizer_fn=slim.batch_norm,
                                                  scope=sc+'/depthwise_conv')

    bn = depthwise_conv #slim.batch_norm(depthwise_conv, scope=sc+'/dw_batch_norm')
    print(bn)
    pointwise_conv = slim.convolution2d(bn,
                                        num_pwc_filters,
                                        kernel_size=[1, 1],
                                        normalizer_fn=slim.batch_norm,
                                        scope=sc+'/pointwise_conv')
    bn = pointwise_conv #slim.batch_norm(pointwise_conv, scope=sc+'/pw_batch_norm')
    return bn      

  batch_norm_params = {
    # Decay for the moving averages.
    'decay': 0.995,
    # epsilon to prevent 0s in variance.
    'epsilon': 0.001,
    # force in-place updates of mean and variance estimates
    'updates_collections': None,
    # Moving averages ends up in the trainable variables collection
    'variables_collections': [ tf.GraphKeys.TRAINABLE_VARIABLES ],
    'is_training': is_training
  }
  end_points = {} 
  with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                      weights_initializer=slim.xavier_initializer_conv2d(uniform=True),
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      normalizer_fn=slim.batch_norm,
                      normalizer_params=batch_norm_params,
                      activation_fn=None):#, 
                      #outputs_collections=[end_points_collection]):
    with tf.variable_scope(scope,[inputs], reuse=reuse):
    #end_points_collection = sc.name + '_end_points'

      with slim.arg_scope([slim.batch_norm],
                          is_training=is_training,
                          activation_fn=tf.nn.relu):
        print(inputs) #batch_size*67*67*3 or  65~80
        net = slim.convolution2d(inputs, round(32 * width_multiplier), [3, 3], stride=1,padding='SAME',scope='conv_1')# padding='SAME', padding='VALID',
        print(net)
        #net = slim.batch_norm(net, scope='conv_1/batch_norm')
        net = _depthwise_separable_conv(net, 48, width_multiplier, sc='conv_ds_2')
        print(net)
        net = _depthwise_separable_conv(net, 96, width_multiplier,downsample=True, sc='conv_ds_3')
        print(net)
        net = _depthwise_separable_conv(net, 96, width_multiplier, sc='conv_ds_4')
        print(net)
        net = _depthwise_separable_conv(net, 192, width_multiplier, downsample=True, sc='conv_ds_5') #downsample=True,
        print(net)
        net = _depthwise_separable_conv(net, 192, width_multiplier, sc='conv_ds_6')
        print(net)
        net = _depthwise_separable_conv(net, 192, width_multiplier, sc='conv_ds_7')
        print(net)
        net = _depthwise_separable_conv(net, 384, width_multiplier, downsample=True, sc='conv_ds_8')
        print(net)
        net = _depthwise_separable_conv(net, 384, width_multiplier, sc='conv_ds_9')
        print(net)
        net = _depthwise_separable_conv(net, 384, width_multiplier, sc='conv_ds_10')
        print(net)
        net = _depthwise_separable_conv(net, 384, width_multiplier, sc='conv_ds_11')
        print(net)
        net = _depthwise_separable_conv(net, 384, width_multiplier, sc='conv_ds_12')
        print(net)
        net = _depthwise_separable_conv(net, 768, width_multiplier, downsample=True,  sc='conv_ds_13')
        print(net)
        end_points['conv_ds_13'] = net
        net = _depthwise_separable_conv(net, 768, width_multiplier,sc='conv_ds_14')
        print(net)
        net = slim.avg_pool2d(net, [5, 5], scope='avg_pool_15')
        print(net)
        end_points['avg_pool_15'] = net

    logits = tf.squeeze(net, [1, 2], name='logits')
    print(logits)
    end_points['logits'] = logits
  pre_embeddings = slim.fully_connected(logits, 128,activation_fn=None,
                                        weights_initializer=tf.truncated_normal_initializer(stddev=0.1), 
                                        weights_regularizer=slim.l2_regularizer(0.0),
                                        normalizer_fn=slim.batch_norm,
                                        normalizer_params=batch_norm_params,
                                        scope='Bottleneck', reuse=False)
  #pre_embeddings = slim.batch_norm(pre_embeddings , scope='Bottleneck/batch_norm')
  end_points['Bottleneck'] = pre_embeddings
  print(pre_embeddings)
  embeddings = tf.nn.l2_normalize(pre_embeddings, 1, 1e-10, name='embeddings')
  print(embeddings)
  end_points['embeddings'] = embeddings
  return embeddings, end_points

mobilenet.default_image_size = 224


def mobilenet_arg_scope(weight_decay=0.0):
  """Defines the default mobilenet argument scope.
  Args:
    weight_decay: The weight decay to use for regularizing the model.
  Returns:
    An `arg_scope` to use for the MobileNet model.
  """
  with slim.arg_scope(
      [slim.convolution2d, slim.separable_convolution2d],
      weights_initializer=slim.initializers.xavier_initializer(),
      biases_initializer=slim.init_ops.zeros_initializer(),
      weights_regularizer=slim.l2_regularizer(weight_decay)) as sc:
    return sc
