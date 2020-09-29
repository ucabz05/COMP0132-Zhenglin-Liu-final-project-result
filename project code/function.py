# The function and some parameter parameter setting is referred from https://github.com/robertocalandra/the-feeling-of-success/tree/master/manu_sawyer/src/tensorflow_model_is_gripping
# with some modification
#
import scipy.ndimage
import numpy as np
import tensorflow as tf
import tf_slim as slim
import os
pj = os.path.join
full_dim = 256
crop_dim = 224
im_names = 'gelsightA_before gelsightA_during gelsightB_before gelsightB_during kinectA_rgb_before kinectA_rgb_during'.split()

def scale(im, scale, order = 3, hires = False):
  if hires == 'auto':
    hires = (im.dtype == np.uint8)

  if np.ndim(scale) == 0:
    new_scale = [scale, scale]
  # interpret scale as dimensions; convert integer size to a fractional scale
  elif ((scale[0] is None) or type(scale[0]) == type(0)) \
           and ((scale[1] is None) or type(scale[1]) == type(0)) \
           and (not (scale[0] is None and scale[1] is None)):
    # if the size of only one dimension is provided, scale the other to maintain the right aspect ratio
    if scale[0] is None:
      dims = (int(float(im.shape[0])/im.shape[1]*scale[1]),  scale[1])
    elif scale[1] is None:
      dims = (scale[0], int(float(im.shape[1])/im.shape[0]*scale[0]))
    else:
      dims = scale[:2]
      
    new_scale = [float(dims[0] + 0.4)/im.shape[0], float(dims[1] + 0.4)/im.shape[1]]
    # a test to make sure we set the floating point scale correctly
    result_dims = [int(new_scale[0]*im.shape[0]), int(new_scale[1]*im.shape[1])]
    assert tuple(result_dims) == tuple(dims)
  elif type(scale[0]) == type(0.) and type(scale[1]) == type(0.):
    new_scale = scale
    #new_scale = scale[1], scale[0]
  else:
    raise RuntimeError("don't know how to interpret scale: %s" % (scale,))
  #scale_param = new_scale if im.ndim == 2 else (new_scale[0], new_scale[1], 1)
  scale_param = new_scale if im.ndim == 2 else (new_scale[0], new_scale[1], 1)

  if hires:
    #sz = map(int, (scale_param*im.shape[1], scale_param*im.shape[0]))
    sz = map(int, (scale_param[1]*im.shape[1], scale_param[0]*im.shape[0]))
    return from_pil(to_pil(im).resize(sz, Image.ANTIALIAS))
  else:
    res = scipy.ndimage.zoom(im, scale_param, order = order)
    # verify that zoom() returned an image of the desired size
    if (np.ndim(scale) != 0) and type(scale[0]) == type(0) and type(scale[1]) == type(0):
      assert res.shape[:2] == (scale[0], scale[1])
    return res

def resize(x):
    x = scale(x, (256, 256), 1)
    return tf.image.random_crop(x, (224, 224, 3))
    
def moving_avg(name, x, vals = {}, avg_win_size = 100):
  add_dict_list(vals, name, x)
  return np.mean(vals[name][-avg_win_size:])

def add_dict_list(m, k, v):
  if k in m:
    m[k].append(v)
  else:
    m[k] = [v]



import copy
class Struct:
  def __init__(self, *dicts, **fields):
    for d in dicts:
      for k, v in d.iteritems():
        setattr(self, k, v)
    self.__dict__.update(fields)

  def to_dict(self):
    return {a : getattr(self, a) for a in self.attrs()}

  def attrs(self):
    xs = set(dir(self)) - set(dir(Struct))
    xs = [x for x in xs if ((not (hasattr(self.__class__, x) and isinstance(getattr(self.__class__, x), property))) \
          and (not inspect.ismethod(getattr(self, x))))]
    return sorted(xs)
             
  def updated(self, other_struct_ = None, **kwargs):
    s = copy.deepcopy(self)
    if other_struct_ is not None:
      s.__dict__.update(other_struct_.to_dict())
    s.__dict__.update(kwargs)
    return s

  def copy(self):
    return copy.deepcopy(self)
  
  def __str__(self):
    attrs = ', '.join('%s=%s' % (a, getattr(self, a)) for a in self.attrs())
    return 'Struct(%s)' % attrs

def mkdir(path, make_all = True):
  # replacement for os.mkdir that does not throw an exception of the directory exists
  if not os.path.exists(path):
    if make_all:
      os.system('mkdir -p "%s"' % path)
    else:
      os.system('mkdir "%s"' % path)
  return path

class Params(Struct):
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

  @property
  def train_dir(self):
    return mkdir(pj(self.resdir, 'training'))

  @property
  def summary_dir(self):
    return mkdir(pj(self.resdir, 'summary'))

def base_v5():
  return Params(
    dsdir = './drive/My Drive/project/result/',
    base_lr = 1e-5,
    lr_gamma = 0.5,
    step_size = 1500,
    batch_size = 32,
    opt_method = 'adam',
    model_iter = 5000,
    train_iters = 5001)

def gel_im_fulldata_v5():
  return base_v5().updated(
    description = 'GelSight + image trained on full dataset',
    resdir = './drive/My Drive/project/result/gel-im-fulldata-v5/',
    batch_size = 16,
    step_size = 2000,
    train_iters = 8001,
    dset_names = ['full_unbalanced'],
    inputs = ['gel', 'im'])

def im_fulldata_v5():
  return base_v5().updated(
    description = 'Image trained on full dataset',
    resdir = './drive/My Drive/project/result/im-fulldata-v5/',
    batch_size = 16,
    step_size = 2000,
    train_iters = 7001,
    dset_names = ['full_unbalanced'],
    inputs = ['im'])

def gel_fulldata_v5():
  return base_v5().updated(
    description = 'Gel trained on full dataset',
    resdir = './drive/My Drive/project/result/gel-fulldata-v5/',
    batch_size = 16,
    step_size = 2000,
    train_iters = 7001,
    dset_names = ['full_unbalanced'],
    inputs = ['gel'])

def during_only_v5():
  return base_v5().updated(
    description = 'trained on after grasping image only',
    resdir = './drive/My Drive/project/result/during_only_v5/',
    batch_size = 16,
    step_size = 2000,
    train_iters = 7001,
    dset_names = ['full_unbalanced'],
    inputs = ['during'])

def normalize_ims(im):
  if type(im) == type(np.array([])):
    im = im.astype('float32')
  else:
    im = tf.cast(im, tf.float32)
  return -1. + (2./255) * im 

def vgg_arg_scope(reuse = False, weight_decay=0.0008):
  """Defines the VGG arg scope.

  Args:
    weight_decay: The l2 regularization coefficient.

  Returns:
    An arg_scope.
  """
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu, reuse = reuse,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc

def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16',
           update_top_only = False,
           fc_conv_padding='VALID',
           reuse = None):
  with tf.compat.v1.variable_scope(scope, 'vgg_16', [inputs]) as sc, \
           slim.arg_scope(vgg_arg_scope(reuse = reuse)):
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      # if update_top_only:
      #   net = tf.stop_gradient(net)
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      net = slim.max_pool2d(net, [2, 2], scope='pool5')
      # Use conv2d instead of fully_connected layers.
      net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout6')
      net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
      net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                         scope='dropout7')
      if num_classes is not None:
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
      # Convert end_points_collection into a end_point dict.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if spatial_squeeze:
        net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      return net, end_points
def vgg_dual_16(inputs1,
                inputs2,
                num_classes=1000,
                is_training=True,
                dropout_keep_prob=0.5,
                spatial_squeeze=True,
                scope='vgg_16',
                update_top_only = False,
                fc_conv_padding='VALID',
                reuse = False):
  with tf.compat.v1.variable_scope(scope, 'vgg_16', [inputs1]) as sc:
    end_points_collection = sc.name + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      nets = []
      for i, inputs in enumerate([inputs1, inputs2]):
        print(i > 0)
#         with slim.arg_scope(vgg_arg_scope(reuse = tf.compat.v1.AUTO_REUSE or (i > 0))):
        with slim.arg_scope(vgg_arg_scope(reuse = reuse or (i > 0))):
          net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
          net = slim.max_pool2d(net, [2, 2], scope='pool1')
          net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
          net = slim.max_pool2d(net, [2, 2], scope='pool2')
          net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
          net = slim.max_pool2d(net, [2, 2], scope='pool3')
          net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
          net = slim.max_pool2d(net, [2, 2], scope='pool4')
          # if update_top_only:
          #   net = tf.stop_gradient(net)
          net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
          nets.append(net)
      with slim.arg_scope(vgg_arg_scope(reuse = reuse)):
        net = tf.concat(nets, 3)
        net = slim.conv2d(net, 512, [1, 1], scope='conv6')
        net = slim.max_pool2d(net, [2, 2], stride = 2, scope='pool6')
        #net = slim.max_pool2d(net, [2, 2], scope='pool6')
        # Use conv2d instead of fully_connected layers.
        #net = slim.conv2d(net, 4096, [7, 7], padding=fc_conv_padding, scope='fc6')
        net = slim.conv2d(net, 2048, [7, 7], padding=fc_conv_padding, scope = 'fc6_')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope = 'dropout6')
        # net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        # net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope = 'dropout7')

        net = slim.conv2d(net, 2048, [1, 1], scope='fc7_')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training, scope = 'dropout7_')
        if num_classes is not None:
          net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                            normalizer_fn = None, scope = 'fc8')
        # Convert end_points_collection into a end_point dict.
        end_points = slim.utils.convert_collection_to_dict(end_points_collection)
        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
          end_points[sc.name + '/fc8'] = net
        return net, end_points

def vgg_gel2(gel0_pre, gel0_post,
             gel1_pre, gel1_post,
             num_classes = 2,
             is_training = True,
             update_top_only = False,
             fc_conv_padding='VALID',
             dropout_keep_prob = 0.5,
             diff = True,
             reuse = False,
             scope = 'vgg_16'):
  print('reuse =', reuse)
  if diff:
    nets = []
    r = reuse
    if gel0_pre is not None:
      nets.append(vgg_dual_16(gel0_post - gel0_pre, gel0_post, reuse = r, is_training = is_training, 
                              num_classes = None, update_top_only = update_top_only, scope = scope)[0])
      r = True
    if gel1_pre is not None:    
      nets.append(vgg_dual_16(gel1_post - gel1_pre, gel1_post, reuse = r, is_training = is_training, 
                              num_classes = None, update_top_only = update_top_only, scope = scope)[0])
      r = True
    return tf.concat(nets, 1)
  else:
    net0 = pair_vgg(gel0_post, gel0_pre, is_training = is_training, update_top_only = update_top_only, scope = scope)
    net1 = pair_vgg(gel1_post, gel1_pre, reuse = True, is_training = is_training, update_top_only = update_top_only, scope = scope)

    with tf.variable_scope(scope, scope), \
             slim.arg_scope(vgg_arg_scope(reuse)):
      net = tf.concat([net0, net1], 3)
      net = slim.conv2d(net, 2048, [7, 7], padding = fc_conv_padding, scope = 'fc6_')
      net = slim.dropout(net, dropout_keep_prob, is_training = is_training, scope = 'dropout6')
      net = slim.conv2d(net, 2048, [1, 1], scope =  'fc7_')
      net = slim.dropout(net, dropout_keep_prob, is_training = is_training, scope = 'dropout7')
      if num_classes is not None:
        net = slim.conv2d(net, num_classes, [1, 1], activation_fn = None, normalizer_fn = None, scope = 'fc8_')
      net = net[:, 0, 0, :]
  return net



def make_model(inputs, pr, train, update_top_only=False,reuse = False, both_ims = True):
  n = normalize_ims
  with slim.arg_scope(vgg_arg_scope(False)):
    feats = []
    if 'during' in pr.inputs:
      feats.append(vgg_16(n(inputs['gelsightA_during']), num_classes = None, update_top_only = update_top_only,
                              scope = 'gel_vgg16', is_training = train)[0])
      feats.append(vgg_16(n(inputs['gelsightB_during']), num_classes = None, update_top_only = update_top_only,
                              scope = 'gel_vgg16', reuse = True, is_training = train)[0])
      feats.append(vgg_16(n(inputs['kinectA_rgb_during']), num_classes = None, update_top_only = update_top_only,
                              scope = 'im_vgg16', is_training = train)[0])

    if 'gel' in pr.inputs:
#       if not hasattr(pr, 'gels') or (0 in pr.gels):
#         print ('Using gel 0')
      gel0_pre, gel0_post = n(inputs['gelsightA_before']), n(inputs['gelsightA_during'])
      gel1_pre, gel1_post = n(inputs['gelsightB_before']), n(inputs['gelsightB_during'])

      if both_ims:
          feats.append(vgg_gel2(
            gel0_pre, gel0_post, 
            gel1_pre, gel1_post,
            is_training = train, 
            num_classes = None,
            update_top_only = update_top_only,
            reuse = reuse,
            scope = 'gel_vgg16'))
    
      else:
          if gel0_pre is not None:
            feats.append(vgg_16(gel0_post - gel0_pre, num_classes = None, update_top_only = update_top_only,
                                    scope = 'gel_vgg16', is_training = train, reuse = reuse)[0])
          if gel1_pre is not None:
            feats.append(vgg_16(gel1_post - gel1_pre, num_classes = None, update_top_only = update_top_only,
                                    scope = 'gel_vgg16', reuse = True, is_training = train)[0])

    if 'im' in pr.inputs:
      feats.append(vgg_16(n(inputs['kinectA_rgb_before']), num_classes = None, update_top_only = update_top_only,
                              scope = 'im_vgg16', is_training = train, reuse = reuse)[0])
      feats.append(vgg_16(n(inputs['kinectA_rgb_during']), num_classes = None, update_top_only = update_top_only,
                              scope = 'im_vgg16', reuse = True, is_training = train)[0])


    net = tf.concat(feats, 1)
    if update_top_only:
      net = tf.stop_gradient(net)

    # logits = slim.fully_connected(net, 2, scope = 'logits', activation_fn = tf.nn.softmax, reuse = reuse)
    logits = slim.fully_connected(net, 2, scope = 'logits', activation_fn = None, reuse = reuse)
    # log=feats[1]
  return logits

def read_example(rec_queue, pr):
  with tf.device('/device:GPU:0'):
    reader = tf.compat.v1.TFRecordReader()
    k, s = reader.read(rec_queue)
    feats = {'is_gripping' : tf.io.FixedLenFeature([], tf.int64)}

    if 'during' in pr.inputs:
      feats.update({'gelsightA_during' : tf.io.FixedLenFeature([], dtype=tf.string),
                    'gelsightB_during' : tf.io.FixedLenFeature([], dtype=tf.string),
                    'kinectA_rgb_during' : tf.io.FixedLenFeature([], dtype=tf.string)})

    if 'gel' in pr.inputs:
      feats.update({'gelsightA_before' : tf.io.FixedLenFeature([], dtype=tf.string),
                    'gelsightA_during' : tf.io.FixedLenFeature([], dtype=tf.string),
                    'gelsightB_before' : tf.io.FixedLenFeature([], dtype=tf.string),
                    'gelsightB_during' : tf.io.FixedLenFeature([], dtype=tf.string)})
    if 'im' in pr.inputs:
      feats.update({'kinectA_rgb_before' : tf.io.FixedLenFeature([], dtype=tf.string),
                    'kinectA_rgb_during' : tf.io.FixedLenFeature([], dtype=tf.string)})

    example = tf.io.parse_single_example(s, features = feats)

    out = {'is_gripping' : example['is_gripping']}

    base_names = ['gel', 'im', 'during']
    for base_name in base_names:
      if base_name not in pr.inputs:
        continue
      if base_name=='im':
        base_name='kinect'
      names = [name for name in im_names if name.startswith(base_name)]
      if base_name=='during':
        names=['gelsightA_during', 'gelsightB_during', 'kinectA_rgb_during']
      ims = []
      for name in names:
        im = example[name]
        im = tf.image.decode_png(im)
        im = tf.cast(im, tf.float32)
        im.set_shape((256, 256, 3))
        ims.append(im)

      combo = tf.concat(ims, 2)
      combo = tf.image.random_crop(combo, (crop_dim, crop_dim, combo.shape[2]))
      combo = tf.image.random_flip_left_right(combo)
      if name.startswith('gel'):
        combo = tf.image.random_flip_up_down(combo)

      print ('group:')
      start = 0
      for name, im in zip(names, ims):
        out[name] = combo[:, :, start : start + im.shape[-1]]
        print (name, out[name].shape)
        start += im.shape[-1]
      
    return out

def read_data(pr,sess):
  tf_files = [pj(pr.dsdir, 'train256_set2.tf'),pj(pr.dsdir, 'train256_set3.tf'),pj(pr.dsdir, 'train256_set4.tf')]
  print ('Tf files:', tf_files)
  queue = tf.compat.v1.train.string_input_producer(tf_files, num_epochs=50)
  sess.run(tf.compat.v1.local_variables_initializer())
  list_read=read_example(queue, pr).items()
  names, vals = zip(*list_read)
  vals = tf.compat.v1.train.shuffle_batch(vals, batch_size = pr.batch_size,
                  capacity = 2000, min_after_dequeue = 500)
  if len(names) == 1:
    vals = [vals]

  output = {}
  for k, v in zip(names, vals):
    output[k] = v
  return output