import tensorflow as tf
import numpy as np
from utils.confutil import register_conf
from layers.tf_ops.pointSIFT_op.pointSIFT_op import pointSIFT_select, pointSIFT_select_four
from layers.tf_ops.grouping.tf_grouping import group_point, query_ball_point, knn_point
from layers.tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point
from layers.tf_ops.interpolation.tf_interpolate import three_nn, three_interpolate
import pdb
# last fully conneted layer, containing 21.conv1d, 22.dropout, 23.conv1d

@register_conf(name="SIFT-fc", scope="layer", conf_func="self")
class pointSIFT_fc_layer(tf.keras.layers.Layer):
    def __init__(self, out_channel, bn=True, bn_decay=0.9, label='fc', num_class=21, **kwargs):
        super(pointSIFT_fc_layer, self).__init__()
        self.bn = bn
        self.bn_decay = bn_decay
        self._name = label
        self.num_class = num_class
        self.out_channel = out_channel
        self.sub_layers = dict()

    def batch_normalization(self, inputs, name, training):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.BatchNormalization(
                momentum=self.bn_decay, name=name)
            self.sub_layers[name] = layer
        return layer(inputs, training=training)

    def conv1d(self, inputs, channel, kernel_size, name, strides, padding, activation_fn, training):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.Conv1D(
                channel, kernel_size, use_bias=False,strides=strides, padding=padding, activation=None)
            self.sub_layers[name] = layer
        x = layer(inputs, training=training)
        if self.bn:
            x = self.batch_normalization(x, training=training, name=name+"-BN")
        if activation_fn == 'ReLU':
            x = tf.nn.relu(x)
        return x

    def dropout(self, inputs, name, training):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.Dropout(0.5)
            self.sub_layers[name] = layer
        return layer(inputs, training)

    def call(self, inputs, training=None, **kwargs):
        points = inputs[0]
        if training is None:
            training = tf.keras.backend.learning_phase()
        seg = self.conv1d(points, self.out_channel, 1, self._name+'-conv0', 1,
                          padding='VALID', activation_fn='ReLU', training=training)
        seg = self.dropout(seg, self._name+'-dropout', training=training)
        seg = self.conv1d(seg, self.num_class, 1, self._name+'-conv1', 1, padding='VALID',
                          activation_fn='None', training=training)
        return seg

# a layer that appears 3 times in the network, tf.concat + conv1d to associate inputs

@register_conf(name="SIFT-associate", scope="layer", conf_func="self")
class associate_module(tf.keras.layers.Layer):
    def __init__(self, channel, bn=True, bn_decay=0.9, label='asso', **kwargs):
        super(associate_module, self).__init__()
        self.channel = channel
        self.bn = bn
        self.bn_decay = bn_decay
        self._name = label
        self.sub_layers = dict()

    def batch_normalization(self, inputs, name, training):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.BatchNormalization(
                momentum=self.bn_decay, name=name)
            self.sub_layers[name] = layer
        return layer(inputs, training=training)

    def conv1d(self, inputs, channel, kernel_size, name, strides, padding, activation_fn, training):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.Conv1D(
                channel, kernel_size, use_bias=False,strides=strides, padding=padding, activation=None)
            self.sub_layers[name] = layer
        x = layer(inputs, training=training)
        if self.bn:
            x = self.batch_normalization(x, training=training, name=name+"-BN")
        if activation_fn == 'ReLU':
            x = tf.nn.relu(x)
        return x

    def call(self, inputs, training=None, **kwargs):
        points = []
        if training is None:
            training = tf.keras.backend.learning_phase()
        # make sure that points is a list but not a tuple
        for part in inputs:
            points.append(part)
        points = tf.concat(points, axis=-1)
        points = self.conv1d(points, self.channel, 1, self._name+'-conv', 1, padding='VALID',
                             activation_fn='ReLU', training=training)
        return points

@register_conf(name="SIFT-res-module", scope="layer", conf_func="self")
class pointSIFT_res_module(tf.keras.layers.Layer):
    def __init__(self, radius, out_channel, bn_decay=0.9, label='SIFT-res', bn=True, use_xyz=True, same_dim=False, merge='add', **kwargs):
        super(pointSIFT_res_module, self).__init__()
        self.radius = radius
        self.out_channel = out_channel
        self.bn_decay = bn_decay
        self._name = label
        self.bn = bn
        self.use_xyz = use_xyz
        self.same_dim = same_dim
        self.merge = merge
        self.sub_layers = dict()

    def batch_normalization(self, inputs, name, training):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.BatchNormalization(
                momentum=self.bn_decay, name=name)
            self.sub_layers[name] = layer
        return layer(inputs, training=training)

    # conv2d from pointSIFT tf_util put bn layer between conv2d and relu (why?),
    # this function simulates that. when first called, it builds the layer before
    # it uses the layer
    def conv2d(self, inputs, channel, kernel_size, name, strides, padding, activation_fn, training):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.Conv2D(
                channel, kernel_size, use_bias=False, strides=strides, padding=padding, activation=None)
            self.sub_layers[name] = layer
        x = layer(inputs, training=training)
        if self.bn:
            x = self.batch_normalization(x, training=training, name=name+"-BN")
        if activation_fn == 'ReLU':
            x = tf.nn.relu(x)
        return x

    def conv1d(self, inputs, channel, kernel_size, name, strides, padding, activation_fn, training):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.Conv1D(
                channel, kernel_size, use_bias=False,strides=strides, padding=padding, activation=None)
            self.sub_layers[name] = layer
        x = layer(inputs, training=training)
        if self.bn:
            x = self.batch_normalization(x, training=training, name=name+"-BN")
        if activation_fn == 'ReLU':
            x = tf.nn.relu(x)
        return x

    def call(self, inputs, training=None, **kwargs):
        xyz, points = inputs
        if training is None:
            training = tf.keras.backend.learning_phase()
        # conv 1
        _, new_points, idx, _ = pointSIFT_group(self.radius,
                                                xyz,
                                                points,
                                                use_xyz=self.use_xyz)
        for i in range(3):
            new_points = self.conv2d(new_points, self.out_channel, [1, 2],
                                     self._name+'-c0_conv%d' % (i), [1, 2], padding='VALID',
                                     activation_fn='ReLU', training=training)
        new_points = tf.squeeze(new_points, [2])
        # conv 2
        _, new_points, idx, _ = pointSIFT_group_with_idx(xyz,
                                                         idx=idx,
                                                         points=new_points,
                                                         use_xyz=self.use_xyz)
        for i in range(3):
            if i == 2:
                act = 'None'
            else:
                act = 'ReLU'
            new_points = self.conv2d(new_points, self.out_channel, [1, 2],
                                     self._name+'-c1_conv%d' % (i), [1, 2], padding='VALID',
                                     activation_fn=act, training=training)
        new_points = tf.squeeze(new_points, [2])
        # residual part
        if points is not None:
            if self.same_dim is True:
                points = self.conv1d(points, self.out_channel, 1, self._name+'-merge_channel_fc', 1, padding='VALID', activation_fn='ReLU', training=training)
            if self.merge == 'add':
                new_points = new_points + points
            elif self.merge == 'concat':
                new_points = tf.concat([new_points, points], axis=-1)
            else:
                print("way not found!!")
        new_points = tf.nn.relu(new_points)
        return xyz, new_points, idx

@register_conf(name="SIFT-module", scope="layer", conf_func="self")
class pointSIFT_module(tf.keras.layers.Layer):
    def __init__(self, radius, out_channel, bn_decay=0.9, label='SIFT', bn=True, use_xyz=True, **kwargs):
        super(pointSIFT_module, self).__init__()
        self.radius = radius
        self.out_channel = out_channel
        self.bn_decay = bn_decay
        self._name = label
        self.bn = bn
        self.use_xyz = use_xyz
        self.sub_layers = dict()

    def batch_normalization(self, inputs, name, training):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.BatchNormalization(
                momentum=self.bn_decay, name=name)
            self.sub_layers[name] = layer
        return layer(inputs, training=training)

    def conv2d(self, inputs, channel, kernel_size, name, strides, padding, activation_fn, training):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.Conv2D(
                channel, kernel_size, use_bias=False,strides=strides, padding=padding, activation=None)
            self.sub_layers[name] = layer
        x = layer(inputs, training=training)
        if self.bn:
            x = self.batch_normalization(x, training=training, name=name+"-BN")
        if activation_fn == 'ReLU':
            x = tf.nn.relu(x)
        return x

    def call(self, inputs, training=None, **kwargs):
        xyz, points = inputs
        if training is None:
            training = tf.keras.backend.learning_phase()
        new_xyz, new_points, idx, grouped_xyz = pointSIFT_group(self.radius,
                                                                xyz,
                                                                points,
                                                                self.use_xyz)
        for i in range(3):
            new_points = self.conv2d(new_points, self.out_channel, [1, 2],
                                     self._name+'-conv%d' % (i), [1, 2], padding='VALID',
                                     activation_fn='ReLU', training=training)

        new_points = self.conv2d(new_points, self.out_channel, [1, 1],
                                 self._name+'-conv_fc', [1, 1], padding='VALID',
                                 activation_fn='ReLU', training=training)
        new_points = tf.squeeze(new_points, [2])
        return new_xyz, new_points, idx

@register_conf(name="pointnet-sa-module", scope="layer", conf_func="self")
class pointnet_sa_module(tf.keras.layers.Layer):
    def __init__(self, npoint, radius, nsample, mlp, mlp2=None, group_all=False, bn_decay=0.9, bn=True, pooling='max', use_xyz=True, label='sa'):
        super(pointnet_sa_module, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp = mlp
        self.mlp2 = mlp2
        self.group_all = group_all
        self.bn_decay = bn_decay
        self.bn = bn
        self.pooling = pooling
        self.use_xyz = use_xyz
        self._name = label
        self.sub_layers = dict()

    def batch_normalization(self, inputs, name, training):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.BatchNormalization(
                momentum=self.bn_decay, name=name)
            self.sub_layers[name] = layer
        return layer(inputs, training=training)

    def conv2d(self, inputs, channel, kernel_size, name, strides, padding, activation_fn, training):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.Conv2D(
                channel, kernel_size, use_bias=False,strides=strides, padding=padding, activation=None)
            self.sub_layers[name] = layer
        x = layer(inputs, training=training)
        if self.bn:
            x = self.batch_normalization(x, training=training, name=name+"-BN")
        if activation_fn == 'ReLU':
            x = tf.nn.relu(x)
        return x

    def call(self, inputs, training=None, **kwargs):
        xyz, points = inputs
        if training is None:
            training = tf.keras.backend.learning_phase()
        # Sample and grouping
        if self.group_all:
            self.nsample = xyz.get_shape()[1].value
            new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(
                xyz, points, self.use_xyz)
        else:
            new_xyz, new_points, idx, grouped_xyz = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, knn=False, use_xyz=self.use_xyz)
        # Point feature embedding
        for i, num_out_channel in enumerate(self.mlp):
            new_points = self.conv2d(new_points, num_out_channel, [1, 1],
                                     self._name+'-conv%d' % (i), [1, 1], padding='VALID',
                                     activation_fn='ReLU', training=training)
        # Pooling in local regions
        if self.pooling == 'max':
            new_points = tf.compat.v1.reduce_max(
                new_points, axis=[2], keepdims=True, name='maxpool')
        elif self.pooling == 'avg':
            new_points = tf.compat.v1.reduce_mean(
                new_points, axis=[2], keepdims=True, name='avgpool')
        else:
            print("not implemented")
        new_points = tf.squeeze(new_points, [2])
        return new_xyz, new_points, idx



@register_conf(name="pointnet-fp-module", scope="layer", conf_func="self")
class pointnet_fp_module(tf.keras.layers.Layer):
    def __init__(self, mlp, bn_decay=0.9, bn=True, label='fp'):
        super(pointnet_fp_module, self).__init__()
        self.mlp = mlp
        self.bn_decay = bn_decay
        self.bn = bn
        self._name = label
        self.sub_layers = dict()

    def batch_normalization(self, inputs, name, training):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.BatchNormalization(
                momentum=self.bn_decay, name=name)
            self.sub_layers[name] = layer
        return layer(inputs, training=training)

    def conv2d(self, inputs, channel, kernel_size, name, strides, padding, activation_fn, training):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.Conv2D(
                channel, kernel_size, use_bias=False,strides=strides, padding=padding, activation=None)
            self.sub_layers[name] = layer
        x = layer(inputs, training=training)
        if self.bn:
            x = self.batch_normalization(x, training=training, name=name+"-BN")
        if activation_fn == 'ReLU':
            x = tf.nn.relu(x)
        return x

    def call(self, inputs, training=None, **kwargs):
        xyz1, xyz2, points1, points2 = inputs
        if training is None:
            training = tf.keras.backend.learning_phase()
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.compat.v1.reduce_sum((1.0 / dist), axis=2, keepdims=True)
        norm = tf.tile(norm, [1, 1, 3])
        weight = (1.0 / dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)
        if points1 is not None:
            # B,ndataset1,nchannel1+nchannel2
            new_points1 = tf.concat(
                axis=2, values=[interpolated_points, points1])
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(self.mlp):
            new_points1 = self.conv2d(new_points1, num_out_channel, [1, 1],
                                      self._name+'-conv_%d' % (i), [1, 1],
                                      padding='VALID', activation_fn='ReLU', training=training)
        new_points1 = tf.squeeze(new_points1, [2])
        return new_points1


def pointSIFT_group(radius, xyz, points, use_xyz=True):
    idx = pointSIFT_select(xyz, radius)
    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, 8, 3)
    # translation normalization
    grouped_xyz -= tf.tile(tf.expand_dims(xyz, 2), [1, 1, 8, 1])
    if points is not None:
        # (batch_size, npoint, 8, channel)
        grouped_points = group_point(points, idx)
        if use_xyz:
            # (batch_size, npoint, 8/32, 3+channel)
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return xyz, new_points, idx, grouped_xyz


def pointSIFT_group_with_idx(xyz, idx, points, use_xyz=True):
    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, 8, 3)
    # translation normalization
    grouped_xyz -= tf.tile(tf.expand_dims(xyz, 2), [1, 1, 8, 1])
    if points is not None:
        # (batch_size, npoint, 8/32, channel)
        grouped_points = group_point(points, idx)
        if use_xyz:
            # (batch_size, npoint, 8/32, 3+channel)
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz
    return xyz, new_points, idx, grouped_xyz


def pointSIFT_group_four(radius, xyz, points, use_xyz=True):
    idx = pointSIFT_select_four(xyz, radius)
    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, 32, 3)
    # translation normalization
    grouped_xyz -= tf.tile(tf.expand_dims(xyz, 2), [1, 1, 32, 1])
    if points is not None:
        # (batch_size, npoint, 8/32, channel)
        grouped_points = group_point(points, idx)
        if use_xyz:
            # (batch_size, npoint, 8/32, 3+channel)
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return xyz, new_points, idx, grouped_xyz


def pointSIFT_group_four_with_idx(xyz, idx, points, use_xyz=True):
    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, 8/32, 3)
    # translation normalization
    grouped_xyz -= tf.tile(tf.expand_dims(xyz, 2), [1, 1, 32, 1])
    if points is not None:
        # (batch_size, npoint, 8/32, channel)
        grouped_points = group_point(points, idx)
        if use_xyz:
            # (batch_size, npoint, 8/32, 3+channel)
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz
    return xyz, new_points, idx, grouped_xyz


def sample_and_group(npoint, radius, nsample, xyz, points, knn=False, use_xyz=True):
    '''
    Input:
        npoint: int32
        radius: float32
        nsample: int32
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        knn: bool, if True use kNN instead of radius search
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Output:
        new_xyz: (batch_size, npoint, 3) TF tensor
        new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
        idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
        grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
            (subtracted by seed point XYZ) in local regions
    '''

    new_xyz = gather_point(xyz, farthest_point_sample(
        npoint, xyz))  # (batch_size, npoint, 3)
    if knn:
        _, idx = knn_point(nsample, xyz, new_xyz)
    else:
        idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, nsample, 3)
    grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2),
                           [1, 1, nsample, 1])  # translation normalization
    if points is not None:
        # (batch_size, npoint, nsample, channel)
        grouped_points = group_point(points, idx)
        if use_xyz:
            # (batch_size, npoint, nample, 3+channel)
            new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)
        else:
            new_points = grouped_points
    else:
        new_points = grouped_xyz

    return new_xyz, new_points, idx, grouped_xyz


def sample_and_group_all(xyz, points, use_xyz=True):
    '''
    Inputs:
        xyz: (batch_size, ndataset, 3) TF tensor
        points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
        use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
    Outputs:
        new_xyz: (batch_size, 1, 3) as (0,0,0)
        new_points: (batch_size, 1, ndataset, 3+channel) TF tensor
    Note:
        Equivalent to sample_and_group with npoint=1, radius=inf, use (0,0,0) as the centroid
    '''
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.constant(np.tile(np.array([0, 0, 0]).reshape((1, 1, 3)), (batch_size, 1, 1)),
                          dtype=tf.float32)  # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape(
        (1, 1, nsample)), (batch_size, 1, 1)))
    # (batch_size, npoint=1, nsample, 3)
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3))
    if points is not None:
        if use_xyz:
            # (batch_size, 16, 259)
            new_points = tf.concat([xyz, points], axis=2)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1)  # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz
