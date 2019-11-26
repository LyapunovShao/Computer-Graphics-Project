import shutil
import re
import layers
import tensorflow_core as tf
import math
import numpy as np
from os import path, makedirs
import logger
from utils.kerasutil import ModelCallback
from utils.confutil import object_from_conf, register_conf
import pdb
# A fake call to register
register_conf(name="adam", scope="optimizer", conf_func=lambda conf: tf.keras.optimizers.Adam(**conf))(None)
register_conf(name="sgd", scope="optimizer", conf_func=lambda conf: tf.keras.optimizers.SGD(**conf))(None)

register_conf(name="exponential_decay", scope="learning_rate",
              conf_func=lambda conf: tf.keras.optimizers.schedules.ExponentialDecay(**conf))(None)

_MODE_RESUME = "resume"
_MODE_NEW = "new"
_MODE_RESUME_COPY = "resume-copy"

nodes = []
name_to_nodes = dict()

def layer_from_config(layer_conf, model_conf, data_conf):
    """
    Get the corresponding keras layer from configurations
    :param layer_conf: The layer configuration
    :param model_conf: The global model configuration, sometimes it is used to generate some
    special layer like "output-classification" and "output-segmentation" layer
    :param data_conf: The dataset configuration, for generating special layers
    :return: A keras layer
    """
    # context = {"class_count": data_conf["class_count"]}
    return object_from_conf(layer_conf, scope="layer", context=None)


def optimizer_from_config(learning_rate, optimizer_conf):
    """
    Get the optimizer from configuration
    :param learning_rate: The learning rate, might be a scalar or a learning rate schedule
    :param optimizer_conf: The optimizer configuration
    :return: An corresponding optimizer
    """
    context = {"learning_rate": learning_rate}
    return object_from_conf(optimizer_conf, scope="optimizer", context=context)


def learning_rate_from_config(learning_rate_conf):
    """
    Get the learning rate scheduler based on configuration
    :param learning_rate_conf: The learning rate configuration
    :return: A learning rate scheduler
    """
    return object_from_conf(learning_rate_conf, scope="learning_rate")

class GraphNode:
    """A class representing a computation node in the graph """
    def __init__(self, param=None):
        """
        :param param: the origin of parameters of this node of form [(str ,int32), (str, int32), ...],
        str denotes the name of a dependency node, int32 denotes the parameter comes from which output of
        that node
        """
        self.param = param or []
        self._value = None
    
    def set_param(self, param):
        """
        Set the parameter
        """
        self.param = param
    
    def value(self):
        """
        Get the value of this node, if it is None, this means the output of this node has not been 
        computed yet, so it requires all the parameters needed from corresponding dependency nodes
        """
        if self._value is not None:
            return self._value
        
        # Compute the dependencies
        inputs = []
        for name, idx in self.param:
            if not name == 'None':
                output = name_to_nodes[name].value()[idx]
                inputs.append(output)
            else:
                inputs.append(None)
        self._value = self._compute(inputs)
        return self._value

    def _compute(self, inputs):
        """
        It determines how the value should be computed, all subclassesof this class should override this 
        method 
        """
        assert False, "The \"compute\" method of the GraphNode should not be called directly"
        return 0

class InputGraphNode(GraphNode):
    """The input node of the graph"""
    def __init__(self, input):
        super(InputGraphNode, self).__init__([]) # No dependency
        self._inputs = input
    
    def _compute(self, inputs):
        return [self._inputs]

class IntermediateLayerGraphNode(GraphNode):
    """The normal graph node """
    def __init__(self, layer, param=None):
        super(IntermediateLayerGraphNode, self).__init__(param)
        self._layer=layer

    def _compute(self, inputs):
        logger.log(f"Computing node for layer {self._layer}")
        output = self._layer(inputs)
        if not isinstance(output, (list, tuple)):
            output = [output]
        return output
class OutputGraphNode(GraphNode):
    """experimental output node"""
    def __init__(self, param=None):
        super(OutputGraphNode, self).__init__(param)

    def _compute(self, inputs):
        return inputs[0]
"""
class OutputGraphNode(IntermediateLayerGraphNode):
    
    def __init__(self, param=None):
        super(OutputGraphNode, self).__init__(layer=tf.keras.layers.Lambda(tf.identity, name="Output"), param=param)
"""
def net_from_config(model_conf, data_conf):
    """
    Generate a keras network from configuration dict
    :param model_conf: The global model configuration dictionary
    :param data_conf: The configuration of the dataset, it might use to initialize some layer like
    "output-classification"
    :param train_dataset: The train dataset, used to add input layer based on shape
    :return: A keras net
    """
    # Get network conf
    net_conf = model_conf["net"]

    # Input layer
    transform_confs = model_conf["dataset"].get("train_transforms", [])
    # Get the shape of the dataset, first check whether we have clip-feature layer in the dataset, if not, we
    # use the feature size in the dataset configuration
    feature_size = None
    """
    for transform_conf in transform_confs[::-1]:
        if type(transform_conf) is dict and transform_conf.get("name") == "clip-feature":
            feature_size = transform_conf["c"]
            logger.log("Get feature_size={} from model configuration".format(feature_size))
    """
    if feature_size is None:
        feature_size = data_conf.get("feature_size")
        logger.log(
            "Get feature_size={} from dataset configuration".format(feature_size))
    assert feature_size is not None, "Cannot determine the feature_size"
    # Get the point size, if possible
    point_count = data_conf.get("point_count")
    """
    for transform_conf in transform_confs[::-1]:
        if type(transform_conf) is dict and transform_conf.get("name") == "sampling":
            point_count = None
            logger.log("Ignore point_count since we have transform sampling from dataset")
    """
    # input_layer = tf.keras.layers.InputLayer(input_shape=(point_count, feature_size))

    # Extend feature layer
    if "extend_feature" in net_conf:
        logger.log(
            "\"extend_feature\" is deprecated, use \"input-feature-extend\" layer instead", color="yellow")
    inputs = tf.keras.Input(shape=(point_count, feature_size), batch_size=16)
    if net_conf["structure"] == "sequence":
        x = inputs  # Input layer

        for layer_conf in net_conf["layers"]:
            logger.log(f"In constructing: {layer_conf}")
            layer = layer_from_config(layer_conf, model_conf, data_conf)
            logger.log(f"Input={x}")
            x = layer(x)
            logger.log(f"Output={x}")

        outputs = x
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    elif net_conf["structure"] == "graph":
        layer_confs = net_conf["layers"]
        graph_confs = net_conf["graph"]
        # Generate all the intermediate nodes and use labels to map them
         
        for conf in layer_confs:
            # Use label to denote the layer
            node_name = conf.get("label", None)
            node = IntermediateLayerGraphNode(layer_from_config(conf, model_conf, data_conf))
            nodes.append(node)
            if node_name is not None:
                assert node_name not in name_to_nodes, f"Layer name \"{node_name}\" conflict, check your labels"
                name_to_nodes[node_name] = node
        
        # Create the input graph node and output graph node
        input_node = InputGraphNode(input=inputs)
        output_node = OutputGraphNode()
        assert "input" not in name_to_nodes and "output" not in name_to_nodes, \
            f"Cannot name label of a layer to \"input\" or \"output\", check your layer labels"
        name_to_nodes["input"] = input_node
        name_to_nodes["output"] = output_node
        # Create the graph
        for conf in graph_confs:
            node_name = conf.get("label", None)
            param = conf.get("param", [])
            name_to_nodes[node_name].set_param(param)
        model = tf.keras.Model(inputs=inputs, outputs=output_node.value())
        return model
    else:
        assert False, "\"{}\" is currently not supported".format(
            net_conf["structure"])


class ModelRunner:
    """
    A class to run a specified model on a specified dataset
    """

    def __init__(self, model_conf, data_conf, name, save_root_dir, train_dataset, test_dataset, mode=None):
        """
        Initialize a model runner
        :param model_conf: The pyconf for model
        :param data_conf: The pyconf for dataset
        :param name: The name for model
        :param save_root_dir: The root for saving. Normally it is the root directory where all the models of a specified
        dataset should be saved. Like something "path/ModelNet40-2048". Note that it is not the "root directory of the
        model", such as "path/ModelNet40-2048/PointCNN-X3-L4".
        :param train_dataset: The dataset to train the model
        :param test_dataset: The dataset to test the model
        :param mode: The mode indicates the strategy of whether to reuse the previous training process and continue
        training. Currently we support 3 types of modes:
            1. "new" or None: Do not use the previous result and start from beginning.
            2. "resume": Reuse previous result
            3. "resume-copy": Reuse previous result but make an exact copy.
        Both the "resume" and "resume-copy" will try to find the last result with the same "name" in the "save_root_dir"
        and reuse it. "resume" mode will continue training in the previous directory while "resume-copy" will try to
        create a new one and maintain the original one untouched. Default is None.
        """
        self.model_conf = model_conf
        self.data_conf = data_conf
        self.name = name
        self.save_root_dir = save_root_dir
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_size = len(train_dataset)
        self.point_size = 8192
        self._mode = mode or "new"
        assert self._mode in [_MODE_NEW, _MODE_RESUME, _MODE_RESUME_COPY], \
            "Unrecognized mode=\"{}\". Currently support \"new\", \"resume\" and \"resume-copy\""
    def rotate_point_cloud_z(self, batch_data):
        """Randomly rotate the point clouts to augment the dataset"""
        rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
        for k in range(batch_data.shape[0]):
            rotation_angle = np.random.uniform() * 2 * np.pi
            cosval = np.cos(rotation_angle)
            sinval = np.sin(rotation_angle)
            rotation_matrix = np.array([[cosval, -sinval, 0],
                                        [sinval, cosval, 0],
                                        [0, 0, 1]])
            shape_pc = batch_data[k, ...]
            rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)),rotation_matrix)
        return rotated_data           

    def get_batch_dropout(self, dataset, idxs, start_idx, end_idx):
        bsize = end_idx - start_idx
        batch_data = np.zeros((bsize, self.point_size, 3))
        batch_label = np.zeros((bsize, self.point_size), dtype=np.int32)
        batch_smpw = np.zeros((bsize, self.point_size), dtype=np.float32)
        for i in range(bsize):
            ps, seg, smpw = dataset[idxs[i + start_idx]]
            batch_data[i, ...] = ps
            batch_label[i, :] = seg
            batch_smpw[i, :] = smpw
            
            dropout_ratio = np.random.random() * 0.875 # 0-0.875
            drop_idx = np.where(np.random.random((ps.shape[0])) <= dropout_ratio)[0]
       
            batch_data[i, drop_idx, :] = batch_data[i, 0, :]
            batch_label[i, drop_idx] = batch_label[i, 0]
            batch_smpw[i, drop_idx] *= 0
        return batch_data, batch_label, batch_smpw

    def train(self):
        control_conf = self.model_conf["control"]

        # Transform the dataset is the dataset is classification dataset and
        # the model_conf's last output layer is output-conditional-segmentation
        train_dataset = test_dataset = None
        if self.data_conf["task"] == "classification" and \
                self.model_conf["net"]["layers"][-1]["name"] == "output-conditional-segmentation":
            layer_conf = self.model_conf["net"]["layers"][-1]
            assert "output_size" in layer_conf, "The dataset is classification dataset " \
                                                "while the model configuration is segmentation. " \
                                                "Cannot find \"output_size\" to transform the " \
                                                "classification dataset to segmentation task"
            seg_output_size = layer_conf["output_size"]
            # Transform function convert the label with (B, 1) to (B, N) where N is the last layer's point output size
            transform_func = (lambda points, label: (points, tf.tile(label, (1, seg_output_size))))
            train_dataset = self.train_dataset.map(transform_func)
            test_dataset = self.test_dataset
            logger.log("Convert classification to segmentation task with output_size={}".format(seg_output_size))
        else:
            train_dataset, test_dataset = self.train_dataset, self.test_dataset

        # Get the suffix of the directory by iterating the root directory and check which suffix has not been
        # created
        suffix = 0

        # The lambda tries to get the save directory based on the suffix
        def save_dir(suffix_=None):
            suffix_ = suffix_ if suffix_ is not None else suffix
            return path.join(self.save_root_dir, self.name + ("-" + str(suffix_) if suffix_ > 0 else ""))

        # Find the last one that the name has not been occupied
        while path.exists(save_dir()):
            suffix += 1

        # Check mode and create directory
        if self._mode == _MODE_NEW or suffix == 0:
            # We will enter here if the mode is "new" or we cannot find the previous model (suffix == 0)
            if self._mode != _MODE_NEW:
                logger.log("Unable to find the model with name \"{}\" to resume. Try to create new one", color="yellow")
            makedirs(save_dir(), exist_ok=False)
        elif self._mode == _MODE_RESUME:
            # Since we reuse the last one, we decrease it by one and do not need to create directory
            suffix -= 1
        elif self._mode == _MODE_RESUME_COPY:
            # Copy the reused one to the new one
            shutil.copytree(save_dir(suffix - 1), save_dir())
        logger.log("Save in directory: \"{}\"".format(save_dir()), color="blue")

        # Try get the infos and previous train step from the info.txt
        infos = dict()
        infos_file_path = path.join(save_dir(), "info.txt")
        if path.exists(infos_file_path) and path.isfile(infos_file_path):
            with open(path.join(save_dir(), "info.txt")) as f:
                pattern = re.compile(r"(\S+)[\s]?=[\s]*(\S+)")
                for line in f:
                    m = re.match(pattern, line.strip())
                    if m:
                        infos[m.group(1)] = eval(m.group(2))
            logger.log("Info loads, info: {}".format(logger.format(infos)), color="blue")
        else:
            logger.log("Do not find info, maybe it is a newly created model", color="blue")

        # Get the step offset
        # Because we store the "have trained" step, so it needs to increase by 1
        step_offset = infos.get("step", -1) + 1
        logger.log("Set step offset to {}".format(step_offset), color="blue")

        # Get the network
        logger.log("Creating network, train_dataset={}, test_dataset={}".format(self.train_dataset, self.test_dataset))
        net = net_from_config(self.model_conf, self.data_conf)
        # Get the learning_rate and optimizer
        logger.log("Creating learning rate schedule")
        lr_schedule = learning_rate_from_config(control_conf["learning_rate"])
        logger.log("Creating optimizer")
        optimizer = optimizer_from_config(lr_schedule, control_conf["optimizer"])

        # Get the loss
        # >>>>>
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(name="Loss")
        # >>>>>

        # Get the metrics
        # We add a logits loss in the metrics since the total loss will have regularization term
        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True, name="logits_loss")
        ]

        # Get the batch size
        batch_size = control_conf["batch_size"]

        # >>>>>
        net.summary()
        train_epoch = control_conf.get("train_epoch", None)
        iter_in_epoch = self.train_size // batch_size
        pdb.set_trace() 
        for epoch in range(train_epoch):
            logger.log("Start of epoch {}".format(epoch))
            train_idx = np.arange(0, self.train_size)
            np.random.shuffle(train_idx)
            for pos in range(iter_in_epoch):
                # prepare batch data
                start_idx = pos * batch_size
                end_idx = (pos + 1) * batch_size
                batch_data, batch_label, batch_smpw = self.get_batch_dropout(train_dataset, train_idx, start_idx, end_idx)
                aug_data = self.rotate_point_cloud_z(batch_data)
                # compute result of the net
                with tf.GradientTape() as tape:
                    logits = net(aug_data)
                    loss_value = loss_fn(batch_label, logits, sample_weight=batch_smpw)
                grads = tape.gradient(loss_value, net.trainable_weights)
                optimizer.apply_gradients(zip(grads, net.trainable_weights))
                logger.log("Training loss (for one batch) at step {}: {}".format(pos, float(loss_value)))
        # >>>>>

        # Get the total step for training
        if "train_epoch" in control_conf:
            train_step = int(math.ceil(control_conf["train_epoch"] * self.data_conf["train"]["size"] / batch_size))
        elif "train_step" in control_conf:
            train_step = control_conf["train_step"]
        else:
            assert False, "Do not set the \"train_step\" or \"train_epoch\" in model configuraiton"

        # Get the validation step
        validation_step = control_conf.get("validation_step", None)
        tensorboard_sync_step = control_conf.get("tensorboard_sync_step", None) or validation_step or 100

        logger.log("Training conf: batch_size={}, train_step={}, validation_step={}, "
                   "tensorboard_sync_step={}".format(batch_size, train_step, validation_step, tensorboard_sync_step))

        # Get the callback
        # Initialize the tensorboard callback, and set the step_offset to make the tensorboard
        # output the correct step
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=save_dir(), update_freq=tensorboard_sync_step)
        if hasattr(tensorboard_callback, "_total_batches_seen"):
            tensorboard_callback._total_batches_seen = step_offset
        else:
            logger.log("Unable to set the step offset to the tensorboard, the scalar output may be a messy",
                       color="yellow")

        model_callback = ModelCallback(train_step, validation_step, train_dataset, test_dataset,
                                       batch_size, save_dir(), infos=infos, step_offset=step_offset)

        logger.log("Compile network, loss={}, metrics={}".format(loss, metrics))
        net.compile(optimizer, loss=loss, metrics=metrics)

        logger.log("Summary of the network:")
        net.summary(line_length=240, print_fn=lambda x: logger.log(x, prefix=False))

        logger.log("Begin training")
        net.fit(
            train_dataset,
            verbose=0,
            steps_per_epoch=train_step,
            callbacks=[tensorboard_callback, model_callback],
            shuffle=False  # We do the shuffle ourself
        )
