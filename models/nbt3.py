import copy
import os
from typing import Dict, List, Any
import collections
import json
import numpy as np
from pathlib import Path
import tensorflow as tf

from components.tf_model import TFModel
from components.log import get_logger

log = get_logger(__name__)

def calculate_dot_sim(x, y):
    sim = tf.multiply(x, y)  # element-wise multiplication
    sim = tf.reduce_sum(sim, axis=1, keepdims=False)
    return sim

class NeuralBeliefTracker(TFModel):
    """
    NBT Model Paper: Neural Belief Tracker: Data-Driven Dialogue State Tracking.

    Args:
        use_softmax: if True, the last layer is softmax
        filters_num: the filters num of utter rep by CNN
        filters_sizes: the filters size of utter rep by CNN
        hidden_dim: the hidden dim of the seconde last layer
        max_utter_length: the max length of utter text
    """

    GRAPH_PARAMS = ["use_softmax", "filters_num", "filter_sizes", "hidden_dim",
                    "max_utter_length"]

    def __init__(self,
                 slot_vector: List[float],                       # shape（1，word_dim）
                 value_vector: List[List[float]],                # shape（label_num,word_dim）label_count including none
                 class_weights: List[float],

                 use_softmax: bool,
                 filters_num: float=100,            # filter num for cnn
                 filter_sizes: List[int]=[1,2,3,4],
                 hidden_dim: int=50,
                 max_utter_length: float=20,

                 start_learning_rate:     float=0.001,
                 end_learning_rate: float = 0.0001,
                 decay_steps:       int = 1000,
                 decay_power:       float = 1.,
                 optimizer:         str = 'AdamOptimizer',
                 tf_configs:        tf.ConfigProto = tf.ConfigProto(),
                 **kwargs):

        self.opt = {
            'use_softmax': use_softmax,
            'filters_num': filters_num,
            'filter_sizes': filter_sizes,
            'hidden_dim': hidden_dim,
            'max_utter_length': max_utter_length
        }

        self.save_path = Path(kwargs['save_path'])
        self.load_path = Path(kwargs['load_path'])

        self.slot_vector = slot_vector
        self.value_vector = value_vector
        self.class_weights = class_weights
        self.use_softmax = use_softmax
        self.filters_num = filters_num
        self.filter_sizes = filter_sizes
        self.hidden_dim = hidden_dim
        self.max_utter_len = max_utter_length

        self.label_num = value_vector.shape[0]
        self.word_dim = value_vector.shape[-1]

        self.start_learning_rate = start_learning_rate
        self.end_learning_rate = end_learning_rate
        self.decay_steps = decay_steps
        self.decay_power = decay_power
        self._optimizer = None
        if hasattr(tf.train, optimizer):
            self._optimizer = getattr(tf.train, optimizer)
        if not issubclass(self._optimizer, tf.train.Optimizer):
            raise NameError("`optimizer` parameter should be a name of"
                              " tf.train.Optimizer subclass")

        self.x_utter = tf.placeholder(tf.float32, [None, self.max_utter_len, self.word_dim], name='x_utter')
        self.x_sys_slot = tf.placeholder(tf.float32, shape=(None, self.word_dim), name='x_sys_slot')
        self.x_sys_value = tf.placeholder(tf.float32, shape=(None, self.word_dim), name='x_sys_value')
        self.y_label = tf.placeholder(tf.float32, [None, self.label_num], name='label')
        self.keep_prob = tf.placeholder("float")
        self.learning_rate = tf.placeholder(tf.float32,)

        self._build_graph()

        self.sess = tf.Session(config=tf_configs)
        self.sess.run(tf.global_variables_initializer())
        self.global_step = 0

    def __call__(self, x_utter, x_sys_slot, x_sys_value, kp=1., prob=False):
        feed_dict={
            self.x_utter: x_utter,
            self.x_sys_slot: x_sys_slot,
            self.x_sys_value: x_sys_value,
            self.keep_prob: kp
        }
        prediction, y = self.sess.run([self.predictions, self.y],feed_dict=feed_dict)
        if prob:
            return y
        else:
            return prediction

    def train_on_batch(self, x_utter, x_sys_slot, x_sys_value, y_label, kp=1.):
        feed_dict={
            self.x_utter: x_utter,
            self.x_sys_slot: x_sys_slot,
            self.x_sys_value: x_sys_value,
            self.y_label: y_label,
            self.keep_prob: kp,
            self.learning_rate: self.get_learning_rate()
        }
        _, loss_value, y, prediction, accuracy = \
            self.sess.run([self._train_op, self.loss, self.y, self.predictions, self.accuracy],feed_dict=feed_dict)
        return loss_value, y, prediction, accuracy

    def test_on_batch(self, x_utter, x_sys_slot, x_sys_value, y_label):
        feed_dict={
            self.x_utter: x_utter,
            self.x_sys_slot: x_sys_slot,
            self.x_sys_value: x_sys_value,
            self.y_label: y_label,
            self.keep_prob: 1.
        }
        loss_value, y, prediction, y_label, true_predictions, accuracy = \
            self.sess.run([self.loss, self.y, self.predictions, self.y_label, self.true_predictions, self.accuracy],feed_dict=feed_dict)
        return loss_value, y, prediction, y_label, true_predictions, accuracy

    def _build_graph(self):
        self._build_model()

        if self.use_softmax:
            self.predictions = tf.cast(tf.argmax(self.y, 1), "float32")
            self.true_predictions = tf.cast(tf.argmax(self.y_label, 1), "float32")
        else:
            self.predictions = tf.cast(tf.round(self.y), "float32")
            self.true_predictions = tf.cast(tf.round(self.y_label), "float32")
        self.correct_prediction = tf.cast(tf.equal(self.predictions, self.true_predictions), "float")
        self.accuracy = tf.reduce_mean(self.correct_prediction)

        self._train_op = self.get_train_op(self.loss,
                                           learning_rate=self.learning_rate,
                                           optimizer=self._optimizer,
                                           clip_norm=2.)

    def _utterance_encoding(self):
        with tf.variable_scope("utterance_encoding"):
            encoded_utterances = []
            for filter_size in self.filter_sizes:
                filter_shape = [filter_size, self.word_dim, self.filters_num]
                weight_filter = tf.get_variable(name='w_filter'+str(filter_size), initializer=tf.truncated_normal(filter_shape))
                bias_filter = tf.get_variable(name='b_filter'+str(filter_size), initializer=tf.zeros([self.filters_num]))

                # input [batch_size, utter_max_len, word_dim]
                # conv [batch_size, n_conv, filters_num]
                # h [batch_size, n_conv, filters_num]
                # pooled [batch_size, filters_num]
                conv = tf.nn.conv1d(value=self.x_utter, filters=weight_filter, stride=1, padding="VALID")
                h = tf.nn.relu(tf.nn.bias_add(conv, bias_filter), name="relu")
                pool_shape = self.max_utter_len-filter_size+1
                pooled = tf.layers.max_pooling1d(inputs=h, pool_size=pool_shape, strides=1, padding='VALID')
                pooled = tf.reshape(pooled, [-1, self.filters_num])

                encoded_utterances.append(pooled)
            encoded_utterance = tf.reduce_mean(tf.transpose(tf.stack(encoded_utterances), [1, 0, 2]), 1)
        return encoded_utterance        # r = [batch_size, filters_num]

    def _candi_encoding(self):
        with tf.variable_scope("candi_encoding"):
            # input [label_num, word_dim]
            self.c_slot = tf.constant(self.slot_vector, name="c_slot", dtype='float32')
            self.c_values = tf.constant(self.value_vector, name="c_values", dtype='float32')
            c = self.c_values
            if not self.use_softmax:
                c = c + tf.reshape(self.c_slot, [1, self.word_dim])
            encoded_candi = tf.layers.dense(c, self.filters_num, activation=tf.nn.sigmoid)
        return encoded_candi       # c = [label_num, filters_num]

    def _semantic_decoding(self, utter_encoded, candi_encoded):
        with tf.variable_scope("semantic_decoding"):
            list_utter_candi_sim = []
            for i in range(self.label_num):
                temp_sim = tf.multiply(utter_encoded, candi_encoded[i, :])     # element-wise multiplication
                list_utter_candi_sim.append(temp_sim)
            decoded_utterance = tf.transpose(tf.stack(list_utter_candi_sim), [1, 0, 2])
        return decoded_utterance        # d = [batch_size, label_num, filters_num]

    def _gating_mechanism_slot(self):
        with tf.variable_scope("gating_mechanism_slot"):
            return calculate_dot_sim(self.c_slot, self.x_sys_slot)

    def _gating_mechanism_value(self):
        with tf.variable_scope("gating_mechanism_value"):
            list_sys_c_values_sim = []
            for i in range(self.label_num):
                temp_sim = calculate_dot_sim(self.c_values[i, :], self.x_sys_value)
                list_sys_c_values_sim.append(temp_sim)
            gate_value = tf.transpose(tf.stack(list_sys_c_values_sim), [1, 0])
        return gate_value               # [batch_size, label_num]

    def _context_modelling_slot(self, utter_encoded, gate_slot):
        with tf.variable_scope("context_modelling_slot"):
            gated_utterance = tf.multiply(tf.reshape(gate_slot, [-1, 1]), utter_encoded)
            gated_utterance = [ gated_utterance for _ in range(self.label_num)]
            gated_utterance = tf.transpose(tf.stack(gated_utterance), [1, 0, 2])
        return gated_utterance

    def _context_modelling_value(self, utter_encoded, gate_value):
        with tf.variable_scope("context_modelling_value"):
            list_gated_utterance = []
            for i in range(self.label_num):
                temp_gated_utterance = tf.multiply(tf.reshape(gate_value[:, i], [-1, 1]), utter_encoded)    # element-wise multiplication
                list_gated_utterance.append(temp_gated_utterance)
            gated_utterance = tf.transpose(tf.stack(list_gated_utterance), [1, 0, 2])
        return gated_utterance          # [batch_size, n_intent, n_filter]

    def _build_model(self):

        utter_encoded = self._utterance_encoding()          # [batch_size, filters_num]
        candi_encoded = self._candi_encoding()              # [label_num, filters_num]

        gate_slot = self._gating_mechanism_slot()           # [batch_size]
        gate_value = self._gating_mechanism_value()         # [batch_size, label_num]

        gate_slot_value = tf.multiply(tf.reshape(gate_slot, [-1, 1]), gate_value) # [batch_size, label_num]

        semantic_decoded_r_c = self._semantic_decoding(utter_encoded, candi_encoded)            # [batch_size, label_num, filters_num]
        context_decoded_m_value = self._context_modelling_value(utter_encoded, gate_slot_value)      # [batch_size, label_num, filters_num]

        hidden_out_d = tf.layers.dense(
            tf.reshape(semantic_decoded_r_c, [-1, self.filters_num]),
            self.hidden_dim,
            activation=tf.nn.sigmoid)           # [batch_size * label_num, hidden_dim]

        hidden_out_m_slot = tf.layers.dense(
            tf.reshape(context_decoded_m_value, [-1, self.filters_num]),
            self.hidden_dim,
            activation=tf.nn.sigmoid)           # [batch_size * label_num, hidden_dim]
        # hidden_out = tf.nn.dropout(hidden_out, self.keep_prob)

        logits = tf.layers.dense(hidden_out_d + hidden_out_m_slot, 1)
        self.logits = tf.reshape(logits, [-1, self.label_num])

        if self.use_softmax:
            self.y = tf.nn.softmax(self.logits)
            # self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_label))

            class_weights = tf.constant([self.class_weights])
            weights = tf.reduce_sum(class_weights * self.y_label, axis=1)
            unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_label)
            self.loss = tf.reduce_mean(unweighted_losses * weights)
        else:
            self.y = tf.nn.sigmoid(self.logits)
            self.loss = tf.reduce_mean(
                tf.where(
                    tf.greater(self.y_label, self.y),
                    tf.square(self.y - self.y_label)*5,
                    tf.square(self.y - self.y_label)*1
                )
            )
            # self.loss = tf.reduce_mean(tf.square(self.y - self.y_label))


    def get_learning_rate(self):
        # polynomial decay
        global_step = min(self.global_step, self.decay_steps)
        decayed_learning_rate = \
            (self.start_learning_rate - self.end_learning_rate) *\
            (1 - global_step / self.decay_steps) ** self.decay_power +\
            self.end_learning_rate
        return decayed_learning_rate

    def load(self, *args, **kwargs):
        self.load_params()
        super().load(*args, **kwargs)

    def save(self, *args, **kwargs):
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        super().save(*args, **kwargs)
        self.save_params()

    def save_params(self):
        path = str(self.save_path.with_suffix('.json').resolve())
        log.info('[saving parameters to {}]'.format(path))
        with open(path, 'w', encoding='utf8') as fp:
            json.dump(self.opt, fp)

    def load_params(self):
        path = str(self.load_path.with_suffix('.json').resolve())
        log.info('[loading parameters from {}]'.format(path))
        with open(path, 'r', encoding='utf8') as fp:
            params = json.load(fp)
        for p in self.GRAPH_PARAMS:
            if self.opt.get(p) != params.get(p):
                raise NameError("`{}` parameter must be equal to saved model "
                                  "parameter value `{}`, but is equal to `{}`"
                                  .format(p, params.get(p), self.opt.get(p)))

    def process_event(self, event_name, data):
        if event_name == "after_epoch":
            log.info("Updating global step, learning rate = {:.6f}."
                     .format(self.get_learning_rate()))
            self.global_step += 1

    def shutdown(self):
        self.sess.close()
