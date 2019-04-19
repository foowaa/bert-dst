import re
import os
from typing import Dict, Any
import numpy as np
import copy
import json
import tensorflow as tf

import pdb

from components.component import Component
from components.nn_model import NNModel
from components.log import get_logger
from models.nbt1 import NeuralBeliefTracker

log = get_logger(__name__)

def base_param():
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    return {
        'max_utter_length': 20,
        'start_learning_rate': 0.001,
        'end_learning_rate': 0.0001,
        'tf_configs': tf_config
    }

class SelfGoalOrientedBot(NNModel):
    """
    The dialogue bot is based on Neural Belief Tracker.

    Args:
        tokenizer: 分词模型
        ontology_fn: 场景配置文件
        network_parameters: NBT网络参数
        save_path: 模型保存路径
        load_path: 模型加载路径
    Funs:
        train_on_batch
        test_on_batch
        __call__
    """
    def __init__(self,
                 tokenizer: Component,
                 ontology_fn: str,
                 network_parameters: Dict[str, Any] = base_param(),
                 save_path: str = 'ckpt/nbt',
                 load_path: str = 'ckpt/nbt',
                 gpuid: str = '1',
                 word_vocab: Component = None,
                 bow_embedder: Component = None,
                 pinyin_embedder: Component = None,
                 char_embedder: Component = None,
                 word_embedder: Component = None,
                 **kwargs):
        super().__init__(load_path=load_path, save_path=save_path, **kwargs)

        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        self.tokenizer = tokenizer
        self.word_vocab = word_vocab

        self.max_utter_length = network_parameters['max_utter_length']

        self.bow_embedder = bow_embedder
        self.pinyin_embedder = pinyin_embedder
        self.char_embedder = char_embedder
        self.word_embedder = word_embedder

        # self.pinyin_embedding_dim = pinyin_embedder.dim
        # self.char_embedding_dim = char_embedder.dim
        self.word_embedding_dim = word_embedder.dim
        self.word_embedder.vectors[None] = np.zeros((self.word_embedding_dim), dtype="float32")

        log.info("[loading ontology from {}]".format(ontology_fn))
        self.ontologys = json.load(open(ontology_fn, encoding='utf-8'))
        self.slots = self.ontologys['slots']
        self.intentions = self.ontologys['意图']

        network_parameters['load_path'] = load_path
        network_parameters['save_path'] = save_path
        self._init_network(network_parameters)


    def _get_network_params(self, params, slot, values, use_softmax):
        value_num = len(values)
        if use_softmax==True:
            value_num += 1

        slot_vector = np.zeros((self.word_embedding_dim), dtype="float32")
        value_vector = np.zeros((value_num, self.word_embedding_dim), dtype="float32")

        slot_vector = self.word_embedder([slot], mean=True)[0]
        for value_idx, value in enumerate(values):
            idx = value_idx
            if use_softmax==True:
                idx += 1
            if len(self.word_embedder([value], mean=True)[0]) != self.word_embedding_dim:
                pdb.set_trace()
            value_vector[idx, :] = self.word_embedder([value], mean=True)[0]

        params['slot_vector'] = slot_vector
        params['value_vector'] = value_vector
        params['class_weights'] = [ 1. for _ in range(value_vector.shape[0]) ]
        # if slot=='身份':
        #     params['class_weights'] = [1.] + [ 10. for _ in range(value_vector.shape[0]-1) ]
        params['use_softmax'] = use_softmax

        params['load_path'] = os.path.join(params['load_path'], slot)
        params['save_path'] = os.path.join(params['save_path'], slot)

        return params

    def _init_network(self, params):
        self.models = {}
        for slot in self.slots.keys():
            this_params = self._get_network_params(copy.deepcopy(params), slot, self.slots[slot], True)
            self.models[slot] = NeuralBeliefTracker(**this_params)

        this_params = self._get_network_params(copy.deepcopy(params), '意图', self.intentions, False)
        self.models['意图'] = NeuralBeliefTracker(**this_params)


    def _encode_input(self, utter_text, sys_text, sys_q, sys_s, sys_v):
        utter_text_tokens = self.tokenizer(utter_text)
        for i, di in enumerate(utter_text_tokens):
            if len(di)>self.max_utter_length:
                utter_text_tokens[i] = di[:10] + di[-11:]
            elif len(di)<self.max_utter_length:
                utter_text_tokens[i] = di + [None for _ in range(self.max_utter_length-len(di))]
        sys_text_tokens = self.tokenizer(sys_text)
        sys_q_tokens = self.tokenizer(sys_q)
        sys_s_tokens = self.tokenizer(sys_s)
        sys_v_tokens = self.tokenizer(sys_v)

        utter_word_emb = np.array(self.word_embedder(utter_text_tokens, mean=False))
        sys_word_emb = np.array(self.word_embedder(sys_text_tokens, mean=True))
        sys_q_word_emb = np.array(self.word_embedder(sys_q_tokens, mean=True))
        sys_s_word_emb = np.array(self.word_embedder(sys_s_tokens, mean=True))
        sys_v_word_emb = np.array(self.word_embedder(sys_v_tokens, mean=True))

        return utter_word_emb, sys_word_emb, sys_q_word_emb, sys_s_word_emb, sys_v_word_emb

    def _encode_label(self, label_intent, label_slot):
        label_dic = { slot_k: [0. for i in range(len(slot_v)+1)] for slot_k,slot_v in self.slots.items()}
        for slot_k in label_dic.keys():
            label_dic[slot_k][0] = 1
        label_dic['意图'] = [ 0. for i in range(len(self.intentions)) ]
        for slot_k, slot_v in label_slot:
            if slot_v in self.slots[slot_k]:
                label_dic[slot_k][self.slots[slot_k].index(slot_v) + 1] = 1.
                label_dic[slot_k][0] = 0
        for label_i in label_intent:
            if label_i in self.intentions:
                label_dic['意图'][self.intentions.index(label_i)] = 1.
        return label_dic

    def _decode_label(self, label_type, label_prob, min_value=0.5):
        if label_type=='意图':
            if label_prob[0]>label_prob[1]>0.3:
                return ['肯定']+[self.intentions[i+2] for i,pi in enumerate(label_prob[2:]) if pi>min_value]
            elif label_prob[1]>label_prob[0]>0.3:
                return ['否定']+[self.intentions[i+2] for i,pi in enumerate(label_prob[2:]) if pi>min_value]
            else:
                return [self.intentions[i] for i,pi in enumerate(label_prob) if pi>min_value]
        else:
            idx = list(label_prob).index(max(label_prob))
            return [self.slots[label_type][idx-1] if idx>0 else None ]

    def train_on_batch(self, datas):
        # utter_text, sys_text, sys_q, sys_s, sys_v, y_dic['intent'], y_dic['slot']
        utter_text, sys_text, sys_q, sys_s, sys_v, y_intent, y_slot = datas
        label_dic = { slot_k: [] for slot_k in self.slots.keys()}
        label_dic['意图'] = []
        for label_intent, label_slot in zip(y_intent, y_slot):
            label_all = self._encode_label(label_intent, label_slot)
            for k, v in label_all.items():
                label_dic[k].append(v)

        utter_word_emb, sys_word_emb, sys_q_word_emb, sys_s_word_emb, sys_v_word_emb = \
            self._encode_input(utter_text, sys_text, sys_q, sys_s, sys_v)

        result = {}
        for slot_k,label_v in label_dic.items():
            loss_value, y, prediction, accuracy = self.models[slot_k].train_on_batch(
                utter_word_emb,
                sys_s_word_emb,
                sys_v_word_emb,
                np.array(label_v)
            )
            result[slot_k] = (loss_value, y, prediction, accuracy)

        return result

    def __call__(self, datas):
        if len(datas) == 7:
            utter_text, sys_text, sys_q, sys_s, sys_v, _, _ = datas
        else:
            utter_text, sys_text, sys_q, sys_s, sys_v = datas

        utter_word_emb, sys_word_emb, sys_q_word_emb, sys_s_word_emb, sys_v_word_emb = \
            self._encode_input(utter_text, sys_text, sys_q, sys_s, sys_v)

        label_dic = {}
        for slot_k, a_model in self.models.items():
            probs = a_model(
                utter_word_emb,
                sys_s_word_emb,
                sys_v_word_emb,
                prob=True
            )
            label_dic[slot_k] = probs

        predict_result = [ {} for _ in range(len(datas[0]))]
        for slot_k, probs in label_dic.items():
            for i in range(len(probs)):
                # pdb.set_trace()
                predict_result[i][slot_k] = self._decode_label(slot_k, probs[i], min_value=0.5)

        return predict_result

    def test_on_batch(self, datas, print_y_pred=False):
        utter_text, sys_text, sys_q, sys_s, sys_v, y_intent, y_slot = datas
        label_dic = { slot_k: [] for slot_k in self.slots.keys()}
        label_dic['意图'] = []
        for label_intent, label_slot in zip(y_intent, y_slot):
            label_all = self._encode_label(label_intent, label_slot)
            for k, v in label_all.items():
                label_dic[k].append(v)

        utter_word_emb, sys_word_emb, sys_q_word_emb, sys_s_word_emb, sys_v_word_emb = \
            self._encode_input(utter_text, sys_text, sys_q, sys_s, sys_v)

        result = {}
        for slot_k,label_v in label_dic.items():
            loss_value, y, prediction, y_label, true_predictions, accuracy = self.models[slot_k].test_on_batch(
                utter_word_emb,
                sys_s_word_emb,
                sys_v_word_emb,
                np.array(label_v)
            )
            result[slot_k] = (loss_value, y, prediction, y_label, true_predictions, accuracy)

        return result

    def process_event(self, *args, **kwargs):
        pass
        # self.network.process_event(*args, **kwargs)

    def save(self, slot):
        """Save the parameters of the model to a file."""
        self.models[slot].save()

    def load(self):
        for s in self.models.keys():
            self.models[s].load()
