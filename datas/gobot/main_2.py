'''
催收
'''
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
import os
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
from components.utils_data import jsonify_data
from dst_reader import JRCSDatasetReader
from dialog_iterator import SelfDialogDatasetIterator
from tencent_embedder import TencentEmbedder
from cn_tokenizer import CNSplitTokenizer
from nlu import SelfGoalOrientedBot
from dm  import DialogueManagement
from nlg import NLG

from easydict import EasyDict as edict
import time
import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from pytorch_pretrained_bertt.tokenization import BertTokenizer
from pytorch_pretrained_bertt.modeling import BertConfig, BertForSeqClassificationDialog
from pytorch_pretrained_bertt.optimization import BertAdam, warmup_linear


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

data = JRCSDatasetReader.read('/home/ubuntu/code/QA/v1/datas/gobot', dataset_fn='dataset_result.csv', split_rate=[0.90, 0.10, 0.00])
data_iterator = SelfDialogDatasetIterator(data)
#embedder = TencentEmbedder('/home/ubuntu/models/Tencent_AILab_ChineseEmbedding.txt')
tokenizer = CNSplitTokenizer()
# gobot = SelfGoalOrientedBot(
#     tokenizer=tokenizer,
#     word_embedder=embedder,
#     ontology_fn='configs/gobot/ontology.json',
#     save_path='ckpt/nbt',
#     load_path='ckpt/nbt'
#     )

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id_intent, label_id_slot1, label_id_slot2, label_id_slot3, label_id_slot4, label_id_slot5):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id_intent = label_id_intent
        self.label_id_slot1 = label_id_slot1
        self.label_id_slot2 = label_id_slot2
        self.label_id_slot3 = label_id_slot3
        self.label_id_slot4 = label_id_slot4
        self.label_id_slot5 = label_id_slot5

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

def print_out_sentence(outs, label_list, thresh=0.5):
    label_map = []
    for (i, label) in enumerate(label_list):
        tmp = {}
        for (j,ele) in enumerate(label):
            tmp[j] = ele
        label_map.append(tmp)

    for i in range(len(outs)):
        if i==0:
            out = np_sigmoid(outs[i].cpu().numpy())
            out = np.squeeze(out)
            x = np.where(out > thresh)[0]
            print("intentions:")
            for j in x:
                print(label_map[i][j])
        else:
            e = np.argmax(outs[i].cpu().numpy(), axis=1)[0]
            print("slots")
            print(label_map[i][e])

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = []
    for (i, label) in enumerate(label_list):
        tmp = {}
        for (j,ele) in enumerate(label):
            tmp[ele] = j
        label_map.append(tmp)

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length


        if example.label is not None:
            label_id_intent = []
            label_id_slot1 = []
            label_id_slot2 = []
            label_id_slot3 = []
            label_id_slot4 = []
            label_id_slot5 = []
            for i,e1 in enumerate(example.label):
                if i==0:
                    idx = []
                    for e2 in e1:
                        idx.append(label_map[i][e2])
                        #label_id_intent
                    tmp = [0 for _ in range(18)]
                    for j in range(18):
                        if j in idx:
                            tmp[j] = 1
                    label_id_intent = tmp
                else:
                    eval('label_id_slot'+str(i)).append(label_map[i][e1])

            #label_id = label_map[tuple(example.label)]
            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % (example.guid))
                # logger.info("tokens: %s" % " ".join(
                #         [BertTokenizer.printable_text(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                        "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                #logger.info("label: %s (id = %d)" % (example.label, label_id))

            features.append(
                    InputFeatures(
                            input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id_intent=label_id_intent,
                            label_id_slot1=label_id_slot1,
                            label_id_slot2=label_id_slot2,
                            label_id_slot3=label_id_slot3,
                            label_id_slot4=label_id_slot4,
                            label_id_slot5=label_id_slot5,
                    ))
        else:
            features.append(
                    InputFeatures(
                            input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id_intent=None,
                            label_id_slot1=None,
                            label_id_slot2=None,
                            label_id_slot3=None,
                            label_id_slot4=None,
                            label_id_slot5=None,
                    ))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy_oridinary(out, labels):
    outputs = np.argmax(out, axis=1)
    labels = np.squeeze(labels, axis=1)
    return np.sum(outputs==labels)

def np_sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))
    if derivative:
        return sigm * (1. - sigm)
    return sigm
# 多标签
# https://github.com/kaushaltrivedi/bert-toxic-comments-multilabel/blob/master/toxic-bert-multilabel-classification.ipynb
def accuracy_multilabel(out, labels, thresh=0.5):
    out = np_sigmoid(out)
    return np.mean(((out > thresh) == labels), axis=1).sum()

def copy_optimizer_params_to_model(named_params_model, named_params_optimizer):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the parameters optimized on CPU/RAM back to the model on GPU
    """
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        param_model.data.copy_(param_opti.data)

def set_optimizer_params_grad(named_params_optimizer, named_params_model, test_nan=False):
    """ Utility function for optimize_on_cpu and 16-bits training.
        Copy the gradient of the GPU parameters to the CPU/RAMM copy of the model
    """
    is_nan = False
    for (name_opti, param_opti), (name_model, param_model) in zip(named_params_optimizer, named_params_model):
        if name_opti != name_model:
            logger.error("name_opti != name_model: {} {}".format(name_opti, name_model))
            raise ValueError
        if test_nan and torch.isnan(param_model.grad).sum() > 0:
            is_nan = True
        if param_opti.grad is None:
            param_opti.grad = torch.nn.Parameter(param_opti.data.new().resize_(*param_opti.data.size()))
        param_opti.grad.data.copy_(param_model.grad.data)
    return is_nan

'''
create examples

params d1: x
params d2: y
params set_type: train/dev
'''
def create_examples(d1, d2, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    if d2 is not None:
        for (i, ele) in enumerate(zip(d1,d2)):
            if i == 0:
                continue
            text_a = ele[0]
            label = ele[1]
            guid = "%s-%s" % (set_type, i)

            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label))
    else:
        for (i, ele) in enumerate(d1):
            text_a = ele
            guid = "%s-%s" % (set_type, i)

            examples.append(
                InputExample(guid=guid, text_a=text_a, label=None))
    return examples

##TODO: slot 中应有nil
##FIXED
def run_cls(training_examples, evaling_examples, label_for_list, test_examples=None):

    args = edict()
    args.task_name = 'dialog'
    args.bert_config_file = '/home/ubuntu/tianchunlin/pytorch_pretrained_BERT_clint/bert-base-chinese/bert_config.json'
    args.vocab_file = '/home/ubuntu/tianchunlin/pytorch_pretrained_BERT_clint/bert-base-chinese/bert-base-chinese-vocab.txt'
    args.init_checkpoint = '/home/ubuntu/tianchunlin/pytorch_pretrained_BERT_clint/bert-base-chinese/pytorch_model.bin'
    args.bert_model = '/home/ubuntu/tianchunlin/pytorch_pretrained_BERT_clint/bert-base-chinese'
    args.output_dir = '/home/ubuntu/tianchunlin/v1_baoxian/cuishou/out/'+str(time.time())+'/'
    args.do_lower_case = False
    args.max_seq_length = 128
    args.do_train = False
    args.do_eval = True
    args.train_batch_size = 10
    args.eval_batch_size = 1
    args.learning_rate = 5e-5
    args.num_train_epochs = 10
    args.warmup_proportion = 0.1
    args.save_checkpoints_steps = 50
    args.no_cuda = False
    args.local_rank = -1
    args.seed = 42
    args.gradient_accumulation_steps = 1
    args.optimize_on_cpu = False
    args.fp16 = False
    args.loss_scale = 128
    args.do_test = False
    args.saving = True


    task_num = len(label_for_list)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
        if args.fp16:
            logger.info("16-bits training currently not supported in distributed training")
            args.fp16 = False  # (see https://github.com/pytorch/pytorch/pull/13496)
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


    bert_config = BertConfig.from_json_file(args.bert_config_file)

    if args.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
                args.max_seq_length, bert_config.max_position_embeddings))

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    # task_name = args.task_name.lower()

    # if task_name not in processors:
    #     raise ValueError("Task not found: %s" % (task_name))

    #processor = processors[task_name]()

    tokenizer = BertTokenizer.from_pretrained(
        pretrained_model_name=args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        #train_examples = processor.get_train_examples(args.data_dir)
        train_examples = training_examples
        num_train_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    label_list = label_for_list


    print("label_list.size:%d\n" % (len(label_list)))

    # Prepare model
    cache_dir = os.path.join('/home/ubuntu/tianchunlin/v1_baoxian', 'distributed_{}'.format(args.local_rank))
    model = BertForSeqClassificationDialog.from_pretrained(args.bert_model,
                                                          cache_dir=cache_dir,
                                                          labels = label_list)
    #model = BertForSequenceClassification(bert_config, len(label_list))
    # if args.init_checkpoint is not None:
    #     model.bert.load_state_dict(torch.load(args.init_checkpoint, map_location='cpu'))
    # if args.fp16:
    #     model.half()
    # model.to(device)
    if args.fp16:
        model = model.half()
    #model = model.to(device)
    model = model.cuda()

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    # 多个 GPU, 但是有错误，先删除之
    # elif n_gpu > 1:
    #     model = torch.nn.DataParallel(model)


    # Prepare optimizer
    if args.fp16:
        param_optimizer = [(n, param.clone().detach().to('cpu').float().requires_grad_()) \
                           for n, param in model.named_parameters()]
    elif args.optimize_on_cpu:
        param_optimizer = [(n, param.clone().detach().to('cpu').requires_grad_()) \
                           for n, param in model.named_parameters()]
    else:
        param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if n not in no_decay], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if n in no_decay], 'weight_decay_rate': 0.0}
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)

    global_step = 0

    if args.do_test:

        model_ = torch.load('./model.pkl')
        model_ = model_.cuda()
        t1 = time.clock()
        test_features = convert_examples_to_features(test_examples, label_list, args.max_seq_length, tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
        model_.eval()
        for input_ids, input_mask, segment_ids in test_data:
            #oo = input_ids.size()
            input_ids = input_ids.view(1, input_ids.size()[0])
            input_mask = input_mask.view(1, input_mask.size()[0])
            segment_ids = segment_ids.view(1, segment_ids.size()[0])
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)

            with torch.no_grad():
                _, logits = model_.forward(input_ids, segment_ids, input_mask)
                t2=time.clock()
                print(t2-t1)
                print_out_sentence(logits, label_list)

    if args.do_train:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids1 = torch.tensor([f.label_id_intent for f in train_features], dtype=torch.long)
        all_label_ids2 = torch.tensor([f.label_id_slot1 for f in train_features], dtype=torch.long)
        all_label_ids3 = torch.tensor([f.label_id_slot2 for f in train_features], dtype=torch.long)
        all_label_ids4 = torch.tensor([f.label_id_slot3 for f in train_features], dtype=torch.long)
        all_label_ids5 = torch.tensor([f.label_id_slot4 for f in train_features], dtype=torch.long)
        all_label_ids6 = torch.tensor([f.label_id_slot5 for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids1,all_label_ids2,all_label_ids3, all_label_ids4,all_label_ids5,all_label_ids6)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:

            train_sampler = RandomSampler(train_data)
            # train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids1,label_ids2,label_ids3,label_ids4,label_ids5,label_ids6 = batch
                #TODO: forward error
                #FIXED
                #loss, _ = model(input_ids, segment_ids, input_mask, [label_ids1,label_ids2,label_ids3,label_ids4,label_ids5,label_ids6])
                loss,_ = model.forward(input_ids, segment_ids, input_mask, [label_ids1,label_ids2,label_ids3,label_ids4,label_ids5,label_ids6])
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.fp16 and args.loss_scale != 1.0:
                    # rescale loss for fp16 training
                    # see https://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html
                    loss = loss * args.loss_scale
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16 or args.optimize_on_cpu:
                        if args.fp16 and args.loss_scale != 1.0:
                            # scale down gradients for fp16 training
                            for param in model.parameters():
                                param.grad.data = param.grad.data / args.loss_scale
                        is_nan = set_optimizer_params_grad(param_optimizer, model.named_parameters(), test_nan=True)
                        if is_nan:
                            logger.info("FP16 TRAINING: Nan in gradients, reducing loss scaling")
                            args.loss_scale = args.loss_scale / 2
                            model.zero_grad()
                            continue
                        optimizer.step()
                        copy_optimizer_params_to_model(model.named_parameters(), param_optimizer)
                    else:
                        optimizer.step()
                    model.zero_grad()
                    global_step += 1

    if args.do_eval:
        time1 = time.clock()
        #eval_examples = processor.get_dev_examples(args.data_dir)
        eval_examples = evaling_examples
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", args.eval_batch_size)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids1 = torch.tensor([f.label_id_intent for f in eval_features], dtype=torch.long)
        all_label_ids2 = torch.tensor([f.label_id_slot1 for f in eval_features], dtype=torch.long)
        all_label_ids3 = torch.tensor([f.label_id_slot2 for f in eval_features], dtype=torch.long)
        all_label_ids4 = torch.tensor([f.label_id_slot3 for f in eval_features], dtype=torch.long)
        all_label_ids5 = torch.tensor([f.label_id_slot4 for f in eval_features], dtype=torch.long)
        all_label_ids6 = torch.tensor([f.label_id_slot5 for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids1,all_label_ids2,all_label_ids3,
                                   all_label_ids4,all_label_ids5,all_label_ids6)
        if args.local_rank == -1:
            eval_sampler = SequentialSampler(eval_data)
        else:

            eval_sampler = SequentialSampler(eval_data)
            # eval_sampler = DistributedSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        eval_intent_acc, eval_slots_acc = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        time2 = time.clock()
        time_s = []
        time_e = []
        for input_ids, input_mask, segment_ids, label_ids1, label_ids2,label_ids3,label_ids4,label_ids5,label_ids6 in eval_dataloader:
            time_s.append(time.clock())
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids1 = label_ids1.to(device)
            label_ids2 = label_ids2.to(device)
            label_ids3 = label_ids3.to(device)
            label_ids4 = label_ids4.to(device)
            label_ids5 = label_ids5.to(device)
            label_ids6 = label_ids6.to(device)

            with torch.no_grad():
                tmp_eval_loss, logits = model(input_ids, segment_ids, input_mask,[label_ids1,label_ids2,label_ids3,label_ids4,label_ids5,label_ids6])
            label_ids1 = label_ids1.to('cpu').numpy()
            label_ids2 = label_ids2.to('cpu').numpy()
            label_ids3 = label_ids3.to('cpu').numpy()
            label_ids4 = label_ids4.to('cpu').numpy()
            label_ids5 = label_ids5.to('cpu').numpy()
            label_ids6 = label_ids6.to('cpu').numpy()
            time_e.append(time.clock())
            # multi-task learning accuracy
            # TODO
            #tmp_eval_accuracy = [0 for _ in range(task_num)]
            tmp_eval_accuracy = np.zeros((task_num,))
            for i in range(task_num):
                logit_cpu = logits[i].detach().cpu().numpy()
                if i==0:
                    tmp_eval_accuracy[i] = accuracy_multilabel(logit_cpu, label_ids1)
                elif i==1:
                    tmp_eval_accuracy[i] = accuracy_oridinary(logit_cpu, label_ids2)
                elif i==2:
                    tmp_eval_accuracy[i] = accuracy_oridinary(logit_cpu, label_ids3)
                elif i==3:
                    tmp_eval_accuracy[i] = accuracy_oridinary(logit_cpu, label_ids4)
                elif i==4:
                    tmp_eval_accuracy[i] = accuracy_oridinary(logit_cpu, label_ids5)
                elif i==5:
                    tmp_eval_accuracy[i] = accuracy_oridinary(logit_cpu, label_ids6)
                else:
                    pass
            #tmp_eval_accuracy = accuracy_oridinary(logits, label_ids)
            # TODO
            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += np.mean(tmp_eval_accuracy)
            eval_intent_acc += tmp_eval_accuracy[0]
            eval_slots_acc += np.mean(tmp_eval_accuracy[1:])

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        eval_intent_acc = eval_intent_acc / nb_eval_examples
        eval_slots_acc = eval_slots_acc / nb_eval_examples
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'eval_intent_acc': eval_intent_acc,
                  'eval_slots_acc': eval_slots_acc,
                  'global_step': global_step,
                  'loss': eval_loss / nb_eval_steps}

        print(time2-time1)
        for e in zip(time_s, time_e):
            print(e[1]-e[0])
        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        if args.saving:
            torch.save(model, 'model.pkl')


def train():
    print('训练集：', len(data_iterator.train))
    print('验证集：', len(data_iterator.valid))
    print('测试集：', len(data_iterator.test))

    #batch_size = len(data_iterator.train)
    #batch_size = 1
    # batch_num = (len(data_iterator.train) - 1) // batch_size + 1
    # epoch_num = 200

    train_data = data_iterator.gen_batches_return(len(data_iterator.train), 'train')
    eval_data = data_iterator.gen_batches_return(len(data_iterator.valid), 'valid')

    logger.info("[loading ontology from {}]".format('./configs/gobot/ontology.json'))
    ontologys = json.load(open("./configs/gobot/ontology.json", encoding='utf-8'))
    slots = ontologys['slots']
    slots_len = len(slots)
    intentions = ontologys['意图']
    #labels = {'intent':intentions,**slots}
    labels_list = [intentions]
    for k,v in slots.items():
        v.append('nil')
        labels_list.append(v)
    #labels = [intentions, slots]
    # 输入 bert 的应该是lists of string
    #train_x = list(map(lambda x: x[0]+'.'+x[2]+'.'+x[3]+'-'+str(x[4]), zip(*train_data)))
    train_x = list(map(lambda x: x[0]+'.'+x[3]+'.'+str(x[4])+'.', zip(*train_data)))
    #eval_x = list(map(lambda x: x[0]+'.'+x[2]+'.'+x[3]+'-'+str(x[4]), zip(*eval_data)))
    eval_x = list(map(lambda x: x[0] + '.' + x[3] + '.' + str(x[4]) + '.', zip(*eval_data)))
    # label是模型的标签，含有1+#slots 个元素，第一个元素是一个list表示意图，之后的元素对应特定的分类器
    # 顺序：身份、配合、买过、了解、预约
    key_serial = {}
    count = 0
    for k,_ in slots.items():
        key_serial[k] = count
        count += 1

    train_slots_list = []
    for e in train_data[6]:
        tmp = ['nil' for _  in range(slots_len)]
        if e:
            for w in e:
                tmp[key_serial[w[0]]] = w[1]
        train_slots_list.append(tmp)

    eval_slots_list = []
    for e in eval_data[6]:
        tmp = ['nil' for _  in range(slots_len)]
        if e:
            for w in e:
                tmp[key_serial[w[0]]] = w[1]
        eval_slots_list.append(tmp)

    train_label = list(map(lambda x: [x[0]]+x[1], zip(train_data[5], train_slots_list)))
    eval_label = list(map(lambda x: [x[0]]+x[1], zip(eval_data[5], eval_slots_list)))

    train_examples = create_examples(train_x, train_label,'train')
    eval_examples = create_examples(eval_x, eval_label, 'dev')

    test_x = ['肯定啊.买过.是.']
    test_example = create_examples(test_x, None, 'dev')
    run_cls(train_examples,eval_examples,labels_list,test_example)


train()