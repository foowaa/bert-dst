import copy
import json
from pathlib import Path
from typing import Dict, List
from random import Random

from components.log import get_logger


log = get_logger(__name__)


class JRCSDatasetReader:
    """
    将文件格式转化为 dstc 格式，并切分为 trn、val、tst三块, 通过调用静态方法 read 使用
    JRCS 输入：
        2,开场语-身份确认|询问语-身份-本人,,您好哎。这里是京东金融公司，请问是无缝为先生吗？
        1,你好,(身份，本人),你好喂，你好。
        2,开场语-是本人|询问语-还款时间-今天,,您在我们平台的借款592点，5陵园已经逾期了2天，是否能在今天还款，避免产生更多费用？
        1,肯定|忘记了,(还款时间，今天)|(欠款原因，忘记),能能能忘了。
        2,结束语-承诺今天还款,,好的。如果您在还款过程中遇到任何问题？可以回电95118？祝您生活愉快，再见好。
        ,,,
        2,开场语-身份确认|询问语-身份-本人,,您好，这里是京东金融公司，请问是周帅先生呢？
        1,肯定,(身份，本人),啊。呃，对。
        2,开场语-是本人|询问语-还款时间-今天,,您在我们平台的借款338.15元已经逾期了2天，是否能在今天还款，避免产生更多费用？
        1,语气词,,啊。说。
        2,回复语-无法理解|询问语-还款时间-今天,,抱歉，没听清借款今天能处理吗？
        1,肯定,(还款时间，今天),可以可以。好的。
        2,结束语-承诺今天还款,,如果您在还款过程中遇到任何问题？可以回电95118，祝您生活愉快再见。
    输出：
        {
            'train': (
                {
                    'text':      # utterance string
                    'act':       # list of string intents and slot, associated with user’s utterance
                },
                {
                    'text':      # sys  string
                    'act':       # an act, associated with the user’s utterance
                }
            ),
            'valid': 同上,
            'test': 同上,
        }
    """

    @staticmethod
    def _data_fname(datatype):
        assert datatype in ('trn', 'val', 'tst'), "wrong datatype name"
        return 'my-dst-{}.csv'.format(datatype)

    @staticmethod
    def _read_file(fn):
        """Returns data from single file"""
        log.info("[loading dialogs from {}]".format(fn))

        datas = [[]]
        for line in open(fn):
            data = line.strip()

            # len(',,,')<4
            if len(data)<4:
                datas.append([])
            else:
                datas[-1].append(data)
        return datas

    @classmethod
    def _split_dataset(cls, fn, split_rate, seed):
        '''
        1. Read file from `fn` and get all datas
        2. shuffle datas by `seed`
        3. Split data to train、valid、test by `split_rate`
        4. return train_data、valid_data、test_data
        '''

        train_rate = split_rate[0]
        valid_data = split_rate[1]

        random = Random(seed)
        datas = cls._read_file(fn)
        data_num = len(datas)

        order = list(range(data_num))
        random.shuffle(order)

        train_num = int(data_num * train_rate)
        valid_num = int(data_num * valid_data)
        train_data = [ datas[order[i]] for i in range(0, train_num) ]
        valid_data = [ datas[order[i]] for i in range(train_num, train_num + valid_num) ]
        test_data = [ datas[order[i]] for i in range(train_num + valid_num, data_num) ]

        return train_data, valid_data, test_data

    @classmethod
    def read(self, 
            data_path: str, 
            dataset_fn: str = 'dataset.csv', 
            split_rate: List[float] = [0.6, 0.3, 0.1], 
            seed: int = 9527, 
            dialogs: bool = False) -> Dict[str, List]:

        required_files = (self._data_fname(dt) for dt in ('trn', 'val', 'tst'))

        if not all(Path(data_path, f).exists() for f in required_files):
            fn = Path(data_path, dataset_fn)
            train_data, valid_data, test_data = self._split_dataset(fn, split_rate, seed)
        else:
            train_data, valid_data, test_data = ( self._read_file(f) for f in required_files )

        data = {
            'train': self._format_data_to_dst(train_data, dialogs),
            'valid': self._format_data_to_dst(valid_data, dialogs),
            'test': self._format_data_to_dst(test_data, dialogs)
        }

        return data

    @staticmethod
    def _get_resp_act(intent_str):
        '''
        Parsing sys info to dst act.
        '''
        result = { 'intent':[], 'slots':[] }
        intent_str = intent_str.split('|')
        for intention_i in intent_str:
            intent_split = intention_i.split('-')
            assert len(intent_split)==2 or len(intent_split)==3, 'intents have to split to split 2 or 3 parts.'

            if len(intent_split)==2:
                intent_split.append(None)
            result['intent'].append('-'.join(intent_split[:2]))
            result['slots'].append([intent_split[1], intent_split[2]])
        return result
    
    @staticmethod
    def _get_utter_act(intent_str, slot_str):
        '''
        Parsing utter info to dst act.
        '''
        result = { 'intent':[], 'slots':[] }
        intent_split = intent_str.split('|')
        slot_split = slot_str.split('|')
        for intention_i in intent_split:
            result['intent'].append(intention_i)
        for slot_i in slot_split:
            if len(slot_i)<3:
                continue
            kv = slot_i[1:-1].split('，')
            if len(kv)!=2:
                print('kv len is not 2, ', intent_str, slot_str, slot_split)
            else:
                result['slots'].append(kv)
        return result

    @classmethod
    def _format_data_to_dst(cls, data, dialogs):
        '''
        Format data to dst.
        Input:
        data: list of dialog, a dialog is a list of sentence information which contains speaker, intents, slots and text.
        dialogs: if True, return dialogs formatting data( list of dialogs).

        Return:
        if dialogs is True, return list of dialogs, each dialog is a list of (utterances, responses)
        if dialogs is False, return list of (utterances, responses)
        '''

        utterances = []
        responses = []
        dialog_indices = []
        n = 0
        for a_dialogue in data:
            num_dialog_utter, num_dialog_resp = 0, 0
            for a_turn in a_dialogue:
                if len(a_turn.split(',')) != 4:
                    log.warning('Turn num is not 4, ' + a_turn)
                    continue

                speaker, intents, slots, text = a_turn.split(',')
                if num_dialog_resp == num_dialog_utter and speaker == '1':
                    log.warning('在面向任务型的机器人中，机器人应该先说话：' + a_dialogue)
                    continue

                if speaker=='2':
                    responses.append({'text':text, 'act':cls._get_resp_act(intents) })
                    num_dialog_resp += 1
                elif speaker=='1':
                    utterances.append({'text':text, 'act':cls._get_utter_act(intents, slots) })
                    num_dialog_utter += 1
                else:
                    raise RuntimeError("The speaker has to 1 or 2: " + a_turn)

            if num_dialog_resp == num_dialog_utter:
                pass
            elif num_dialog_resp == num_dialog_utter + 1:
                responses = responses[:-1]
            else:
                raise RuntimeError("Datafile in the wrong format. ")

            n += num_dialog_utter
            dialog_indices.append({
                'start': n - num_dialog_utter,
                'end': n,
            })

        result = list(zip(utterances, responses))
        if dialogs:
            return [result[idx['start']:idx['end']] for idx in dialog_indices]
        return result

