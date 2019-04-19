from overrides import overrides
from collections import Counter

from components.data_learning_iterator import DataLearningIterator


class SelfDialogDatasetIterator(DataLearningIterator):
    """
    Iterates over dialog data,
    generates batches where one sample is one dialog.

    A subclass of :class:`~deeppavlov.core.data.data_learning_iterator.DataLearningIterator`.

    Attributes:
        train: list of training dialogs (tuples ``(context, response)``)
        valid: list of validation dialogs (tuples ``(context, response)``)
        test: list of dialogs used for testing (tuples ``(context, response)``)
    """

    @staticmethod
    def _get_input_x(datas):
        utter_text = []
        sys_text = []
        sys_act = []
        sys_q = []
        sys_s = []
        sys_v = []
        for data in datas:
            x = data[0]
            y = data[1]
            utter_text.append(x['text'])
            sys_text.append(y['text'])

            this_act = y['act']
            this_intent = this_act['intent']
            this_slots = this_act['slots']

            sys_act.append(this_act)
            sys_q.append(this_intent[-1])
            
            this_s = None
            this_v = None
            if len(this_slots)>0:
                this_s = this_slots[-1][0]
                this_v = this_slots[-1][1]
            sys_s.append(this_s)
            sys_v.append(this_v)
        return utter_text, sys_text, sys_q, sys_s, sys_v

    @staticmethod
    def _get_input_y(datas):
        y_dic = {'intent':[], 'slots':[]}
        for data in datas:
            y = data[0]['act']
            y_intent = y['intent']
            y_slot = y['slots']
            y_dic['intent'].append(y_intent)
            y_dic['slots'].append(y_slot)
        return y_dic

    def _preprocess_data(self, datas):
        utter_text, sys_text, sys_q, sys_s, sys_v = self._get_input_x(datas)
        y_dic = self._get_input_y(datas)

        return list(zip(utter_text, sys_text, sys_q, sys_s, sys_v, y_dic['intent'], y_dic['slots']))

    @overrides
    def split(self, *args, **kwargs):
        self.train = self._preprocess_data(self.train)
        self.valid = self._preprocess_data(self.valid)
        self.test = self._preprocess_data(self.test)