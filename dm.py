import json

class DialogueManagement:
    # get action  from nlu by intention and slots
    # Args:
    #    slot_name: slot name for slot
    #    json_path: json_path for config file(define node)
    def __init__(self,
                 slot_name,
                 json_path):
        self.slot_name=slot_name
        self.json_path=json_path
        self.node_name = []
        self.node_trigger=[]
        self.globel_node_name = []
        self.globel_node_trigger=[]
        self.node_action=[]
        self.intention2end_intention=[]
        self.intention2end_name=[]
        self.repeat = 0
        self.globel=0

        self.get_json_infor()
        self.get_init_state()

    def get_json_infor(self):
        # get config information including node(common node、ntention2end node、globel node) and node triggle and global intention
        f = open(self.json_path, encoding='utf-8')
        file=json.load(f)

        self.node=file['node']
        self.globel_node=file['global_node']
        self.global_error_node=file['global_error_node']
        self.quetion=file['question']
        self.intention2slot=file['intention2slot']
        self.intention2end=file['intention2end']


        for node in self.intention2end.keys():
            self.intention2end_intention.append(self.intention2end[node]['triggle'])
            self.intention2end_name.append(node)

        for node in self.node.keys():
            for trigger in  self.node[node]['triggle']:
                self.node_trigger.append(trigger)
                self.node_name.append(node)

        for node in self.globel_node.keys():
            for trigger in self.globel_node[node]['triggle']:
                self.globel_node_trigger.append(trigger)
                self.globel_node_name.append(node)

    def get_init_state(self):
        # init state ,need for new  dialogue
        self.history =[]
        for i in self.slot_name:
            tmp= i + ',none'
            self.history.append(tmp)
        self.current=self.history
        self.current_dict={}
        for i in self.history:
            tmp=i.split(',')
            self.current_dict[tmp[0]]=tmp[1]


    def dict_to_list(self,dict_data):
        # change dict to list
        #example {'身份':'本人'} to ['身份,本人']
        list_data = []
        for i in dict_data.keys():
            tmp = i + ',' + dict_data[i]
            list_data.append(tmp)
        return list_data


    def update_state(self,slot_dict):
        # get current sate by slot
        self.history=self.current
        slot_list=self.dict_to_list(slot_dict)
        #slot_concate=self.history+slot_list

        for i in slot_list:
            tmp = i.split(',')
            if tmp[1] !='none':
                self.current_dict[tmp[0]] = tmp[1]
        self.current=self.dict_to_list(self.current_dict)

    def get_global_question_antion(self,intention, isSlotInfo):
        # get global question action by intention
        # args:
        #     isSlotInfo :if true , not all slot is none
        #                 to avoid retrun "全局问题_无法理解" when get slot information and intention ='无法理解'
        self.intention_action=[]
        for i in intention:
            if i in self.quetion:
                if i=='无法理解' and isSlotInfo:
                    continue
                tmp='全局问题_'+i
                self.intention_action.append(tmp)


    def get_node(self):
        # get node by state(slot)
        if self.history==self.current:
            self.repeat+=1
        else:
            self.repeat=0
        self.globel = 0
        for i in self.globel_node_trigger:
            if set(i)<set(self.current):
                self.now_node=self.globel_node[self.globel_node_name[self.globel_node_trigger.index(i)]]

                return None

        for i in self.node_trigger:
            if set(i)<set(self.current):
                self.now_node=self.node[self.node_name[self.node_trigger.index(i)]]
                return None

        print('不是触发条件')

    def get_node_action(self):
        # get node action
        self.node_action = []

        if self.repeat>self.now_node['repeat'] :
            self.node_action.append(self.global_error_node['welcome'])
            return None
        need_infor=0
        if self.repeat==0:
            self.node_action.append(self.now_node['welcome'])
        for i in self.now_node['required_slots'][0]:
            if self.current_dict[i]=='none' or self.current_dict[i]=='不是本人' or self.current_dict[i]=='不是今天':
                need_infor=1
                idx=self.now_node['required_slots'][0].index(i)
                self.node_action.append(self.now_node['required_slots'][1][idx])
                return None
        if need_infor==0 and self.now_node['end']=='True':
            self.node_action.append(self.now_node['ending'])

    def get_sys_action_from_nlu(self, nlu_result, sys_q):
        # update slot by intention

        print('        NLU输出：', nlu_result)
        slot = {}
        for slot_k in self.slot_name:
            if len(nlu_result[slot_k])>0 and nlu_result[slot_k][0] is not None:
                slot[slot_k] = nlu_result[slot_k][0]
            else:
                slot[slot_k] = 'none'
        intention = nlu_result['意图']
        if len(intention)==0:
            intention = ['无法理解']
        if sys_q in self.intention2slot.keys():
            for intent_i in intention:
                if intent_i in self.intention2slot[sys_q].keys():
                    k, v = self.intention2slot[sys_q][intent_i]
                    slot[k] = v
        print('        DM 输入：', slot)

        return self.get_sys_action(slot, intention)

    def get_sys_action(self, slot, intention):
       # wheather intention in intention2end
        for i in intention:
            for idx, trigg in enumerate(self.intention2end_intention):
                if i in trigg:
                    self.sys_action=[]
                    self.now_node=self.intention2end[self.intention2end_name[idx]]
                    self.current=slot
                    self.sys_action.append(self.now_node['welcome'])
                    return self.sys_action
        # intention not in intention2end
        isSlotInfo = False
        for k, v in slot.items():
            if v != 'none':
                isSlotInfo = True
        self.update_state(slot)
        self.get_global_question_antion(intention, isSlotInfo)
        self.get_node()
        self.get_node_action()
        sys_action=self.intention_action+self.node_action
        return sys_action

