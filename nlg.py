import  json


class NLG:
    # get text by action from config file
    def __init__(self,
                 json_path):

        self.json_path=json_path
        self.get_json_infor()

    def get_json_infor(self):
    # get config file
        f = open(self.json_path, encoding='utf-8')
        file=json.load(f)

        self.action_to_text=file
        self.action_name=list(self.action_to_text.keys())

    def get_text(self,actions):
    # get text
        text=''
        for i in actions:
            if i in self.action_name and i!=' ':
                tmp=self.action_to_text[i]
                text+=tmp
        return  text
