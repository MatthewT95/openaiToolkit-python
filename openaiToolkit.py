from openai import OpenAI
class chatgpt_messages:
    def __init__(self):
        self.SYSTEM_ROLE_FLAG="system"
        self.ASSISTANT_ROLE_FLAG="assistant"
        self.USER_ROLE_FLAG="user"
        self.messages = []  # Assign to instance
    
    def append_user_message(self,message):
        self.messages.append({"role":self.USER_ROLE_FLAG,"content":message})
    def append_system_message(self,message):
        self.messages.append({"role":self.SYSTEM_ROLE_FLAG,"content":message})
    def append_assistant_message(self,message):
        self.messages.append({"role":self.ASSISTANT_ROLE_FLAG,"content":message})
    def clear_messages(self):
        self.messages=[]
    def get_messages(self):
        return self.messages

class chatgpt_parameters:
    def __init__(self):
        self.model="gpt-4o-mini"
        self.temperature=1
        self.max_tokens=2048
        self.top_p=1
        self.frequency_penalty=0
        self.presence_penalty=0
    
    def set_model(self,model="gpt-o4-mini"):
        self.model=model
    def set_temperature(self,temperature=1):
        self.temperature=temperature
    def set_max_tokens(self,max_tokens=2048):
        self.max_tokens=max_tokens
    def set_top_p(self,top_p=1):
        self.top_p=top_p
    def set_frequency_penalty(self,frequency_penalty=0):
        self.frequency_penalty=frequency_penalty
    def set_presence_penalty(self,presence_penalty=0):
        self.presence_penalty=presence_penalty
    def get_parameters(self):
        return {
            "model":self.model,
            "temperature":self.temperature,
            "max_tokens":self.max_tokens,
            "top_p":self.top_p,
            "frequency_penalty":self.frequency_penalty,
            "presence_penalty":self.presence_penalty
        }

def chatgpt_send(openai_client,chatgpt_parameters,chatgpt_messages):
    request=chatgpt_parameters.get_parameters()
    request["messages"]=chatgpt_messages.get_messages()
    response = openai_client.chat.completions.create(**request)
    return response

class chatgpt_client:
    def __init__(self,key):
        self.parameters=chatgpt_parameters()
        self.messages=chatgpt_messages()
        self.dynamic_system_messages={}
        self.client=OpenAI(api_key=key)
        self.SYSTEM_ROLE_FLAG="system"
        self.ASSISTANT_ROLE_FLAG="assistant"
        self.USER_ROLE_FLAG="user"
    def set_dynamic_system_message(self,key,message):
        self.dynamic_system_messages[key]=message
    def set_parameters(self,parameters):
        self.parameters=parameters
    def set_model(self,model="gpt-o4-mini"):
        self.parameters.model=model
    def set_temperature(self,temperature=1):
        self.parameters.temperature=temperature
    def set_max_tokens(self,max_tokens=2048):
        self.parameters.max_tokens=max_tokens
    def set_top_p(self,top_p=1):
        self.parameters.top_p=top_p
    def set_frequency_penalty(self,frequency_penalty=0):
        self.parameters.frequency_penalty=frequency_penalty
    def set_presence_penalty(self,presence_penalty=0):
        self.parameters.presence_penalty=presence_penalty
    def append_user_message(self,message):
        self.messages.append_user_message(message)
    def append_system_message(self,message):
        self.messages.append_system_message(message)
    def append_assistant_message(self,message):
        self.messages.append_assistant_message(message)
    def clear_messages(self):
        self.messages.clear_messages()
    def get_messages(self):
        return self.messages.get_messages()
    def submit(self,user_message):
        self.messages.append_user_message(user_message)
        request=self.parameters.get_parameters()
        request["messages"]=self.messages.get_messages()
        for key in self.dynamic_system_messages:
            request["messages"].append({"role":"system","content":self.dynamic_system_messages[key]})
        response = self.client.chat.completions.create(**request)
        assistant_message = response.choices[0].message.content
        self.messages.append_assistant_message(assistant_message)
        return assistant_message