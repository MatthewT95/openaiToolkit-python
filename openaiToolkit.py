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

