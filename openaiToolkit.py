# required imports
from openai import OpenAI
import re
import json
import tiktoken
import requests
import webbrowser
import numpy as np

def download_img(image_url, save_path="./image.jpg"):
    """
    Downloads an image from a given URL and saves it to the specified path.

    Parameters:
        image_url (str): The URL of the image to download.
        save_path (str): The file path where the downloaded image will be saved.
                         Defaults to './image.jpg'.

    Returns:
        None
    """
    # Fetch the image content from the provided URL
    img_data = requests.get(image_url).content

    # Open the specified file in binary write mode and save the image data
    with open(save_path, 'wb') as handler:
        handler.write(img_data)

class Chatgpt_messages:
    """
    A class to manage and organize messages exchanged in a ChatGPT-like system.
    This includes messages from the system, user, and assistant roles.
    """

    def __init__(self):
        """
        Initializes the Chatgpt_messages object.
        Attributes:
            SYSTEM_ROLE_FLAG (str): Identifier for system messages.
            ASSISTANT_ROLE_FLAG (str): Identifier for assistant messages.
            USER_ROLE_FLAG (str): Identifier for user messages.
            messages (list): List to store message objects in the format 
                             {"role": <role>, "content": <message content>}.
        """
        self.SYSTEM_ROLE_FLAG = "system"
        self.ASSISTANT_ROLE_FLAG = "assistant"
        self.USER_ROLE_FLAG = "user"
        self.messages = []  # Initialize an empty list to store messages

    def append_user_message(self, message):
        """
        Appends a user message to the messages list.

        Parameters:
            message (str): The content of the user's message.
        """
        self.messages.append({"role": self.USER_ROLE_FLAG, "content": message})

    def append_system_message(self, message):
        """
        Appends a system message to the messages list.

        Parameters:
            message (str): The content of the system's message.
        """
        self.messages.append({"role": self.SYSTEM_ROLE_FLAG, "content": message})

    def append_assistant_message(self, message):
        """
        Appends an assistant message to the messages list.

        Parameters:
            message (str): The content of the assistant's message.
        """
        self.messages.append({"role": self.ASSISTANT_ROLE_FLAG, "content": message})

    def clear_messages(self):
        """
        Clears all messages from the messages list.
        """
        self.messages = []

    def get_messages(self):
        """
        Retrieves all messages currently stored.

        Returns:
            list: A list of message objects.
        """
        return self.messages


class Chatgpt_parameters:
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

def chatgpt_submit(openai_client,chatgpt_parameters,chatgpt_messages):
    request=chatgpt_parameters.get_parameters()
    request["messages"]=chatgpt_messages.get_messages()
    response = openai_client.chat.completions.create(**request)
    return response

class Text_embedding:
    def __init__(self,key):
        self.client=OpenAI(api_key=key)
    def get_embedding(self,text,model=0):
        if model==0:
            return self.client.embeddings.create(input=text,model="text-embedding-3-small").data[0].embedding
        elif model==1:
            return self.client.embeddings.create(input=text,model="text-embedding-3-large").data[0].embedding
    def save_embedding(self,embedding,save_path):
        f = open(save_path, "a")
        f.write(json.dumps(embedding))
        f.close()
    def load_embedding(self,save_path):
        f = open(save_path, "r")
        embedding = json.loads(f.read())
        f.close()
        return embedding
    def compare_embeddings(self,embedding_1,embedding_2):
        # Example embeddings
        embedding_1 = np.array(embedding_1)
        embedding_2 = np.array(embedding_2)

        # Compute cosine similarity
        cosine_similarity = np.dot(embedding_1, embedding_2) / (np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2))
        return cosine_similarity

class Dalle_client:
    def __init__(self,key):
        self.client=OpenAI(api_key=key)
        self.SIZES=["1024x1024","1024x1792","1792x1024"]
        self.dalle_model="dall-e-3"
        self.size=0
        self.quality="standard"

    def generate(self,prompt,save_path=""):
        response = self.client.images.generate(
                model=self.dalle_model,
                prompt=prompt,
                size=self.SIZES[self.size],
                quality=self.quality,
                n=1
        )

        image_url = response.data[0].url
        if not save_path=="":
            download_img(image_url,save_path)
        return image_url
class Chatgpt_client:
    def __init__(self,key):
        self.parameters=Chatgpt_parameters()
        self.messages=Chatgpt_messages()
        self.dynamic_system_messages={}
        self.client=OpenAI(api_key=key)
        self.SYSTEM_ROLE_FLAG="system"
        self.ASSISTANT_ROLE_FLAG="assistant"
        self.USER_ROLE_FLAG="user"
        self.APIs={}
    def add_api(self,api_name,api_function):
        self.APIs[api_name]=api_function
    def update_api(self,api_name,api_function):
        self.APIs[api_name]=api_function
    def remove_api(self,api_name):
        del self.APIs[api_name]
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
        api_message=""
        self.messages.append_user_message(user_message)
        request=self.parameters.get_parameters()
        request["messages"]=self.messages.get_messages()
        for key in self.dynamic_system_messages:
            request["messages"].append({"role":"system","content":self.dynamic_system_messages[key]})
        request["messages"].append({"role":"system","content":"<API_CALL|(api_name)|(json for the API request)>."})
        request["messages"].append({"role":"system","content":"APIs may respond from requests with their messages inside <API_RESPONSE> tags."})
        response = self.client.chat.completions.create(**request)
        assistant_message = response.choices[0].message.content
        pattern = r"<API_CALL\|[^|]+\|[^>]+>"
        for api_call in re.finditer(pattern,assistant_message):
            api_call_parts=api_call.string[1:-1].split("|")
            if self.APIs.__contains__(api_call_parts[1]):
                api_response = self.APIs[api_call_parts[1]](json.loads(api_call_parts[2]))
                if not response == None and not response=="":
                    api_message+="<API_RESPONSE name=\"{}\">{}</API_RESPONSE>".format(api_call_parts[1],api_response)
        self.messages.append_assistant_message(assistant_message)

        return assistant_message,api_message