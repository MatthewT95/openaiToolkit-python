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
    """
    A class to manage parameters for a ChatGPT model configuration.
    These parameters control the model's behavior, output length, and sampling strategies.
    """

    def __init__(self):
        """
        Initializes the Chatgpt_parameters object with default values for the parameters.
        Attributes:
            model (str): The identifier for the model to use.
            temperature (float): Controls randomness in the model's output (higher = more random).
            max_tokens (int): Maximum number of tokens the model can generate in a response.
            top_p (float): Probability threshold for nucleus sampling (0 < top_p <= 1).
            frequency_penalty (float): Penalizes repeated tokens in the output.
            presence_penalty (float): Penalizes new tokens based on their presence in the input.
        """
        self.model = "gpt-4o-mini"
        self.temperature = 1
        self.max_tokens = 2048
        self.top_p = 1
        self.frequency_penalty = 0
        self.presence_penalty = 0

    def set_model(self, model="gpt-4o-mini"):
        """
        Sets the model to be used.

        Parameters:
            model (str): The identifier for the model (default: "gpt-4o-mini").
        """
        self.model = model

    def set_temperature(self, temperature=1):
        """
        Sets the temperature for the model.

        Parameters:
            temperature (float): Value controlling randomness (default: 1).
        """
        self.temperature = temperature

    def set_max_tokens(self, max_tokens=2048):
        """
        Sets the maximum number of tokens for the model's response.

        Parameters:
            max_tokens (int): Maximum token count (default: 2048).
        """
        self.max_tokens = max_tokens

    def set_top_p(self, top_p=1):
        """
        Sets the top-p value for nucleus sampling.

        Parameters:
            top_p (float): Value controlling probability threshold for sampling (default: 1).
        """
        self.top_p = top_p

    def set_frequency_penalty(self, frequency_penalty=0):
        """
        Sets the frequency penalty to control repetition in responses.

        Parameters:
            frequency_penalty (float): Penalty for repeated tokens (default: 0).
        """
        self.frequency_penalty = frequency_penalty

    def set_presence_penalty(self, presence_penalty=0):
        """
        Sets the presence penalty to control token diversity in responses.

        Parameters:
            presence_penalty (float): Penalty for encouraging diversity (default: 0).
        """
        self.presence_penalty = presence_penalty

    def get_parameters(self):
        """
        Retrieves the current parameters as a dictionary.

        Returns:
            dict: A dictionary containing all the parameters and their values.
        """
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty
        }

def chatgpt_submit(openai_client, chatgpt_parameters, chatgpt_messages):
    """
    Submits a request to the OpenAI API using the provided parameters and messages.

    Parameters:
        openai_client (object): An instance of the OpenAI API client to handle the request.
        chatgpt_parameters (Chatgpt_parameters): An object containing configuration parameters for the request.
        chatgpt_messages (Chatgpt_messages): An object containing the list of messages to be included in the request.

    Returns:
        response (dict): The response from the OpenAI API.
    """
    # Retrieve parameters for the API request
    request = chatgpt_parameters.get_parameters()
    
    # Add the list of messages to the request
    request["messages"] = chatgpt_messages.get_messages()
    
    # Send the request to the OpenAI API and get the response
    response = openai_client.chat.completions.create(**request)
    
    # Return the API response
    return response


class Text_embedding:
    """
    A class to generate, save, load, and compare text embeddings using the OpenAI API.
    """

    def __init__(self, key):
        """
        Initializes the Text_embedding object with an API key.

        Parameters:
            key (str): The API key for authenticating with the OpenAI service.
        """
        self.client = OpenAI(api_key=key)

    def get_embedding(self, text, model=0):
        """
        Generates a text embedding for the given text using the specified model.

        Parameters:
            text (str): The input text to generate the embedding for.
            model (int): The model to use for embedding. 
                         - 0: Small embedding model ("text-embedding-3-small").
                         - 1: Large embedding model ("text-embedding-3-large").

        Returns:
            list: The generated embedding as a list of floats.
        """
        if model == 0:
            return self.client.embeddings.create(input=text, model="text-embedding-3-small").data[0].embedding
        elif model == 1:
            return self.client.embeddings.create(input=text, model="text-embedding-3-large").data[0].embedding

    def save_embedding(self, embedding, save_path):
        """
        Saves an embedding to a file in JSON format.

        Parameters:
            embedding (list): The embedding to save.
            save_path (str): The file path where the embedding will be saved.
        """
        with open(save_path, "a") as f:
            f.write(json.dumps(embedding))

    def load_embedding(self, load_path):
        """
        Loads an embedding from a file in JSON format.

        Parameters:
            load_path (str): The file path to load the embedding from.

        Returns:
            list: The loaded embedding as a list of floats.
        """
        with open(load_path, "r") as f:
            embedding = json.loads(f.read())
        return embedding

    def compare_embeddings(self, embedding_1, embedding_2):
        """
        Computes the cosine similarity between two embeddings.

        Parameters:
            embedding_1 (list): The first embedding as a list of floats.
            embedding_2 (list): The second embedding as a list of floats.

        Returns:
            float: The cosine similarity between the two embeddings.
        """
        # Convert embeddings to numpy arrays
        embedding_1 = np.array(embedding_1)
        embedding_2 = np.array(embedding_2)

        # Compute cosine similarity
        cosine_similarity = np.dot(embedding_1, embedding_2) / (
            np.linalg.norm(embedding_1) * np.linalg.norm(embedding_2)
        )
        return cosine_similarity

class Dalle_client:
    """
    A client class to interact with the OpenAI DALL-E API for generating images
    based on text prompts. The class supports different image sizes and qualities.
    """

    def __init__(self, key):
        """
        Initializes the Dalle_client with API credentials and default settings.

        Parameters:
            key (str): The API key for authenticating with the OpenAI service.
        """
        self.client = OpenAI(api_key=key)  # OpenAI client instance
        self.SIZES = ["1024x1024", "1024x1792", "1792x1024"]  # Supported image sizes
        self.dalle_model = "dall-e-3"  # Default model for image generation
        self.size = 0  # Default size index (corresponds to "1024x1024")
        self.quality = "standard"  # Default image quality

    def generate(self, prompt, save_path=""):
        """
        Generates an image based on the provided text prompt.

        Parameters:
            prompt (str): The text prompt to describe the desired image.
            save_path (str): The file path to save the generated image. If empty,
                             the image is not saved locally.

        Returns:
            str: The URL of the generated image.
        """
        # Send the request to generate an image using the DALL-E API
        response = self.client.images.generate(
            model=self.dalle_model,
            prompt=prompt,
            size=self.SIZES[self.size],
            quality=self.quality,
            n=1  # Number of images to generate
        )

        # Extract the URL of the generated image
        image_url = response.data[0].url

        # Save the image locally if a save path is provided
        if save_path != "":
            download_img(image_url, save_path)

        # Return the URL of the generated image
        return image_url

import re
import json
from openai import OpenAI

class Chatgpt_client:
    """
    A client class to interact with OpenAI's ChatGPT API, manage parameters, messages,
    dynamic system messages, and APIs for enhanced functionality.
    """

    def __init__(self, key):
        """
        Initializes the Chatgpt_client with an API key and default configurations.

        Parameters:
            key (str): The API key for authenticating with the OpenAI service.
        """
        self.parameters = Chatgpt_parameters()  # ChatGPT parameters instance
        self.messages = Chatgpt_messages()  # ChatGPT messages instance
        self.dynamic_system_messages = {}  # Dynamic system messages
        self.client = OpenAI(api_key=key)  # OpenAI client instance
        self.SYSTEM_ROLE_FLAG = "system"  # Flag for system role
        self.ASSISTANT_ROLE_FLAG = "assistant"  # Flag for assistant role
        self.USER_ROLE_FLAG = "user"  # Flag for user role
        self.APIs = {}  # Dictionary to manage API functions

    def add_api(self, api_name, api_function):
        """
        Adds a new API to the client.

        Parameters:
            api_name (str): The name of the API.
            api_function (callable): The function to handle the API logic.
        """
        self.APIs[api_name] = api_function

    def update_api(self, api_name, api_function):
        """
        Updates an existing API with a new function.

        Parameters:
            api_name (str): The name of the API.
            api_function (callable): The new function to handle the API logic.
        """
        self.APIs[api_name] = api_function

    def remove_api(self, api_name):
        """
        Removes an API from the client.

        Parameters:
            api_name (str): The name of the API to remove.
        """
        del self.APIs[api_name]

    def set_dynamic_system_message(self, key, message):
        """
        Sets a dynamic system message.

        Parameters:
            key (str): The identifier for the dynamic message.
            message (str): The content of the dynamic message.
        """
        self.dynamic_system_messages[key] = message

    def set_parameters(self, parameters):
        """
        Sets the ChatGPT parameters.

        Parameters:
            parameters (Chatgpt_parameters): The new parameters object.
        """
        self.parameters = parameters

    # Parameter configuration methods
    def set_model(self, model="gpt-4-mini"):
        self.parameters.model = model

    def set_temperature(self, temperature=1):
        self.parameters.temperature = temperature

    def set_max_tokens(self, max_tokens=2048):
        self.parameters.max_tokens = max_tokens

    def set_top_p(self, top_p=1):
        self.parameters.top_p = top_p

    def set_frequency_penalty(self, frequency_penalty=0):
        self.parameters.frequency_penalty = frequency_penalty

    def set_presence_penalty(self, presence_penalty=0):
        self.parameters.presence_penalty = presence_penalty

    # Message management methods
    def append_user_message(self, message):
        self.messages.append_user_message(message)

    def append_system_message(self, message):
        self.messages.append_system_message(message)

    def append_assistant_message(self, message):
        self.messages.append_assistant_message(message)

    def clear_messages(self):
        self.messages.clear_messages()

    def get_messages(self):
        return self.messages.get_messages()

    def submit(self, user_message):
        """
        Submits a user message to the OpenAI API and processes API calls.

        Parameters:
            user_message (str): The user's input message.

        Returns:
            tuple: A tuple containing:
                - assistant_message (str): The assistant's response message.
                - api_message (str): The API responses, if any, formatted within <API_RESPONSE> tags.
        """
        api_message = ""

        # Append the user message to the message history
        self.messages.append_user_message(user_message)

        # Prepare the request with parameters and message history
        request = self.parameters.get_parameters()
        request["messages"] = self.messages.get_messages()

        # Add dynamic system messages to the request
        for key in self.dynamic_system_messages:
            request["messages"].append({"role": "system", "content": self.dynamic_system_messages[key]})

        # Append API call instructions to the request
        request["messages"].append({"role": "system", "content": "<API_CALL|(api_name)|(json for the API request)>."})
        request["messages"].append({"role": "system", "content": "APIs may respond from requests with their messages inside <API_RESPONSE> tags."})

        # Send the request to the OpenAI API
        response = self.client.chat.completions.create(**request)

        # Extract the assistant's message from the response
        assistant_message = response.choices[0].message.content

        # Regex pattern to identify API call instructions
        pattern = r"<API_CALL\|[^|]+\|[^>]+>"

        # Process API calls from the assistant's response
        for api_call in re.finditer(pattern, assistant_message):
            api_call_parts = api_call.string[1:-1].split("|")  # Parse API call components
            if api_call_parts[1] in self.APIs:  # Check if the API exists
                # Execute the API function with the parsed JSON arguments
                api_response = self.APIs[api_call_parts[1]](json.loads(api_call_parts[2]))
                if api_response not in [None, ""]:
                    # Format the API response within <API_RESPONSE> tags
                    api_message += "<API_RESPONSE name=\"{}\">{}</API_RESPONSE>".format(api_call_parts[1], api_response)

        # Append the assistant's message to the message history
        self.messages.append_assistant_message(assistant_message)

        return assistant_message, api_message