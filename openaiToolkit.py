# Standard library imports
import os  # File system utilities, e.g., checking paths, creating directories
import re  # Regular expressions for pattern matching
import json  # JSON handling for API responses, requests, and file operations

# Third-party library imports
import numpy as np  # Mathematical operations, e.g., cosine similarity
from urllib.parse import urlparse  # URL validation and parsing
import requests  # Making HTTP requests
from concurrent.futures import ThreadPoolExecutor  # For parallel processing
import webbrowser  # Opening URLs in the default web browser

# OpenAI-specific imports
from openai import OpenAI  # OpenAI client for interacting with the API
import tiktoken  # Tokenization library for token counting (if needed)


def download_img(image_url, save_path="./image.jpg"):
    """
    Downloads an image from a given URL and saves it to the specified path.

    Parameters:
        image_url (str): The URL of the image to download.
        save_path (str): The file path where the downloaded image will be saved.
                         Defaults to './image.jpg'.

    Returns:
        bool: True if the image was successfully downloaded and saved, False otherwise.
    """
    try:
        # Validate the image URL
        if not isinstance(image_url, str) or not image_url.strip():
            raise ValueError("The image URL must be a non-empty string.")
        
        parsed_url = urlparse(image_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            raise ValueError("The image URL is not valid.")
        
        # Validate the save path
        if not isinstance(save_path, str) or not save_path.strip():
            raise ValueError("The save path must be a non-empty string.")
        
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Fetch the image content from the provided URL
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()  # Raise an HTTPError for bad HTTP responses (4xx or 5xx)

        # Save the image content to the specified path
        with open(save_path, 'wb') as handler:
            handler.write(response.content)

        print(f"Image successfully downloaded and saved to {save_path}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Failed to download the image: {e}")
    except ValueError as e:
        print(f"Validation error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return False


class Message_roles:
    """
    Centralized configuration for constants used across the application.
    """
    SYSTEM_ROLE_FLAG = "system"
    ASSISTANT_ROLE_FLAG = "assistant"
    USER_ROLE_FLAG = "user"

    @staticmethod
    def get_valid_roles():
        """
        Returns a set of valid roles for message validation.
        """
        return {Message_roles.SYSTEM_ROLE_FLAG, Message_roles.ASSISTANT_ROLE_FLAG, Message_roles.USER_ROLE_FLAG}

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

    def append_user_message(self, message):
        """
        Appends a user message to the messages list.

        Parameters:
            message (str): The content of the user's message.
        """
        self.messages.append({"role": Message_roles.USER_ROLE_FLAG, "content": message})

    def append_system_message(self, message):
        """
        Appends a system message to the messages list.

        Parameters:
            message (str): The content of the system's message.
        """
        self.messages.append({"role": Message_roles.SYSTEM_ROLE_FLAG, "content": message})

    def append_assistant_message(self, message):
        """
        Appends an assistant message to the messages list.

        Parameters:
            message (str): The content of the assistant's message.
        """
        self.messages.append({"role": Message_roles.ASSISTANT_ROLE_FLAG, "content": message})

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
    @staticmethod
    def validate(messages):
        """
        Validates the messages list.

        Parameters:
            messages (list): The list of message dictionaries.

        Returns:
            list: The validated messages list.

        Raises:
            ValueError: If any message is invalid or improperly formatted.
        """
        if not isinstance(messages, list):
            raise ValueError("Messages must be a list.")

        valid_roles = Message_roles.get_valid_roles()
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError(f"Each message must be a dictionary. Invalid message: {message}")
            if "role" not in message or message["role"] not in valid_roles:
                raise ValueError(f"Each message must have a 'role' key with a value of 'system', 'assistant', or 'user'. Invalid message: {message}")
            if "content" not in message or not isinstance(message["content"], str):
                raise ValueError(f"Each message must have a 'content' key with a non-empty string value. Invalid message: {message}")

        return messages


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

        Raises:
            ValueError: If the model is not a valid string.
        """
        if not isinstance(model, str) or not model.strip():
            raise ValueError("The 'model' must be a non-empty string.")
        self.model = model

    def set_temperature(self, temperature=1):
        """
        Sets the temperature for the model.

        Parameters:
            temperature (float): Value controlling randomness (default: 1, range: 0 to 2).

        Raises:
            ValueError: If temperature is not within the range [0, 2].
        """
        if not (0 <= temperature <= 2):
            raise ValueError("The 'temperature' must be between 0 and 2.")
        self.temperature = temperature

    def set_max_tokens(self, max_tokens=2048):
        """
        Sets the maximum number of tokens for the model's response.

        Parameters:
            max_tokens (int): Maximum token count (default: 2048, must be positive).

        Raises:
            ValueError: If max_tokens is not a positive integer.
        """
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("The 'max_tokens' must be a positive integer.")
        self.max_tokens = max_tokens

    def set_top_p(self, top_p=1):
        """
        Sets the top-p value for nucleus sampling.

        Parameters:
            top_p (float): Value controlling probability threshold for sampling (default: 1, range: 0 to 1).

        Raises:
            ValueError: If top_p is not within the range [0, 1].
        """
        if not (0 <= top_p <= 1):
            raise ValueError("The 'top_p' must be between 0 and 1.")
        self.top_p = top_p

    def set_frequency_penalty(self, frequency_penalty=0):
        """
        Sets the frequency penalty to control repetition in responses.

        Parameters:
            frequency_penalty (float): Penalty for repeated tokens (default: 0, range: -2 to 2).

        Raises:
            ValueError: If frequency_penalty is not within the range [-2, 2].
        """
        if not (-2 <= frequency_penalty <= 2):
            raise ValueError("The 'frequency_penalty' must be between -2 and 2.")
        self.frequency_penalty = frequency_penalty

    def set_presence_penalty(self, presence_penalty=0):
        """
        Sets the presence penalty to control token diversity in responses.

        Parameters:
            presence_penalty (float): Penalty for encouraging diversity (default: 0, range: -2 to 2).

        Raises:
            ValueError: If presence_penalty is not within the range [-2, 2].
        """
        if not (-2 <= presence_penalty <= 2):
            raise ValueError("The 'presence_penalty' must be between -2 and 2.")
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

    @staticmethod
    def validate(parameters):
        """
        Validates the parameters dictionary.

        Parameters:
            parameters (dict): The dictionary containing configuration parameters.

        Returns:
            dict: The validated parameters dictionary.

        Raises:
            ValueError: If any parameter is invalid.
        """
        if not isinstance(parameters, dict):
            raise ValueError("Parameters must be a dictionary.")

        # Mandatory key
        if "model" not in parameters or not isinstance(parameters["model"], str):
            raise ValueError("The 'model' key is mandatory and must be a non-empty string.")

        # Validate optional keys
        valid_keys = {
            "temperature": (0, 2),
            "max_tokens": (1, float('inf')),
            "top_p": (0, 1),
            "frequency_penalty": (-2, 2),
            "presence_penalty": (-2, 2)
        }
        for key, (min_val, max_val) in valid_keys.items():
            if key in parameters and not (min_val <= parameters[key] <= max_val):
                raise ValueError(f"The '{key}' must be between {min_val} and {max_val}.")

        return parameters
    
def chatgpt_submit(openai_client, chatgpt_parameters, chatgpt_messages):
    """
    Submits a request to the OpenAI API using the provided parameters and messages.

    Parameters:
        openai_client (object): An instance of the OpenAI API client to handle the request.
        chatgpt_parameters (object or dict): A ChatgptParameters object or a raw dictionary of parameters.
        chatgpt_messages (object or list): A ChatgptMessages object or a raw list of messages.

    Returns:
        response (dict): The response from the OpenAI API if successful.
        None: If an exception occurs during the API request or if inputs are invalid.

    Raises:
        ValueError: If the inputs or their derived values are invalid.
    """
    try:
        # Validate and retrieve parameters
        if isinstance(chatgpt_parameters, dict):
            parameters = Chatgpt_parameters.validate(chatgpt_parameters)
        elif isinstance(chatgpt_parameters, Chatgpt_parameters):
            parameters = Chatgpt_parameters.validate(chatgpt_parameters.get_parameters())
        else:
            raise ValueError("The 'chatgpt_parameters' argument must be a dictionary or a ChatgptParameters object.")

        # Validate and retrieve messages
        if isinstance(chatgpt_messages, list):
            messages = Chatgpt_messages.validate(chatgpt_messages)
        elif isinstance(chatgpt_messages, Chatgpt_messages):
            messages = Chatgpt_messages.validate(chatgpt_messages.get_messages())
        else:
            raise ValueError("The 'chatgpt_messages' argument must be a list or a ChatgptMessages object.")

        # Add messages to the parameters
        parameters["messages"] = messages

        # Send the request to the OpenAI API and get the response
        response = openai_client.chat.completions.create(**parameters)
        return response

    except OpenAI.error.OpenAIError as e:
        # Handle OpenAI-specific errors
        print(f"OpenAI API error: {e}")
    except ValueError as e:
        # Handle validation issues
        print(f"Input validation error: {e}")
    except Exception as e:
        # Handle unexpected errors
        print(f"An unexpected error occurred: {e}")

    return None

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

    def get_embeddings_batch(self, texts, model=0, max_workers=4):
        """
        Generates embeddings for a batch of texts using the specified model.

        Parameters:
            texts (list of str): A list of input texts to generate embeddings for.
            model (int): The model to use for embedding.
                         - 0: Small embedding model ("text-embedding-3-small").
                         - 1: Large embedding model ("text-embedding-3-large").
            max_workers (int): The maximum number of threads for parallel processing.

        Returns:
            list of list: A list of embeddings, where each embedding is a list of floats.
        """
        def generate_embedding(text):
            return self.get_embedding(text, model)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            embeddings = list(executor.map(generate_embedding, texts))
        return embeddings

    def save_embedding(self, embedding, save_path):
        """
        Saves an embedding to a file in JSON format.

        Parameters:
            embedding (list): The embedding to save.
            save_path (str): The file path where the embedding will be saved.
        """
        with open(save_path, "a") as f:
            f.write(json.dumps(embedding) + "\n")

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
            embedding_1 (list or np.ndarray): The first embedding as a list or numpy array of floats.
            embedding_2 (list or np.ndarray): The second embedding as a list or numpy array of floats.

        Returns:
            float: The cosine similarity between the two embeddings.

        Raises:
            ValueError: If the inputs are invalid (e.g., None, not the same dimension, or not numeric).
        """
        # Validate inputs
        if embedding_1 is None or embedding_2 is None:
            raise ValueError("Embedding inputs must not be None.")
        
        if not isinstance(embedding_1, (list, np.ndarray)) or not isinstance(embedding_2, (list, np.ndarray)):
            raise ValueError("Embeddings must be lists or numpy arrays.")
        
        embedding_1 = np.array(embedding_1)
        embedding_2 = np.array(embedding_2)

        if embedding_1.ndim != 1 or embedding_2.ndim != 1:
            raise ValueError("Embeddings must be 1-dimensional.")
        
        if embedding_1.shape[0] != embedding_2.shape[0]:
            raise ValueError("Embeddings must have the same dimension.")

        # Ensure embeddings contain numeric data
        if not np.issubdtype(embedding_1.dtype, np.number) or not np.issubdtype(embedding_2.dtype, np.number):
            raise ValueError("Embeddings must contain numeric values.")

        # Compute cosine similarity
        norm_1 = np.linalg.norm(embedding_1)
        norm_2 = np.linalg.norm(embedding_2)

        if norm_1 == 0 or norm_2 == 0:
            raise ValueError("Embeddings must not have zero magnitude.")

        cosine_similarity = np.dot(embedding_1, embedding_2) / (norm_1 * norm_2)
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
            request["messages"].append({"role":Message_roles.SYSTEM_ROLE_FLAG, "content": self.dynamic_system_messages[key]})

        # Append API call instructions to the request
        request["messages"].append({"role":Message_roles.SYSTEM_ROLE_FLAG, "content": "<API_CALL|(api_name)|(json for the API request)>."})
        request["messages"].append({"role":Message_roles.SYSTEM_ROLE_FLAG, "content": "APIs may respond from requests with their messages inside <API_RESPONSE> tags."})

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