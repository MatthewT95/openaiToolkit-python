# OpenAI-specific imports
from openai import OpenAI  # OpenAI client for interacting with the API
import tiktoken  # Tokenization library for token counting (if needed)
import logging
from concurrent.futures import ThreadPoolExecutor  # For parallel processing
import re  # Regular expressions for pattern matching
import json  # JSON handling for API responses, requests, and file operations
import numpy as np  # Mathematical operations, e.g., cosine similarity
import logging
from openAiToolkit.helper_functions import download_img
from openAiToolkit.helper_classes import *

# Configure the logging
logging.basicConfig(
    level=logging.INFO,  # Set default logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Define log format
    handlers=[
         logging.FileHandler("app.log"),  # Log to a file
    ]
)

# Create a logger instance
logger = logging.getLogger(__name__)

class Text_embedding_client:
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
            logger.error("Embedding inputs must not be None.")
            raise ValueError("Embedding inputs must not be None.")
        
        if not isinstance(embedding_1, (list, np.ndarray)) or not isinstance(embedding_2, (list, np.ndarray)):
            logger.error("Embeddings must be lists or numpy arrays.")
            raise ValueError("Embeddings must be lists or numpy arrays.")
        
        embedding_1 = np.array(embedding_1)
        embedding_2 = np.array(embedding_2)

        if embedding_1.ndim != 1 or embedding_2.ndim != 1:
            logger.error("Embeddings must be 1-dimensional.")
            raise ValueError("Embeddings must be 1-dimensional.")
        
        if embedding_1.shape[0] != embedding_2.shape[0]:
            logger.error("Embeddings must have the same dimension.")
            raise ValueError("Embeddings must have the same dimension.")

        # Ensure embeddings contain numeric data
        if not np.issubdtype(embedding_1.dtype, np.number) or not np.issubdtype(embedding_2.dtype, np.number):
            logger.error("Embeddings must contain numeric values.")
            raise ValueError("Embeddings must contain numeric values.")

        # Compute cosine similarity
        norm_1 = np.linalg.norm(embedding_1)
        norm_2 = np.linalg.norm(embedding_2)

        if norm_1 == 0 or norm_2 == 0:
            logger.error("Embeddings must not have zero magnitude.")
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
        self.input_tokens_max=64
        self.max_token_enforcement=2 # 0 Don't enforce, 1 Return error message, 2 Trim message

    def set_input_token_max(self,max_tokens):
        self.input_tokens_max=max_tokens
    def set_input_token_max_enforcement(self,method):
        if method in range(2):
            self.max_token_enforcement=method
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

        tokenizer=tiktoken.encoding_for_model(self.parameters.model)
        user_tokenized_message=tokenizer.encode(user_message)
        if len(user_tokenized_message) > self.input_tokens_max and not self.max_token_enforcement==0:
            if self.max_token_enforcement == 1:
                logger.warning(f"Max input tokens exceeded. You used {len(user_tokenized_message)} tokens and the max is {self.input_tokens_max}.")
                return f"Max input tokens exceeded. You used {len(user_tokenized_message)} tokens and the max is {self.input_tokens_max}.",""
            elif self.max_token_enforcement == 2:
                logger.warning(f"Max input tokens exceeded. You used {len(user_tokenized_message)} tokens and the max is {self.input_tokens_max}.")
                logger.warning(f"Message was trimmed to max token length.")
                user_message=tokenizer.decode(user_tokenized_message[:self.input_tokens_max])
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