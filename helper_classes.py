import logging

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
        self.messages=[]

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
                logger.error("Each message must be a dictionary. Invalid message: {message}")
                raise ValueError(f"Each message must be a dictionary. Invalid message: {message}")
            if "role" not in message or message["role"] not in valid_roles:
                logger.error(f"Each message must have a 'role' key with a value of 'system', 'assistant', or 'user'. Invalid message: {message}")
                raise ValueError(f"Each message must have a 'role' key with a value of 'system', 'assistant', or 'user'. Invalid message: {message}")
            if "content" not in message or not isinstance(message["content"], str):
                logger.error(f"Each message must have a 'content' key with a non-empty string value. Invalid message: {message}")
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
            logger.error("Chat completion parameters must be a dictionary.")
            raise ValueError("Parameters must be a dictionary.")

        # Mandatory key
        if "model" not in parameters or not isinstance(parameters["model"], str):
            logger.error("Chat completion parameters must have key model.")
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
                logger.error(f"The '{key}' must be between {min_val} and {max_val}.")
                raise ValueError(f"The '{key}' must be between {min_val} and {max_val}.")

        return parameters