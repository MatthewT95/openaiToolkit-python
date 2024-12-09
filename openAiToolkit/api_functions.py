import logging
from openAiToolkit.helper_classes import *
from openai import OpenAI  # OpenAI client for interacting with the API

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
            logger.error("The 'chatgpt_parameters' argument must be a dictionary or a ChatgptParameters object.")
            raise ValueError("The 'chatgpt_parameters' argument must be a dictionary or a ChatgptParameters object.")

        # Validate and retrieve messages
        if isinstance(chatgpt_messages, list):
            messages = Chatgpt_messages.validate(chatgpt_messages)
        elif isinstance(chatgpt_messages, Chatgpt_messages):
            messages = Chatgpt_messages.validate(chatgpt_messages.get_messages())
        else:
            logger.error("The 'chatgpt_messages' argument must be a list or a ChatgptMessages object.")
            raise ValueError("The 'chatgpt_messages' argument must be a list or a ChatgptMessages object.")

        # Add messages to the parameters
        parameters["messages"] = messages

        # Send the request to the OpenAI API and get the response
        response = openai_client.chat.completions.create(**parameters)
        return response

    except OpenAI.error.OpenAIError as e:
        # Handle OpenAI-specific errors
        logger.error(f"OpenAI API error: {e}")
        print(f"OpenAI API error: {e}")
    except ValueError as e:
        # Handle validation issues
        logger.error(f"Input validation error: {e}")
        print(f"Input validation error: {e}")
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"An unexpected error occurred: {e}")
        print(f"An unexpected error occurred: {e}")

    return None