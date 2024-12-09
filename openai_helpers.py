import os  # File system utilities, e.g., checking paths, creating directories
import logging
from urllib.parse import urlparse  # URL validation and parsing
import requests  # Making HTTP requests

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
            logger.error("The image URL must be a non-empty string.")
            raise ValueError("The image URL must be a non-empty string.")
        
        parsed_url = urlparse(image_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            logger.error("The image URL is not valid. "+image_url)
            raise ValueError("The image URL is not valid.")
        
        # Validate the save path
        if not isinstance(save_path, str) or not save_path.strip():
            logger.error("The save path must be a non-empty string. "+save_path)
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

        logger.info(f"Image successfully downloaded and saved to {save_path}")
        return True

    except requests.exceptions.RequestException as e:
        print(f"Failed to download the image: {e}")
    except ValueError as e:
        print(f"Validation error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return False