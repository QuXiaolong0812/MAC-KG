import os
import requests
from urllib.parse import urlparse
import wikipedia

from SelfUtils.Logger import Logger
from SelfUtils.FileOperator import FileOperator

class ContentCrawler:
    def __init__(self, user_agent: str = None, logger: Logger = None):
        """
        Initialize the ContentCrawler with a user agent.
        :param user_agent: Custom User-Agent header for requests. If None, a default will be used.
        """
        self.user_agent = user_agent or 'WikiImageDownloader/1.0 (your_email@example.com)'
        self.logger = logger or Logger(name="ContentCrawler")
        self.file_operator = FileOperator()


    def crawl_corpus(self, entity_name: str, output_file: str = 'wiki_corpus.txt'):
        """
        Crawl the Wikipedia page for the specified entity and return its content.
        :param entity_name: The name of the Wikipedia page to crawl.
        :param output_file: The file where the content will be saved. Defaults to 'wiki_corpus.txt'.
        :return: The content of the Wikipedia page.
        """
        try:
            # Search and get page content
            summary = wikipedia.summary(entity_name, sentences=10)  # Get the first 5 sentences of the summary
            content = wikipedia.page(entity_name).content  # Get the full content
            title = wikipedia.page(entity_name).title

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Title: {title}\n\nSummary:\n{summary}\n\nContent:\n{content}")

            self.logger.info(f"Page '{title}' has been saved to file: {output_file}")

        except wikipedia.exceptions.DisambiguationError as e:
            self.logger.warning("❗ Disambiguation error, please choose a more specific keyword, e.g.:", e.options[:5])
        except wikipedia.exceptions.PageError:
            self.logger.warning("❌ Page not found.")
        except Exception as e:
            self.logger.warning(f"An error occurred: {e}")


    def crawl_images(self, entity_name: str, output_dir: str = 'wiki_images'):
        """
        Download images from the specified Wikipedia page and save them to the output directory.
        :param entity_name: The name of the Wikipedia page to download images from.
        :param output_dir: The directory where images will be saved. Defaults to 'wiki_images'.
        :return: None
        """
        try:
            # Get the Wikipedia page
            page = wikipedia.page(entity_name)
            # Create the output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            # Check if the page has images
            if not page.images:
                self.logger.error(f"No images found on the page '{entity_name}'.")
                return
            self.logger.info(f"Found {len(page.images)} images, starting download...")
            headers = {
                'User-Agent': self.user_agent
            }
            download_count = 0
            # Download each image
            for i, image_url in enumerate(page.images):
                try:
                    # Parse URL to get filename
                    filename = os.path.basename(urlparse(image_url).path)
                    # Add a default extension if none exists
                    if not os.path.splitext(filename)[1]:
                        filename += '.jpg'

                    # Construct save path
                    save_path = os.path.join(output_dir, filename)

                    # Send request to download the image
                    response = requests.get(image_url, headers=headers, stream=True, timeout=10)
                    response.raise_for_status()

                    # Save the image
                    with open(save_path, 'wb') as f:
                        for chunk in response.iter_content(1024):
                            f.write(chunk)
                    download_count += 1
                except Exception as e:
                    self.logger.warning(f"({i + 1}/{len(page.images)}) Failed to download: {image_url} - {str(e)}")
            self.file_operator.delete_invalid_files(output_dir, keep_extensions=['.jpg', '.jpeg', '.png'])
            self.file_operator.batch_rename_files(output_dir, prefix=f'{entity_name}_', start_index=1)
            self.logger.info(f"Totally {download_count} images download completed, saved in: {os.path.abspath(output_dir)}")
        except wikipedia.exceptions.PageError:
            self.logger.error(f"Page not found: '{entity_name}'")
        except wikipedia.exceptions.DisambiguationError as e:
            self.logger.error(f"Disambiguation error for '{entity_name}'. Possible options: {e.options}")
        except Exception as e:
            self.logger.error(f"An error occurred while downloading images for '{entity_name}': {e}")