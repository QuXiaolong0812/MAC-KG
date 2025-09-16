import os
from openai import OpenAI
import json
from typing import List, Tuple
import time
import base64
from SelfUtils.Logger import Logger
from DataStructure.KGTriple import KGTriple, convert_json_list_to_triples
from SelfUtils.FileOperator import FileOperator

class VisualEnhancingAgent:
    """
    VisualEnhancingAgent is a class that enhances visual data by extracting features from images
    and generating triples based on those features.
    It uses OpenAI's API to process images and generate structured data.
    """

    def __init__(self, vea_i2d_client: OpenAI, vea_i2d_model_name: str, vea_d2t_client: OpenAI, vea_d2t_model_name: str, logger: Logger= None):
        """
        Initialize the VisualEnhancingAgent with an OpenAI client and model name.
        :param vea_i2d_client:
            The OpenAI client instance for image to description processing.
        :param vea_i2d_model_name:
            The name of the OpenAI model to use for image to description processing.
        :param vea_d2t_client:
            The OpenAI client instance for description to triples processing.
        :param vea_d2t_model_name:
            The name of the OpenAI model to use for description to triples processing.
        :param logger:
            An optional logger instance for logging messages.
        """
        self.vea_i2d_client = vea_i2d_client
        self.vea_i2d_model_name = vea_i2d_model_name
        self.vea_d2t_client = vea_d2t_client
        self.vea_d2t_model_name = vea_d2t_model_name
        self.logger = logger
        self.file_operator = FileOperator()
        self.system_prompt4image2desc = """
You are a helpful assistant that extracts description from images.
Next, you will be given an entity and an image related to the entity. 
Please combine the content in the picture and generate a rigorous and detailed description text around the entity from the perspectives of shape, color, appearance, etc.
"""
        self.system_prompt4desc2triples = """
You are a helpful assistant that extracts triples from descriptions.
Next, you are given an entity and a rigorous and detailed description text (there may be repeated text) about the entity's shape, color, appearance, etc. extracted from the entity's image. 
Please use the entity as the head node and combine the description text to generate triple data.
Be careful not to generate duplicate triplets.

Output Format:
Each triple should follow this structure:
{"head": <head entity>, "relation": <relation>, "tail": <tail entity>}
Output as a JSON object with a "triples" key containing a list of such triples.

POSITIVE EXAMPLE:
Input: 
{
  "head": "Leopard cat",
  "text": "The leopard cat is a small wild cat native to Asia. It inhabits forests, shrublands, and mountain regions. It feeds on rodents, birds, and other small animals. The leopard cat is mainly nocturnal and has excellent night vision."
}
Output:
{
"triples": [
{"head": "pandas", "relation": "has face color", "tail": "light brown"},
{"head": "pandas", "relation": "has eye markings color", "tail": "darker brown"},
{"head": "pandas", "relation": "has facial expression", "tail": "gentle"},
{"head": "pandas", "relation": "has facial expression", "tail": "expressive"},
{"head": "pandas", "relation": "has ear shape", "tail": "small and round"},
{"head": "pandas", "relation": "has ear color", "tail": "dark brown"},
{"head": "pandas", "relation": "has head fur color", "tail": "lighter"},
{"head": "pandas", "relation": "has eye demeanor", "tail": "curious"},
{"head": "pandas", "relation": "has eye demeanor", "tail": "calm"}
]
}
"""

    def generate_feature_description(self, entity: str, image_dir: str) -> Tuple[List[str], int, int, float]:
        """
        Generate a feature description for the given entity based on the provided image URL.
        :param entity: The entity for which to generate the description.
        :param image_dir: The location of the image related to the entity.
        :return: A tuple containing a list of descriptions, prompt tokens count, completion tokens count, and processing time.
        """
        images = self.file_operator.get_files_in_folder(image_dir)
        if not images:
            self.logger.warning(f"No images found in directory: {image_dir}")
            return [], 0, 0, 0.0
        descriptions = []
        start_time = time.time()
        total_prompt_tokens = 0
        total_completion_tokens = 0
        for image in images:
            # Read the image and encode it to base64
            with open(image, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode("utf-8")

            # Create the prompt for the model
            prompt = f"""
                    Generate a detailed description for the entity "{entity}" based on the image provided.
                    The description should include aspects such as shape, color, and appearance.
                    """
            try:
                response = self.vea_i2d_client.chat.completions.create(
                    model=self.vea_i2d_model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt4image2desc},
                        {"role": "user", "content": prompt},
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": f"Please describe the image of {entity}."},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/jpeg;base64,{image_data}"
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=0.7
                )
                description = response.choices[0].message.content.strip()
                descriptions.append(description)
                total_prompt_tokens += response.usage.prompt_tokens
                total_completion_tokens += response.usage.completion_tokens
            except Exception as e:
                self.logger.warning(f"Failed to generate description for {entity} with image {image}. Error: {e}")
                descriptions.append(f"Error generating description: {str(e)}")
        return descriptions, total_prompt_tokens, total_completion_tokens, time.time() - start_time

    def generate_feature_triples(self, entity: str, descriptions: List[str]) -> Tuple[List[KGTriple], int, int, float]:
        """
        Generate feature triples from the given entity and its descriptions.
        :param entity: The entity for which to generate triples.
        :param descriptions: A list of descriptions related to the entity.
        :return: A tuple containing a list of KGTriples, prompt tokens count, completion tokens count, and processing time.
        """
        triples = []
        start_time = time.time()
        total_prompt_tokens = 0
        total_completion_tokens = 0
        texts = " ".join(descriptions)
        prompt = f"""
                Extract triples from the following descriptions of the entity "{entity}":
                {texts}
                Please format each triple as a JSON object with keys "head", "relation", and "tail".
                """
        try:
            response = self.vea_d2t_client.chat.completions.create(
                model=self.vea_d2t_model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt4desc2triples},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            content = response.choices[0].message.content.strip()
            try:
                # Check if response contains JSON array
                if "[" in content and "]" in content:
                    json_str = content[content.find("["):content.rfind("]") + 1]
                    triples_data = json.loads(json_str)
                    triples = convert_json_list_to_triples(triples_data)
                    total_prompt_tokens += response.usage.prompt_tokens
                    total_completion_tokens += response.usage.completion_tokens
                else:
                    self.logger.warning("Response does not contain a valid JSON array.")
            except Exception as e:
                self.logger.warning(f"Failed to parse response as JSON: {e}")
                return [], 0, 0, time.time() - start_time
        except Exception as e:
            self.logger.warning(f"Feature triples generation failed. Error: {e}")
            return [], 0, 0, time.time() - start_time
        return triples, total_prompt_tokens, total_completion_tokens, time.time() - start_time