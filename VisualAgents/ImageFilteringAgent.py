import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from SelfUtils.Logger import Logger
from typing import List
from datetime import datetime
from SelfUtils.FileOperator import FileOperator

class ImageFilteringAgent:
    """
    ImageFilteringAgent uses a CLIP model to filter images based on whether they contain a specified species.
    It processes images in a given directory and retains only those that match the species name.
    """
    def __init__(self, model_name: str, logger: Logger = None):
        self.image_filtering_model = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(self.image_filtering_model).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.image_filtering_model)
        self.logger = logger if logger else Logger(name="ImageFilteringAgent")
        self.file_operator = FileOperator()

    def filter_images(self, image_dir: str, entity_name: str):
        """
        Filter images in the specified directory based on whether they contain the specified species.
        :param images_dir: Directory containing images to be filtered.
        :param species_name: The species name to filter images by.
        :return: List of filtered image paths.
        """

        # === generate text prompts ===
        text_prompts = [f"a photo of a {entity_name}", f"not a {entity_name}"]
        text_inputs = self.processor(text=text_prompts, return_tensors="pt", padding=True).to(self.device)

        # === get all image paths ===
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        # === handle each image ===
        self.logger.info(f"Processing {len(image_files)} images to filter by: {entity_name}")

        filtered_result = []
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        for image_file in tqdm(image_files, desc=f"{current_time} - MAC-KG - INFO - Filtering images: "):
            image_path = os.path.join(image_dir, image_file)
            image = Image.open(image_path).convert("RGB")

            # Process pictures and text together
            inputs = self.processor(images=image, text=text_prompts, return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_per_image = outputs.logits_per_image  # shape: [1, 2]
                probs = logits_per_image.softmax(dim=1).squeeze()

            result = {
                "image": image_file,
                "is_species_prob": probs[0].item(),
                "not_species_prob": probs[1].item(),
                "predicted": 1 if probs[0] > probs[1] else 0
            }

            filtered_result.append(result)

        self.filter_and_save(image_dir, filtered_result)

    def filter_and_save(self, images_dir: str, filtered_result: List[dict]) -> None:
        """
        Filter images based on the results and save the filtered images to a new directory.
        :param images_dir: Directory containing original images.
        :param filtered_result: List of filtering results.
        :return: Path to the directory containing filtered images.
        """
        saved_images = [item['image'] for item in filtered_result if item['predicted'] == 1]
        filtered_images = [item['image'] for item in filtered_result if item['predicted'] == 0]

        self.logger.info(f"Filtered {len(filtered_images)} images that do not contain the species.")
        self.file_operator.remove_files_not_in_list(images_dir, saved_images)