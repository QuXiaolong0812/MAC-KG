import os
import torch
from transformers import CLIPProcessor, CLIPModel
from typing import List, Tuple, Any
from SelfUtils.FileOperator import FileOperator
from SelfUtils.Logger import Logger

class Image2textMatchingAgent:
    """
    Image2textMatchingAgent is a class that matches images with text descriptions using a pre-trained CLIP model.
    It calculates the similarity between images in a specified folder and a given text description.
    """
    def __init__(self, IMA_model: str, logger: Logger, IMA_topK: int = 20):
        """
        Initialize the Image2textMatchingAgent with a pre-trained CLIP model.
        :param IMA_model: The name or path of the pre-trained CLIP model.
        :param IMA_topK: The number of top similar images to return (default is 20).
        """
        self.image_filtering_model = IMA_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained(self.image_filtering_model).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.image_filtering_model)
        self.file_operator = FileOperator()
        self.IMA_topK = IMA_topK
        self.logger = logger

    def calculate_image_similarity(self, image_dir: str, text_desc: str):
        """
        Calculate the similarity between images in a folder and a text description.

        :param image_dir: Directory containing images to be compared.
        :param text_desc: Text description to compare against the images.
        :return: List of tuples with image names and their similarity scores.
        """
        # Load images
        images, image_names = self.file_operator.load_images(image_dir)
        # Prepare inputs for the model
        inputs = self.processor(text=[text_desc] * len(images), images=images, return_tensors="pt", padding=True).to(self.device)
        # Forward pass through the model
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image  # Similarity scores for each image

            # Convert logits to probabilities
            scores = logits_per_image.softmax(dim=0).squeeze().detach().cpu().numpy().tolist()
            scores = [float(score) if isinstance(score, (int, float)) else float(score[0]) for score in scores]

        # Sort results by similarity score
        image_matching_results = sorted(zip(image_names, scores), key=lambda x: x[1], reverse=True)

        # Limit results to top K if specified
        if hasattr(self, 'IMA_topK') and self.IMA_topK > 0:
            image_matching_results = image_matching_results[:self.IMA_topK]

        for image_name, score in image_matching_results:
            self.logger.info(f"Image: {image_name}, Similarity Score: {score:.4f}")