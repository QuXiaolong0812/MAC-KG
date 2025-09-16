import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
from sklearn.cluster import KMeans
from tqdm import tqdm
import numpy as np
from collections import defaultdict
from SelfUtils.Logger import Logger
from SelfUtils.FileOperator import FileOperator
from DataStructure.KGTriple import KGTriple
from typing import List
import json

class ImageClusteringAgent:
    """
    ImageClusteringAgent is a class that clusters images in a specified folder using KMeans clustering.
    It extracts features from images using a pre-trained ResNet50 model and groups them into clusters.
    """
    def __init__(self, logger: Logger, ICA_clusters=3, store_dir='triples_store'):
        self.ICA_clusters = ICA_clusters
        self.clusters = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-1])
        self.logger = logger
        self.file_operator = FileOperator()
        self.store_dir = store_dir

    def load_images_and_extract_feature(self, image_folder: str) -> tuple[np.ndarray, list[str]]:
        """
        Load images from a folder and extract features using a pre-trained model.

        :param image_folder: Directory containing images to be clustered.
        :return: Tuple of features (numpy array) and image paths (list of strings).
        """
        image_paths = []
        features = []
        self.model.eval().to(self.device)
        for file in tqdm(os.listdir(image_folder)):
            if file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp')):
                path = os.path.join(image_folder, file)
                try:
                    image = Image.open(path).convert("RGB")
                    img_tensor = self.transform(image).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        feat = self.model(img_tensor).cpu().numpy().flatten()  # Output shape is (2048,)
                    features.append(feat)
                    image_paths.append(path)
                except Exception as e:
                    print(f"Error loading {file}: {e}")

        features = np.array(features)
        return features, image_paths

    def cluster_images(self, image_dir: str, head_id: int):
        """
        Cluster images in a folder using KMeans clustering.

        :param image_dir: Directory containing images to be clustered.
        :return: Dictionary mapping cluster labels to lists of image paths.
        """
        features, image_paths = self.load_images_and_extract_feature(image_dir)
        kmeans = KMeans(n_clusters=self.ICA_clusters, random_state=42)
        labels = kmeans.fit_predict(features)
        clustered_images = defaultdict(list)
        for label, path in zip(labels, image_paths):
            clustered_images[label].append(path)

        for cluster_id, images in clustered_images.items():
            self.logger.info(f"Cluster {cluster_id}: {', '.join(images)}")

        # Select the cluster with the largest number of images
        largest_cluster = max(clustered_images.values(), key=len)
        self.logger.info(f"Largest cluster selected with {len(largest_cluster)} images.")

        # Copy images from the largest cluster to the store directory
        # self.file_operator.copy_files_to_folder(largest_cluster, os.path.join(self.store_dir, 'images'))

        rename_list = []
        for idx, img_path in enumerate(largest_cluster):
            new_name = f"{head_id}_{idx}"
            rename_list.append(new_name)

        # Copy and rename files to the store directory
        self.file_operator.copy_and_rename_files(largest_cluster, os.path.join(self.store_dir, 'images'), rename_list)

        visual_triples = []
        for new_name in rename_list:
            visual_triples.append(KGTriple(head=head_id, relation='hasImage', tail=new_name, type_='visual'))
        self._save_visual_triples2id(visual_triples)


    # ---------------- Save triples2id ----------------

    def _save_visual_triples2id(self, triples: List[KGTriple]):
        """
        Save triples to visual_triples2id.jsonl.
        - If the file does not exist, create it.
        - If it exists, append new triples.
        - Skip already existing triples (avoid duplication).
        """
        os.makedirs(self.store_dir, exist_ok=True)
        path = os.path.join(self.store_dir, "visual_triples2id.jsonl")

        # Load existing triples (if any) into a set for fast lookup
        existing = set()
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        obj = json.loads(line.strip())
                        existing.add((obj["head"], obj["relation"], obj["tail"], obj["type"]))
                    except Exception:
                        continue

        # Append new triples, skipping duplicates
        with open(path, "a", encoding="utf-8") as f:
            for t in triples:
                head_id = t.head
                tail_id = t.tail
                rel_id = -1
                type_id = 1
                obj = (head_id, rel_id, tail_id, type_id)

                if obj not in existing:
                    f.write(json.dumps(
                        {"head": head_id, "relation": rel_id, "tail": tail_id, "type": type_id},
                        ensure_ascii=False
                    ) + "\n")
                    existing.add(obj)