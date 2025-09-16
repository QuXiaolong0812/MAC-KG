import os
from openai import OpenAI
import json

from DataStructure.KGTriple import KGTriple

from SelfUtils.Logger import Logger
from SelfUtils.ContentCrawler import ContentCrawler

from TextualAgents.CorpusExtractingAgent import CorpusExtractingAgent
from TextualAgents.TextualEnhancingAgent import TextualEnhancingAgent
from TextualAgents.StructureVerifyingAgent import StructureVerifyingAgent
from TextualAgents.TripleAligningAgent import TripleAligningAgent

from TextualAgents.TailPruningAgent import TailPruningAgent

from VisualAgents.ImageFilteringAgent import ImageFilteringAgent
from VisualAgents.VisualEnhancingAgent import VisualEnhancingAgent
from VisualAgents.Image2textMatchingAgent import Image2textMatchingAgent
from VisualAgents.ImageClusteringAgent import ImageClusteringAgent


class MAC_KG:
    """
    MAC-KG is a knowledge graph conflict detection agent that uses OpenAI's API to filter out conflicting triples.
    It processes a list of KGTriples and returns only those that are logically consistent.
    """

    def __init__(self, args):
        # General parameters
        self.start_entity = args.start_entity
        self.textual_base = args.textual_base
        self.visual_base = args.visual_base
        self.hop = args.hop
        self.max_neighbors = args.max_neighbors
        self.store_dir = args.store_dir

        # Agent module parameters
        self.IMA_topK = args.IMA_topK
        self.ICA_clusters = args.ICA_clusters

        # Ensure texts directory exists
        if not os.path.exists(self.textual_base):
            os.makedirs(self.textual_base)
        # Ensure image directory exists
        if not os.path.exists(self.visual_base):
            os.makedirs(self.visual_base)

        # ========== OpenAI client initialization ==========
        self.content_crawler_user_agent = args.content_crawler_user_agent

        # Initialize OpenAI client for Corpus Extracting Agent
        cea_client_kwargs = {"api_key": args.CEA_api_key}
        if args.CEA_url:
            cea_client_kwargs["base_url"] = args.CEA_url
        self.cea_client = OpenAI(**cea_client_kwargs)
        self.cea_model = args.CEA_model

        # Initialize OpenAI client for Textual Enhancing Agent
        tea_client_kwargs = {"api_key": args.TEA_api_key}
        if args.TEA_url:
            tea_client_kwargs["base_url"] = args.TEA_url
        self.tea_client = OpenAI(**tea_client_kwargs)
        self.tea_model = args.TEA_model

        # Initialize OpenAI client for Structure Verifying Agent
        sva_client_kwargs = {"api_key": args.SVA_api_key}
        if args.SVA_url:
            sva_client_kwargs["base_url"] = args.SVA_url
        self.sva_client = OpenAI(**sva_client_kwargs)
        self.sva_model = args.SVA_model

        # Initialize Triple Aligning Agent
        self.taa_model = args.TAA_model
        self.taa_threshold = args.TAA_threshold

        # Initialize OpenAI client for Tail Pruning Agent
        tpa_client_kwargs = {"api_key": args.TPA_api_key}
        if args.TPA_url:
            tpa_client_kwargs["base_url"] = args.TPA_url
        self.tpa_client = OpenAI(**tpa_client_kwargs)
        self.tpa_model = args.TPA_model

        # Initialize Image Filtering Agent
        self.ifa_model = args.IFA_model

        # Initialize OpenAI client for Visual Enhancing Agent Image to Description Module
        vea_i2d_client_kwargs = {"api_key": args.VEA_I2D_api_key}
        if args.VEA_I2D_url:
            vea_i2d_client_kwargs["base_url"] = args.VEA_I2D_url
        self.vea_i2d_client = OpenAI(**vea_i2d_client_kwargs)
        self.vea_i2d_model = args.VEA_I2D_model

        # Initialize OpenAI client for Visual Enhancing Agent Description to Triple Module
        vea_d2t_client_kwargs = {"api_key": args.VEA_D2T_api_key}
        if args.VEA_D2T_url:
            vea_d2t_client_kwargs["base_url"] = args.VEA_D2T_url
        self.vea_d2t_client = OpenAI(**vea_d2t_client_kwargs)
        self.vea_d2t_model = args.VEA_D2T_model

        # Initialize Image2text Matching Agent
        self.IMA_model = args.IMA_model

        # Logging and tracking
        self.logger = Logger(name="MAC-KG", log_to_file="mac_kg.log")

        # Initialize all agent classes
        self.content_crawler = ContentCrawler(self.content_crawler_user_agent)
        self.corpus_extracting_agent = CorpusExtractingAgent(
            client=self.cea_client,
            model_name=self.cea_model,
            logger=self.logger
        )
        self.textual_enhancing_agent = TextualEnhancingAgent(
            client=self.tea_client,
            model_name=self.tea_model,
            logger=self.logger
        )
        self.structure_verifying_agent = StructureVerifyingAgent(
            client=self.sva_client,
            model_name=self.sva_model,
            logger=self.logger
        )
        self.triple_aligning_agent = TripleAligningAgent(
            model_name=self.taa_model,
            threshold=self.taa_threshold,
            store_dir=self.store_dir,
            logger=self.logger
        )
        self.tail_pruning_agent = TailPruningAgent(
            client=self.tpa_client,
            model_name=self.tpa_model,
            k_neighbors=args.max_neighbors,
            logger=self.logger
        )
        self.image_filtering_agent = ImageFilteringAgent(
            model_name=self.ifa_model,
            logger=self.logger
        )
        self.visual_enhancing_agent = VisualEnhancingAgent(
            vea_i2d_client=self.vea_i2d_client,
            vea_i2d_model_name=self.vea_i2d_model,
            vea_d2t_client=self.vea_d2t_client,
            vea_d2t_model_name=self.vea_d2t_model,
            logger=self.logger
        )
        self.image2text_matching_agent = Image2textMatchingAgent(
            IMA_model=self.IMA_model,
            logger=self.logger,
            IMA_topK=self.IMA_topK,
        )
        self.image_clustering_agent = ImageClusteringAgent(
            logger=self.logger,
            ICA_clusters=self.ICA_clusters,
            store_dir=self.store_dir
        )

        # Internally stored knowledge graph representation
        self.knowledge_graph = {
            "entities": set(),  # set of string IDs or entity names
            "relations": set(),  # set of string relation names
            "images": set(),  # set of image file paths
            "triples": []  # list of KnowledgeTriple objects
        }


    def mmkg_automatic_construction(self):
        """
        Automatically construct the multi-modal knowledge graph by crawling content and images from the start entity.
        """
        current_entities = [self.start_entity]
        for hop in range(self.hop):
            self.logger.info(f"============================= Starting Hop {hop + 1} =============================")
            next_entities = []
            for entity in current_entities:
                pruned_entities = self.mmkg_automatic_construction_by_step(entity)
                next_entities.extend(pruned_entities)
            # Remove duplicates for the next hop
            current_entities = list(set(next_entities))
            self.logger.info(f"Entities for next hop: {current_entities}")
        self.logger.info("MAC-KG automatic construction completed.")


    def mmkg_automatic_construction_by_step(self, current_entity):
        """
        Automatically construct the multi-modal knowledge graph by crawling content and images from the current entity.
        """
        self.logger.info(f"Starting MAC-KG automatic construction for the current entity: {current_entity}")
        try:
            image_dir = self.visual_base + '/' + current_entity
            texts_dir = self.textual_base + '/' + current_entity + '.txt'
            self.logger.info(f"Corpus directory: {image_dir}")
            self.logger.info(f"Image directory: {texts_dir}")

            # Crawl content and images
            self.logger.info("============================= Content Crawler =============================")
            # Crawl corpus from Wikipedia
            self.content_crawler.crawl_corpus(current_entity, output_file=texts_dir)
            self.logger.info(f"Corpus for {current_entity} has been crawled and saved.")

            # Crawl images from Wikipedia
            self.content_crawler.crawl_images(current_entity, output_dir=image_dir)
            self.logger.info(f"Images for {current_entity} have been crawled and saved.")

            # Extract triples from the corpus using the Corpus Extracting Agent
            self.logger.info("============================= Corpus Extracting Agent =============================")
            cea_triples, cea_prompt_tokens, cea_completion_tokens, cea_processing_time = self.corpus_extracting_agent.extracting_triples(
                head=current_entity,
                text_dir=texts_dir
            )
            self.logger.info(f"Extracted {len(cea_triples)} triples from the corpus for {current_entity}.")
            self.logger.info(f"CEA Prompt Tokens: {cea_prompt_tokens}, Completion Tokens: {cea_completion_tokens}, Processing Time: {cea_processing_time:.2f} seconds")
            desc_triples, desc_prompt_tokens, desc_completion_tokens, desc_processing_time = self.corpus_extracting_agent.classify_desc_triples(
                triples=cea_triples
            )
            self.logger.info(f"Classified {len(desc_triples)} descriptive triples for {current_entity}.")
            self.logger.info(f"Description Classification Prompt Tokens: {desc_prompt_tokens}, Completion Tokens: {desc_completion_tokens}, Processing Time: {desc_processing_time:.2f} seconds")

            # Filter images using the Image Filtering Agent
            self.logger.info("============================= Image Filtering Agent =============================")
            self.image_filtering_agent.filter_images(
                image_dir=image_dir,
                entity_name=current_entity
            )
            self.logger.info(f"Images for {current_entity} have been filtered and saved.")

            # Enhance triples to generate description text using the Textual Enhancing Agent
            self.logger.info("============================= Textual Enhancing Agent =============================")
            trp_description, trp_prompt_tokens, trp_completion_tokens, trp_processing_time = self.textual_enhancing_agent.verbalize_triples(
                head=current_entity,
                triples=desc_triples
            )
            self.logger.info(f"Generated description text for {current_entity}: {trp_description}")
            self.logger.info(f"Textual Enhancing Prompt Tokens: {trp_prompt_tokens}, Completion Tokens: {trp_completion_tokens}, Processing Time: {trp_processing_time:.2f} seconds")

            # Generate feature descriptions from images using the Visual Enhancing Agent
            self.logger.info("============================= Visual Enhancing Agent - Feature Descriptions =============================")
            feature_descriptions, vea_prompt_tokens, vea_completion_tokens, vea_processing_time = self.visual_enhancing_agent.generate_feature_description(
                entity=current_entity,
                image_dir=image_dir
            )
            self.logger.info(f"Generated {len(feature_descriptions)} feature descriptions for {current_entity}.")
            self.logger.info(f"Visual Enhancing Prompt Tokens: {vea_prompt_tokens}, Completion Tokens: {vea_completion_tokens}, Processing Time: {vea_processing_time:.2f} seconds")
            # Generate feature triples from the feature descriptions using the Visual Enhancing Agent
            self.logger.info("============================= Visual Enhancing Agent - Feature Triples =============================")
            vea_triples, vea_trp_prompt_tokens, vea_trp_completion_tokens, vea_trp_processing_time = self.visual_enhancing_agent.generate_feature_triples(
                entity=current_entity,
                descriptions=feature_descriptions
            )
            self.logger.info(f"Generated {len(vea_triples)} feature triples for {current_entity}.")
            self.logger.info(f"Visual Enhancing Feature Triples Prompt Tokens: {vea_trp_prompt_tokens}, Completion Tokens: {vea_trp_completion_tokens}, Processing Time: {vea_trp_processing_time:.2f} seconds")

            # Combine textual triples and feature triples
            all_textual_triples = cea_triples + vea_triples

            # pure textual channel
            # Verify the structure of the triples using the Structure Verifying Agent
            self.logger.info("============================= Structure Verifying Agent =============================")
            sva_triples, sva_prompt_tokens, sva_completion_tokens, sva_processing_time = self.structure_verifying_agent.verify_triples(
                triples=all_textual_triples
            )
            self.logger.info(f"Verified {len(sva_triples)} triples for {current_entity}.")
            self.logger.info(f"Structure Verifying Prompt Tokens: {sva_prompt_tokens}, Completion Tokens: {sva_completion_tokens}, Processing Time: {sva_processing_time:.2f} seconds")

            # Align the triples using the Triple Aligning Agent
            self.logger.info("============================= Triple Aligning Agent =============================")
            taa_triples, current_head_id, taa_processing_time = self.triple_aligning_agent.align_triples(
                triples=sva_triples
            )
            # Attention: The taa_triples are the final textual triples
            self.logger.info(f"Aligned {len(taa_triples)} triples for {current_entity}.")
            self.logger.info(f"Triple Aligning Prompt Tokens: 0, Completion Tokens: 0, Processing Time: {taa_processing_time:.2f} seconds, Head ID: {current_head_id}")

            # Prune the triples using the Tail Pruning Agent
            self.logger.info("============================= Tail Pruning Agent =============================")
            pruned_entities, tpa_prompt_tokens, tpa_completion_tokens, tpa_processing_time = self.tail_pruning_agent.prune_tail_entities(
                triples=taa_triples
            )
            # Attention: The pruned_entities are the final entities after pruning, is using for next hop crawling!
            self.logger.info(f"Pruned to {len(pruned_entities)} entities for {current_entity}.")


            # Pure visual channel
            # Image-text matching using the Image2text Matching Agent
            self.logger.info("============================= Image2Text Matching Agent =============================")
            self.image2text_matching_agent.calculate_image_similarity(
                image_dir=image_dir,
                text_desc=trp_description
            )


            # Cluster images using the Image Clustering Agent
            self.logger.info("============================= Image Clustering Agent =============================")
            self.image_clustering_agent.cluster_images(
                image_dir=image_dir,
                head_id=current_head_id  # Placeholder, replace with actual head_id if available
            )


            self.logger.info(f"MAC-KG automatic construction for {current_entity} step completed.")
            # return pruned_entities
            return []  # Placeholder, replace with actual pruned entities when the code is uncommented

        except Exception as e:
            self.logger.error(f"An error occurred during MAC-KG automatic construction: {e}")
            raise e


    def get_knowledge_graph(self):
        entity2id_path = os.path.join(self.store_dir, "entity2id.jsonl")
        id2entity = {}
        with open(entity2id_path, 'r', encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        id2entity[obj["id"]] = obj["name"]
                        if "name" in obj:
                            self.knowledge_graph.get("entities").add(obj["name"])
                    except json.JSONDecodeError:
                        continue

        relation2id_path = os.path.join(self.store_dir, "relation2id.jsonl")
        id2relation = {}
        with open(relation2id_path, 'r', encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        id2relation[obj["id"]] = obj["name"]
                        if "name" in obj:
                            self.knowledge_graph.get("relations").add(obj["name"])
                    except json.JSONDecodeError:
                        continue

        visual_triples2id_path = os.path.join(self.store_dir, "visual_triples2id.jsonl")
        textual_triples2id_path = os.path.join(self.store_dir, "textual_triples2id.jsonl")

        for path in [visual_triples2id_path, textual_triples2id_path]:
            with open(path, 'r', encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            obj = json.loads(line)
                            if all(k in obj for k in ("head", "relation", "tail", "type")):
                                head = id2entity[obj["head"]]
                                relation = ''
                                tail = obj["tail"]
                                type_ = "textual" if obj["type"] == 1 else "visual"
                                if type_ == "visual":
                                    relation = "hasImage"
                                    self.knowledge_graph.get("images").add(obj["tail"])
                                elif type_ == "textual":
                                    relation = id2relation[obj["relation"]]
                                    tail = id2entity[obj["tail"]]
                                triple = KGTriple(head=head, relation=relation, tail=tail, type_=type_)
                                self.knowledge_graph.get("triples").append(triple)
                        except json.JSONDecodeError:
                            continue
        return self.knowledge_graph

    def summarize_constructed_mmkg(self):
        """
        Summarize the constructed multi-modal knowledge graph.
        """
        self.logger.info("Summarizing the constructed multi-modal knowledge graph...")
        kg = self.get_knowledge_graph()
        num_entities = len(kg.get("entities"))
        num_relations = len(kg.get("relations"))
        num_images = len(kg.get("images"))
        num_triples = len(kg.get("triples"))
        self.logger.info(f"Number of entities: {num_entities}")
        self.logger.info(f"Number of relations: {num_relations}")
        self.logger.info(f"Number of images: {num_images}")
        self.logger.info(f"Number of triples: {num_triples}")
        self.logger.info("Summary completed.")