import json
import random
from typing import List, Tuple
import time
import wikipedia
from SelfUtils.Logger import Logger
from DataStructure.KGTriple import KGTriple
from openai import OpenAI

class TailPruningAgent:
    """
    TailPruningAgent is a class that filters out tail entities from a list of triples based on their suitability for knowledge graph expansion.
    It uses OpenAI's API to process the input and generate a filtered list of tail entities.
    """

    def __init__(self, client: OpenAI, model_name: str, k_neighbors: int, logger: Logger = None):
        """
        Initialize the TailPruningAgent with an OpenAI client and model name.
        :param client:
            The OpenAI client instance to interact with the API.
        :param model_name:
            The name of the OpenAI model to use for processing.
        :param logger:
            An optional logger instance for logging messages.
        """
        self.client = client
        self.model_name = model_name
        self.logger = logger
        self.k_neighbors = k_neighbors
        self.system_prompt = """
You are the Tail Pruning Agent (TPA).
Given a list of tail entities extracted from (head, relation, tail) triples, your task is to select only those tail entities that are suitable for further expansion in a knowledge graph.
These entities must be concrete and specific, such as locations, organisms, people, or identifiable physical objects or events. Do not include abstract concepts (e.g., "freedom", "beauty", "effectiveness", "behavior") or vague/uncategorizable terms.

Input Format:
{
  "tail_entities": [
    "tropical rainforest",
    "hunting behavior",
    "Asia",
    "curiosity",
    "African elephant",
    "diet",
    "climate change",
    "Dr. Jane Smith"
  ]
}

Output Format:
Return a JSON object with a single key "expandable_entities" that contains a list of tail entities from the input only that are suitable for knowledge graph expansion.
{
  "expandable_entities": [
    "tropical rainforest",
    "Asia",
    "African elephant",
    "Dr. Jane Smith"
  ]
}

Filtering Guidelines:

✅ Include only tail entities that are:
- Specific locations (e.g., "Asia", "Amazon Basin")
- Biological species (e.g., "African elephant", "oak tree")
- Identifiable people (e.g., "Charles Darwin", "Dr. Jane Smith")
- Concrete objects or artifacts (e.g., "volcano", "carbon dioxide", "microscope")
- Events with clear identity (e.g., "Great Migration", "2008 Financial Crisis")
❌ Exclude:
- Abstract ideas (e.g., "freedom", "effectiveness")
- General concepts or categories (e.g., "behavior", "diet", "emotion")
-Adjectives or qualities used as nouns (e.g., "strength", "curiosity")
🚫 Do not invent or introduce any new entities.
✅ Only use the entities provided in the input list.

Ambiguity Handling:
- If an entity could be interpreted both concretely and abstractly, include it only if it refers to a specific real-world referent.
- When in doubt, exclude rather than guess.

POSITIVE EXAMPLE:
Input:
{
  "tail_entities": [
    "Siberian tiger", "ecosystem", "predation", "Russia", "aggression", "tiger"
  ]
}

Output:
{
  "expandable_entities": [
    "Siberian tiger", "Russia", "tiger"
  ]
}

NEGATIVE EXAMPLE (Don’t do this):
{
  "expandable_entities": [
    "ecosystem", "aggression", "predation"
  ]
}
"""

    def _filter_wikipedia_entries(self, entities: List[str]) -> List[str]:
        """
        Filter entities by checking their existence on Wikipedia.
        :param entities: A list of entity names to check.
        :return: A list of entities that have a corresponding Wikipedia page.
        """
        filtered_entities = []
        for entity in entities:
            try:
                wikipedia.summary(entity, sentences=1)  # 尝试抓取摘要
                filtered_entities.append(entity)
            except wikipedia.DisambiguationError as e:
                continue
            except wikipedia.PageError:
                continue
            except Exception as e:
                continue
        return filtered_entities


    def prune_tail_entities(self, triples: List[KGTriple]) -> Tuple[List[str], int, int, float]:
        """
        Prune the tail entities from the given triples based on their suitability for knowledge graph expansion.
        :param triples: A list of KGTriple objects.
        :return: A tuple containing a list of pruned tail entities,
                 prompt tokens count, completion tokens count, and processing time.
        """
        if not triples:
            self.logger.warning("No triples provided for pruning.")
            return [], 0, 0, 0.0

        # Extract unique tail entities from the triples
        tail_entities = list({triple.tail for triple in triples})
        if not tail_entities:
            self.logger.warning("No tail entities extracted from triples.")
            return [], 0, 0, 0.0

        start_time = time.time()
        entities_str = {"tail_entities": tail_entities}
        entities_string = json.dumps(entities_str, ensure_ascii=False)

        prompt = f"""
    You are the Tail Pruning Agent (TPA). Given the following tail entities, select only those that are suitable for further expansion in a knowledge graph.
    These entities must be concrete and specific, such as locations, organisms, people, or identifiable physical objects or events. Do not include abstract concepts or vague/uncategorizable terms.

    Input:
    {entities_string}
    """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            content = response.choices[0].message.content.strip()

            # Try to parse JSON entities from the response
            try:
                json_str = content[content.find("{"):content.rfind("}") + 1]
                result = json.loads(json_str)
                pruned_entities = result.get("expandable_entities", [])
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse JSON response: {content}")
                pruned_entities = []
            except KeyError:
                self.logger.warning(f"Response does not contain 'expandable_entities': {content}")
                pruned_entities = []
            except Exception as e:
                self.logger.warning(f"Unexpected error while parsing response: {e}")
                pruned_entities = []

            end_time = time.time()
            # Further filter entities by checking their existence on Wikipedia
            pruned_entities = self._filter_wikipedia_entries(pruned_entities)
            # Limit to k neighbors if necessary
            if len(pruned_entities) > self.k_neighbors:
                pruned_entities = random.sample(pruned_entities, self.k_neighbors)

            prompt_tokens = getattr(response.usage, "prompt_tokens", 0)
            completion_tokens = getattr(response.usage, "completion_tokens", 0)
            processing_time = end_time - start_time
            return pruned_entities, prompt_tokens, completion_tokens, processing_time

        except Exception as e:
            self.logger.warning(f"Interaction pruning failed. Error: {e}")
            return [], 0, 0, time.time() - start_time