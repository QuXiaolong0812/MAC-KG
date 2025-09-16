import json
from openai import OpenAI
from typing import List, Tuple
import time
from SelfUtils.Logger import Logger
from DataStructure.KGTriple import KGTriple, convert_json_list_to_triples
from SelfUtils.FileOperator import FileOperator

class CorpusExtractingAgent:
    """
    Corpus Extracting Agent for MAC-KG
    This agent extracts textual triples from a given text using OpenAI's API.
    """

    def __init__(self, client: OpenAI, model_name: str, logger: Logger):
        """
        Initialize the Corpus Extracting Agent.

        Args:
            client: OpenAI client instance
            model_name: LLM model identifier
            logger: Logger instance for logging messages
        """
        self.client = client
        self.model_name = model_name
        self.logger = logger
        self.file_operator = FileOperator()
        self.system_prompt = """
You are the Corpus Extracting Agent (CEA). 
Given a head entity and a block of text, extract a list of meaningful (head, relation, tail) triples that describe factual knowledge related to the head entity. This is suitable for constructing open-domain or domain-specific knowledge graphs (e.g., in wildlife, geography, general science, etc.).

Output Format:
Each triple should follow this structure:
{"head": <head entity>, "relation": <relation>, "tail": <tail entity>}
Output as a JSON object with a "triples" key containing a list of such triples.

Extraction Guidelines:
1. The head entity is explicitly provided. Only extract triples where this head is the subject.
2. The relation must express a clear, meaningful connection (e.g., "inhabits", "feeds on", "is a type of", "has ability", "active during").
3. Tail entities can be locations, categories, other species, time periods, attributes, etc.
4. Do not extract vague or grammatically fragmented triples (e.g., avoid using “often”, “strong”, or “yes” as tails).
5. Support multi-hop facts and compositional mentions.
6. If multiple candidate relations or tails exist, select the most specific one.
7. If no valid triples can be extracted, return an empty list.

Ambiguity Handling:
1. If multiple valid tail candidates are possible, extract the most informative or concrete one.
2. If the text is ambiguous or lacks enough context, skip that triple rather than guessing.

POSITIVE EXAMPLE:
Input: 
{
  "head": "Leopard cat",
  "text": "The leopard cat is a small wild cat native to Asia. It inhabits forests, shrublands, and mountain regions. It feeds on rodents, birds, and other small animals. The leopard cat is mainly nocturnal and has excellent night vision."
}
Output:
{
"triples": [
{"head": "Leopard cat", "relation": "is a type of", "tail": "wild cat"},
{"head": "Leopard cat", "relation": "native to", "tail": "Asia"},
{"head": "Leopard cat", "relation": "inhabits", "tail": "forests"},
{"head": "Leopard cat", "relation": "inhabits", "tail": "shrublands"},
{"head": "Leopard cat", "relation": "inhabits", "tail": "mountain regions"},
{"head": "Leopard cat", "relation": "feeds on", "tail": "rodents"},
{"head": "Leopard cat", "relation": "feeds on", "tail": "birds"},
{"head": "Leopard cat", "relation": "active during", "tail": "night"},
{"head": "Leopard cat", "relation": "has ability", "tail": "night vision"}
]
}

NEGATIVE EXAMPLE:
Input:
{
"head_entity": "Leopard cat",
"text": "The leopard cat is a small animal. It lives in Asia. It is active often and has strong sight."
}
Incorrect Output:
{
"triples": [
{"head": "Leopard cat", "relation": "is", "tail": "small"},
{"head": "Leopard cat", "relation": "active", "tail": "often"},
{"head": "Leopard cat", "relation": "has", "tail": "strong"}
]
}
Why it's wrong:
1. Relations like "is", "active", "has" are too vague or grammatically incomplete.
2. Tails like "small", "often", and "strong" are not informative or well-formed entities.
"""
        self.system_prompt4classify_triple = """
You are the Appearance Triple Filtering Expert.
Given a list of factual (head, relation, tail) triples where all triples share the same head entity, your task is to identify and return only those triples that describe the head entity's visual appearance, specifically in terms of shape, color, and overall external features.

Output Format:
Return a JSON object with a single key "triples" containing the filtered list of appearance-related triples.
Each triple must follow this structure:
{"head": <head entity>, "relation": <relation>, "tail": <tail entity>}

Filtering Guidelines:

✅ Only include triples that describe the visual characteristics of the head entity:
- Shape-related: e.g., "has shape", "is shaped like", "has body form", etc.
- Color-related: e.g., "has color", "is colored", "has markings", "has stripes", etc.
- General appearance: e.g., "has appearance", "has texture", "has surface", "has pattern", "has fur type", etc.
❌ Exclude any triples that describe:
- Behavior, habitat, diet, taxonomy, temporal activity, abilities, or internal organs.
- Abstract or vague features not directly visible in physical appearance.
✅ Relations should clearly imply external, observable traits.
✅ Tail entities must be visually descriptive terms (e.g., "brown fur", "striped body", "oval shape", "rough skin").
❌ Do not include triples with non-visual tails (e.g., "nocturnal", "Asia", "mammal", "feeds on insects").

Ambiguity Handling:
- When unsure, only keep triples where the relation or tail clearly implies a visual trait.
- If none of the triples qualify, return an empty list: "triples": [].

POSITIVE EXAMPLE:
Input:
[
{"head": "Leopard cat", "relation": "has fur color", "tail": "brown with black spots"},
{"head": "Leopard cat", "relation": "inhabits", "tail": "mountain forests"},
{"head": "Leopard cat", "relation": "has body shape", "tail": "slender"},
{"head": "Leopard cat", "relation": "feeds on", "tail": "rodents"}
]
Output:
{
  "triples": [
    {"head": "Leopard cat", "relation": "has fur color", "tail": "brown with black spots"},
    {"head": "Leopard cat", "relation": "has body shape", "tail": "slender"}
  ]
}

NEGATIVE EXAMPLE:
Input:
[
{"head": "Leopard cat", "relation": "native to", "tail": "Asia"},
{"head": "Leopard cat", "relation": "is a type of", "tail": "wild cat"},
{"head": "Leopard cat", "relation": "has ability", "tail": "night vision"}
]
Output:
{
  "triples": []
}
"""

    def extracting_triples(self, head: str, text_dir: str) -> Tuple[List[KGTriple], int, int, float]:
        """
        Extract triples from the given text using OpenAI's API.

        Args:
            head: The head entity to extract triples for.
            text_dir: The block of text directory to extract triples from.

        Returns:
            A tuple containing:
            - List of extracted KGTriples
            - Number of prompt tokens used
            - Number of completion tokens generated
            - Processing time in seconds
        """
        text = self.file_operator.read_file(text_dir)
        self.logger.info(f"Extracting triples for head entity: {head}")
        # Prepare the prompt
        prompt = f"""
Extract triples from the given text based on the given head entity.
For each triple:
1. The head entity is provided as "{head}".
2. Extract the relation and tail entity from the text.
3. Format each triple as a JSON object with keys "head", "relation", and "tail".

Head Entity: {head}
Text: {text}
"""
        start_time = time.time()
        try:
            # Call OpenAI API to generate triples
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": self.system_prompt},
                          {"role": "user", "content": prompt}],
                temperature=0.1,
            )
            # Parse the response
            content = response.choices[0].message.content.strip()

            # Convert JSON string to list of KGTriples
            try:
                # Check if response contains JSON array
                if "[" in content and "]" in content:
                    json_str = content[content.find("["):content.rfind("]") + 1]
                    triples_data = json.loads(json_str)

                    prompt_tokens = response.usage.prompt_tokens
                    completion_tokens = response.usage.completion_tokens
                    processing_time = time.time() - start_time
                    triples = convert_json_list_to_triples(triples_data)
                    return triples, prompt_tokens, completion_tokens, processing_time
                else:
                    self.logger.warning("Response does not contain a valid JSON array.")
                    return [], 0, 0, time.time() - start_time
            except Exception as e:
                self.logger.warning(f"Failed to parse response as JSON: {e}")
                return [], 0, 0, time.time() - start_time
        except Exception as e:
            self.logger.warning(f"Entity extraction failed. Error: {e}")
            return [], 0, 0, time.time() - start_time  # Return an empty list for now

    def classify_desc_triples(self, triples: List[KGTriple]) -> tuple[List[KGTriple], int, int, float]:
        """
        Classify triples into descriptive text.

        Args:
            triples: List of KGTriples to classify.

        Returns:
            A tuple containing:
            - List of classified KGTriples
            - Number of prompt tokens used
            - Number of completion tokens generated
            - Processing time in seconds
        """
        if not triples:
            return [], 0, 0, 0.0

        # Convert triples to JSON format for the prompt
        triples_list = [{"head": triple.head, "relation": triple.relation, "tail": triple.tail} for triple in triples]

        prompt = f"""
Classify the following triples to identify those that describe the visual appearance of the head entity.
The triples are provided in JSON format. Only include those that relate to shape, color, and overall external features.
Input triples:
{json.dumps(triples_list, indent=2)}
"""

        start_time = time.time()
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self.system_prompt4classify_triple},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0
        )

        content = response.choices[0].message.content
        try:
            # Check if response contains JSON array
            if "[" in content and "]" in content:
                json_str = content[content.find("["):content.rfind("]") + 1]
                triples_data = json.loads(json_str)
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                processing_time = time.time() - start_time
                triples = convert_json_list_to_triples(triples_data)
                return triples, prompt_tokens, completion_tokens, processing_time
            else:
                self.logger.warning("Response does not contain valid JSON array.")
                return [], 0, 0, time.time() - start_time
        except json.JSONDecodeError as e:
            self.logger.warning(f"Failed to decode JSON response: {e}")
            return [], 0, 0, time.time() - start_time