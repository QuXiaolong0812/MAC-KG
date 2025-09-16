from openai import OpenAI
from typing import List, Tuple
import time
from SelfUtils.Logger import Logger
from DataStructure.KGTriple import KGTriple, convert_json_list_to_triples

class TextualEnhancingAgent:
    """
    A class to represent a Textual Enhancing Agent.
    This agent is designed to enhance textual data by applying various transformations
    and improvements to the input text.
    """

    def __init__(self, client: OpenAI, model_name: str, logger: Logger):
        """
        Initializes the TextualEnhancingAgent with an OpenAI client and model name.

        Args:
            client (OpenAI): An instance of the OpenAI client.
            model_name (str): The name of the model to be used for text enhancement.
            logger (Logger): An instance of Logger for logging messages.
        """
        self.client = client
        self.model_name = model_name
        self.logger = logger
        self.system_prompt4triple2desc = """
You are the Appearance Triple Verbalization Expert.
Your task is to convert a list of structured (head, relation, tail) triples into a concise and coherent English description text that summarizes the visual appearance of the given head entity. 
The triples are guaranteed to describe the shape, color, or overall external appearance of the head entity.

Input Format:
{
  "head": "<head entity>",
  "triples": [
    {"head": "<head entity>", "relation": "<relation>", "tail": "<tail entity>"},
    ...
  ]
}

Output Format:
Return a single English sentence or a few tightly connected sentences that describe the entity’s visual traits.
- Avoid redundant phrasing.
- Make the description fluent and human-like.
- Do not mention the relation or triple structure—only the meaning should be conveyed.

Verbalization Guidelines:
✅ Describe the entity's color, shape, texture, markings, or notable features.
✅ Combine related triples naturally into fluent sentences.
✅ Use varied and natural phrasing (e.g., "has a slender body", "its fur is golden with black spots").
❌ Do not output raw triple syntax or mention words like “relation”, “triple”, “tail”.
❌ Do not invent facts outside of the provided triples.
If no valid triples are given, return an empty string.

POSITIVE EXAMPLE:
Input:
{
  "head": "Leopard cat",
  "triples": [
    {"head": "Leopard cat", "relation": "has fur color", "tail": "golden with black spots"},
    {"head": "Leopard cat", "relation": "has body shape", "tail": "slender"},
    {"head": "Leopard cat", "relation": "has eye color", "tail": "pale green"}
  ]
}

Output:
"The leopard cat has a slender body, golden fur with black spots, and pale green eyes."

NEGATIVE EXAMPLE (Don't do this):
"Leopard cat has fur color golden with black spots; has body shape slender; has eye color pale green." ❌
"Relation: has fur color. Tail: golden with black spots." ❌
"""

    def verbalize_triples(self, head: str, triples: List[KGTriple]) -> Tuple[str, int, int, float]:
        """
        Converts a list of triples into a natural language description of the head entity's visual appearance.

        Args:
            head (str): The head entity for which the description is generated.
            triples (List[KGTriple]): The list of triples to be verbalized.

        Returns:
            Tuple[str, int, int, float]: A tuple containing the verbalized description,
                                          number of prompt tokens, number of completion tokens,
                                          and processing time in seconds.
        """
        if not triples:
            return "", 0, 0, 0.0
        start_time = time.time()
        # Prepare the input for the prompt
        triples_str = ",\n".join(
            f'{{"head": "{triple.head}", "relation": "{triple.relation}", "tail": "{triple.tail}"}}' for triple in
            triples)
        prompt = f"""
        The input entities and triples are as follows:
        "head": "{head}",
        "triples": [
        {triples_str}
        ]
                """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt4triple2desc},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            content = response.choices[0].message.content.strip()
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            processing_time = time.time() - start_time
            return content, prompt_tokens, completion_tokens, processing_time
        except Exception as e:
            self.logger.warning(f"Failed to convert triples to description text: {e}")
            return "", 0, 0, time.time() - start_time
