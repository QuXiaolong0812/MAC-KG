import json
from openai import OpenAI
from typing import List, Tuple
import time
from SelfUtils.Logger import Logger
from DataStructure.KGTriple import KGTriple, convert_json_list_to_triples

class StructureVerifyingAgent:
    """
    Structure Verifying Agent for MAC-KG.
    This agent verifies the structure of knowledge graph triples using OpenAI's API.
    """

    def __init__(self, client: OpenAI, model_name: str, logger: Logger = None):
        self.client = client
        self.model_name = model_name
        self.logger = logger
        self.system_rule_verify_prompt = """
You are the Structure Verifying Agent (SVA).
Your task is to analyze a list of (head, relation, tail) triples and identify and remove logically incorrect or contradictory triples. 
These errors may include directional contradictions, mutually exclusive attributes, or logical inconsistencies in quantity or state.
All output triples must be selected from the original input. Do not create or infer any new triples.

Input Format:
{
  "triples": [
    {"head": "Lion", "relation": "has color", "tail": "golden"},
    {"head": "Lion", "relation": "has color", "tail": "blue"},
    {"head": "Lion", "relation": "has number of legs", "tail": "4"},
    {"head": "Lion", "relation": "has number of legs", "tail": "2"},
    {"head": "Lion", "relation": "is prey of", "tail": "Zebra"},
    {"head": "Zebra", "relation": "is prey of", "tail": "Lion"}
  ]
}

Output Format:
Return a JSON object with a single key "valid_triples" containing only the logically correct and consistent triples.
{
  "valid_triples": [
    {"head": "Lion", "relation": "has color", "tail": "golden"},
    {"head": "Lion", "relation": "has number of legs", "tail": "4"}
  ]
}

🧩 Error Detection Guidelines
You must identify and filter out any triple that fits the following categories:
1. 🔁 Directional Contradictions
Two triples that contradict each other due to inverse but exclusive relations, e.g.:
- A is parent of B and B is parent of A → contradiction
- X is prey of Y and Y is prey of X → contradiction
2. 🚫 Mutually Exclusive Attributes
Triples that cannot logically coexist, e.g.:
- has gender: male and is pregnant: true
- has habitat: desert and has habitat: deep sea
- has color: golden and has color: blue
3. 🔢 Quantity Conflicts
Numerical claims that contradict each other for the same entity and property:
- has number of legs: 4 vs. has number of legs: 2
- has lifespan: 2 years vs. has lifespan: 20 years (for same entity type)
✅ Keep triples that:
- Are logically consistent with others.
- Are descriptive without contradicting.
- Are duplicated but non-conflicting (e.g., same color listed twice).

❌ Never:
- Guess or infer missing facts.
- Generate new triples.
- Alter the content of the input triples.

🔎 Ambiguity Handling:
- When two or more contradictory triples appear, keep the one that is more common, plausible, or informative, and discard the others.
- If you're unsure whether a triple is invalid, err on the side of caution and keep it.

POSITIVE EXAMPLE
Input:
{
  "triples": [
    {"head": "Whale", "relation": "is a type of", "tail": "mammal"},
    {"head": "Whale", "relation": "lays eggs", "tail": "true"},
    {"head": "Whale", "relation": "has number of fins", "tail": "2"},
    {"head": "Whale", "relation": "has number of fins", "tail": "2"}
  ]
}

Output:
{
  "valid_triples": [
    {"head": "Whale", "relation": "is a type of", "tail": "mammal"},
    {"head": "Whale", "relation": "has number of fins", "tail": "2"}
  ]
}
"""

    def verify_triples(self, triples: List[KGTriple]) -> Tuple[List[KGTriple], int, int, float]:
        """
        Verify the structure of knowledge graph triples.

        Args:
            triples (List[KGTriple]): The list of triples to verify.

        Returns:
            List[KGTriple]: Verified triples.
            int: Total prompt tokens used.
            int: Total completion tokens used.
            float: Processing time in seconds.
        """
        if not triples:
            return [], 0, 0, 0.0

        start_time = time.time()
        # Prepare the input for the language model
        input_triples = {
            "triples": [triple.to_dict() for triple in triples]
        }
        input_string = json.dumps(input_triples, ensure_ascii=False)
        prompt = f"""
You are the Structure Verifying Agent (SVA).
Your task is to analyze a list of (head, relation, tail) triples and identify and remove logically incorrect or contradictory triples.
These errors may include directional contradictions, mutually exclusive attributes, or logical inconsistencies in quantity or state.
All output triples must be selected from the original input. Do not create or infer any new triples.
Input triples:
{input_string}  
Output Format:
Return a JSON object with a single key "valid_triples" containing only the logically correct and consistent triples.
"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_rule_verify_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            content = response.choices[0].message.content.strip()

            # Try to parse JSON entities from the response
            try:
                json_str = content[content.find("{"):content.rfind("}") + 1]
                json_data = json.loads(json_str)
                valid_triples = convert_json_list_to_triples(json_data.get("valid_triples", []))
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                processing_time = time.time() - start_time
                return valid_triples, prompt_tokens, completion_tokens, processing_time
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse JSON from the response.")
                return [], 0, 0, 0.0
            except KeyError:
                self.logger.warning("Response does not contain 'valid_triples'.")
                return [], 0, 0, 0.0
            except Exception as e:
                self.logger.warning(f"An unexpected error occurred: {e}")
                return [], 0, 0, 0.0
        except Exception as e:
            self.logger.warning(f"An error occurred during API call: {e}")
            return [], 0, 0, 0.0