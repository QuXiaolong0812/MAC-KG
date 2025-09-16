import os
import json
import time
from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from SelfUtils.Logger import Logger
from DataStructure.KGTriple import KGTriple


class TripleAligningAgent:
    """
    TripleAligningAgent aligns entities and relations in knowledge graph triples using semantic similarity.
    It uses a sentence transformer model to generate embeddings and FAISS for efficient similarity search.
    This agent supports incremental alignment and persistent storage using jsonl files.
    """

    TYPE_MAP = {"textual": 0, "visual": 1}  # Type Number Mapping

    def __init__(self, model_name: str, threshold: float, store_dir='triples_store', logger: Logger = None):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.store_dir = store_dir

        self.entity2id = {}
        self.relation2id = {}
        self.id2entity = {}
        self.id2relation = {}
        self.entity_embeddings = []
        self.relation_embeddings = []

        self.entity_index = faiss.IndexFlatIP(384)
        self.relation_index = faiss.IndexFlatIP(384)

        self.logger = logger

        self._load_store()

        self.existing_triples_set = set()
        self._load_existing_triples()

    # ---------------- Persistence Methods ----------------

    def _save_store(self, name: str, idx: int, emb: np.ndarray, is_entity=True):
        """Append a new entity or relation to jsonl files."""
        os.makedirs(self.store_dir, exist_ok=True)

        if is_entity:
            path_map = os.path.join(self.store_dir, 'entity2id.jsonl')
            path_emb = os.path.join(self.store_dir, 'entity_embeddings.jsonl')
            record = {"name": name, "id": idx}
        else:
            path_map = os.path.join(self.store_dir, 'relation2id.jsonl')
            path_emb = os.path.join(self.store_dir, 'relation_embeddings.jsonl')
            record = {"name": name, "id": idx}

        # 逐行追加写
        with open(path_map, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        with open(path_emb, 'a', encoding='utf-8') as f:
            f.write(json.dumps(emb[0].tolist()) + "\n")

    def _load_store(self):
        """Load the entity and relation store from jsonl files."""
        try:
            # Load entities
            ent_map_file = os.path.join(self.store_dir, 'entity2id.jsonl')
            ent_emb_file = os.path.join(self.store_dir, 'entity_embeddings.jsonl')
            if os.path.exists(ent_map_file) and os.path.exists(ent_emb_file):
                with open(ent_map_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        record = json.loads(line)
                        self.entity2id[record["name"]] = record["id"]
                        self.id2entity[record["id"]] = record["name"]
                with open(ent_emb_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        vec = json.loads(line)
                        self.entity_embeddings.append(vec)
                emb = np.array(self.entity_embeddings).astype("float32")
                if emb.shape[0] > 0:
                    faiss.normalize_L2(emb)
                    self.entity_index.add(emb)

            # Load relations
            rel_map_file = os.path.join(self.store_dir, 'relation2id.jsonl')
            rel_emb_file = os.path.join(self.store_dir, 'relation_embeddings.jsonl')
            if os.path.exists(rel_map_file) and os.path.exists(rel_emb_file):
                with open(rel_map_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        record = json.loads(line)
                        self.relation2id[record["name"]] = record["id"]
                        self.id2relation[record["id"]] = record["name"]
                with open(rel_emb_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        vec = json.loads(line)
                        self.relation_embeddings.append(vec)
                emb = np.array(self.relation_embeddings).astype("float32")
                if emb.shape[0] > 0:
                    faiss.normalize_L2(emb)
                    self.relation_index.add(emb)

        except Exception as e:
            self.logger and self.logger.warning(f"Failed to load persistent data, reinitializing... {e}")

    def _load_existing_triples(self):
        path = os.path.join(self.store_dir, 'textual_triples2id.jsonl')
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    obj = json.loads(line)
                    key = (obj['head'], obj['relation'], obj['tail'], obj['type'])
                    self.existing_triples_set.add(key)

    # ---------------- Alignment Methods ----------------

    def _get_aligned_name(self, name: str, is_entity=True) -> Tuple[str, np.ndarray]:
        """Get the aligned name from the store or return None if not found."""
        emb = self.model.encode(name)
        emb = np.array([emb]).astype('float32')
        faiss.normalize_L2(emb)

        index = self.entity_index if is_entity else self.relation_index
        reverse_map = self.id2entity if is_entity else self.id2relation

        if index.ntotal == 0:
            return None, emb

        D, I = index.search(emb, 1)
        if D[0][0] >= self.threshold:
            return reverse_map[I[0][0]], emb
        return None, emb

    def _add_to_store(self, name: str, emb: np.ndarray, is_entity=True):
        """Add a new entity or relation to the store."""
        idx = len(self.entity2id) if is_entity else len(self.relation2id)
        if is_entity:
            self.entity2id[name] = idx
            self.id2entity[idx] = name
            self.entity_embeddings.append(emb[0].tolist())
            self.entity_index.add(emb)
        else:
            self.relation2id[name] = idx
            self.id2relation[idx] = name
            self.relation_embeddings.append(emb[0].tolist())
            self.relation_index.add(emb)

        self._save_store(name, idx, emb, is_entity=is_entity)

    def _deduplicate_triples(self, triples: List[KGTriple]) -> List[KGTriple]:
        """Remove duplicate triples while preserving order."""
        seen = set()
        deduped = []
        for triplet in triples:
            key = (triplet.head, triplet.relation, triplet.tail, triplet.type)
            if key not in seen:
                seen.add(key)
                deduped.append(triplet)
        return deduped

    def align_triples(self, triples: List[KGTriple]) -> Tuple[List[KGTriple], int, float]:
        """Align entities and relations in the given triples, return deduped triples, head entity ID, and elapsed time."""
        aligned_triples = []
        start_time = time.time()

        # Since all triples have the same head, we take the first one here
        head_name = triples[0].head
        aligned_head, head_emb = self._get_aligned_name(head_name, is_entity=True)
        if not aligned_head:
            self._add_to_store(head_name, head_emb, is_entity=True)
            aligned_head = head_name
        head_id = self.entity2id[aligned_head]

        for triple in triples:
            # head is aligned，so using aligned_head
            relation, tail = triple.relation, triple.tail

            aligned_relation, relation_emb = self._get_aligned_name(relation, is_entity=False)
            if not aligned_relation:
                self._add_to_store(relation, relation_emb, is_entity=False)
                aligned_relation = relation

            aligned_tail, tail_emb = self._get_aligned_name(tail, is_entity=True)
            if not aligned_tail:
                self._add_to_store(tail, tail_emb, is_entity=True)
                aligned_tail = tail

            aligned_triples.append(KGTriple(aligned_head, aligned_relation, aligned_tail, triple.type))

        deduped_triples = self._deduplicate_triples(aligned_triples)
        self.logger and self.logger.info(f"Aligned {len(triples)} triples to {len(deduped_triples)} triples.")

        # Save triples2id.jsonl (incremental write)
        self._save_textual_triples2id(deduped_triples)

        return deduped_triples, head_id, time.time() - start_time

    # ---------------- Save triples2id ----------------

    def _save_textual_triples2id(self, triples: List[KGTriple]):
        """Append new triples to triples2id.jsonl, skipping already existing ones."""
        os.makedirs(self.store_dir, exist_ok=True)
        path = os.path.join(self.store_dir, 'textual_triples2id.jsonl')

        with open(path, 'a', encoding='utf-8') as f:
            for t in triples:
                head_id = self.entity2id.get(t.head, -1)
                tail_id = self.entity2id.get(t.tail, -1)
                rel_id = self.relation2id.get(t.relation, -1)
                type_id = self.TYPE_MAP.get(t.type, 0)
                key = (head_id, rel_id, tail_id, type_id)
                if key not in self.existing_triples_set:
                    self.existing_triples_set.add(key)
                    obj = {"head": head_id, "relation": rel_id, "tail": tail_id, "type": type_id}
                    f.write(json.dumps(obj, ensure_ascii=False) + "\n")