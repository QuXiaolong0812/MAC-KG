from dataclasses import dataclass

@dataclass
class KGTriple:
    """
    Data class representing a single knowledge triple made by MAC-KG.

    Attributes:
        head: The subject entity
        relation: The relationship type
        tail: The object entity
        type: The type of the triple (e.g., "textual", "visual")
    """

    head: str
    relation: str
    tail: str
    type: str

    def __init__(self, head: str, relation: str, tail: str, type_: str = "textual"):
        """Initialize a KGTriple instance."""
        self.head = head
        self.relation = relation
        self.tail = tail
        self.type = type_

    def __repr__(self):
        """Official string representation of the KGTriple."""
        return f"KGTriple({self.head}, {self.relation}, {self.tail}, type={self.type})"

    def __eq__(self, other):
        """Check equality between two KGTriple instances."""
        if not isinstance(other, KGTriple):
            return False
        return (self.head, self.relation, self.tail, self.type) == (other.head, other.relation, other.tail, other.type)

    def __hash__(self):
        """Compute the hash of the KGTriple instance."""
        return hash((self.head, self.relation, self.tail, self.type))

    def __str__(self) -> str:
        """String representation of the knowledge triple."""
        return f"({self.head}) -[{self.relation}]-> ({self.tail}) [type: {self.type}]" if self.type else f"({self.head}) -[{self.relation}]-> ({self.tail})"

    def to_dict(self):
        """Convert the KGTriple instance to a dictionary."""
        return {
            'head': self.head,
            'relation': self.relation,
            'tail': self.tail,
        }

def convert_json_list_to_triples(json_list):
    """
    Convert a list of JSON objects to a list of KGTriple instances.

    Args:
        json_list (list): List of JSON objects containing 'head', 'relation', 'tail', and 'type'.

    Returns:
        List[KGTriple]: List of KGTriple instances.
    """
    triples = []
    for item in json_list:
        head = item.get('head', '')
        relation = item.get('relation', '')
        tail = item.get('tail', '')
        type_ = item.get('type', 'textual')  # Default to 'textual' if type is not specified
        if head and relation and tail:
            triples.append(KGTriple(head=head, relation=relation, tail=tail, type_=type_))
        else:
            pass
    return triples