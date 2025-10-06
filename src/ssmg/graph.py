"""
Session-scoped Short-term Memory Graph (SSMG) - Core Graph Implementation
Handles nodes, edges, metadata, and eviction policies for ephemeral dialogue memory.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Tuple
from datetime import datetime
import json
import logging
from collections import defaultdict, deque
from enum import Enum

logger = logging.getLogger(__name__)

class NodeType(Enum):
    ENTITY = "entity"
    FACT = "fact" 
    INTENT = "intent"
    CONSTRAINT = "constraint"

class RelationType(Enum):
    OWNS = "owns"
    PREFERS = "prefers"
    ASKED_ABOUT = "asked_about"
    BEFORE = "before"
    AFTER = "after"
    MENTIONS = "mentions"
    CONFLICTS_WITH = "conflicts_with"


relation_map = {
    NodeType.INTENT: {
        NodeType.ENTITY: RelationType.MENTIONS,
        NodeType.FACT: RelationType.ASKED_ABOUT,
        NodeType.CONSTRAINT: RelationType.PREFERS
    },
    NodeType.ENTITY: {
        NodeType.FACT: RelationType.OWNS,
        NodeType.CONSTRAINT: RelationType.PREFERS,
        NodeType.ENTITY: RelationType.MENTIONS
    },
    NodeType.FACT: {
        NodeType.ENTITY: RelationType.ASKED_ABOUT,
        NodeType.FACT: RelationType.BEFORE,  # or AFTER if known
        NodeType.CONSTRAINT: RelationType.CONFLICTS_WITH
    },
    NodeType.CONSTRAINT: {
        NodeType.ENTITY: RelationType.PREFERS,
        NodeType.FACT: RelationType.CONFLICTS_WITH,
        NodeType.INTENT: RelationType.PREFERS,
        NodeType.CONSTRAINT: RelationType.CONFLICTS_WITH
    }
}


@dataclass
class Node:
    """Represents a node in the SSMG graph"""
    id: str
    type: NodeType
    content: str
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    turn_id: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def age_in_turns(self, current_turn: int) -> int:
        """Calculate age in turns"""
        return max(0, current_turn - self.turn_id)

    def decay_confidence(self, decay_rate: float = 0.05) -> None:
        """Apply confidence decay over time"""
        self.confidence = max(0.1, self.confidence - decay_rate)

@dataclass 
class Edge:
    """Represents an edge in the SSMG graph"""
    source_id: str
    target_id: str
    relation: RelationType
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)
    turn_id: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

from nltk.stem import WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()


class SSMGGraph:
    """Session-scoped Short-term Memory Graph implementation"""

    def __init__(self, 
                 max_nodes: int = 50,
                 max_ttl_turns: int = 8,
                 decay_rate: float = 0.05):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}
        self.adjacency: Dict[str, Set[str]] = defaultdict(set)
        self.node_index = defaultdict(dict)  # {NodeType: {content_lower: node_id}}

        self.lemmatizer = WordNetLemmatizer()

        # Configuration
        self.max_nodes = max_nodes
        self.max_ttl_turns = max_ttl_turns
        self.decay_rate = decay_rate
        self.current_turn = 0

        # Priority weights for eviction
        self.type_priorities = {
            NodeType.CONSTRAINT: 0.9,  # Highest priority
            NodeType.INTENT: 0.7,
            NodeType.ENTITY: 0.5,
            NodeType.FACT: 0.3
        }

        logger.info(f"Initialized SSMG with max_nodes={max_nodes}, ttl={max_ttl_turns}")

    def _normalize_content(self, text: str) -> str:
        """Normalize text for deduplication consistency."""
        text = text.strip().lower()
        # Remove common determiners and punctuation noise
        for token in ["the ", "a ", "an "]:
            text = text.replace(token, "")
        text = text.replace(":", "").replace(",", "").strip()
        # Lemmatize word by word
        # text = " ".join(self.lemmatizer.lemmatize(word) for word in text.split())
        return text


    def add_node(self, node: Node) -> bool:
        # """Add a node to the graph with eviction if necessary"""
        # node.turn_id = self.current_turn

        # # Check if node already exists - update if so
        # if node.id in self.nodes:
        #     self._update_existing_node(node)
        #     return True

        # # Evict if at capacity
        # if len(self.nodes) >= self.max_nodes:
        #     self._evict_nodes(1)

        # self.nodes[node.id] = node
        # self.adjacency[node.id] = set()

        # logger.debug(f"Added node: {node.id} ({node.type.value})")
        # return True
        """Add a node to the graph, merging duplicates of same type and content."""
        node.turn_id = self.current_turn
        # content_key = node.content.strip().lower()
        content_key = self._normalize_content(node.content)


        # --- Deduplication using node_index (O(1) lookup) ---
        if content_key in self.node_index[node.type]:
            existing_id = self.node_index[node.type][content_key]
            existing_node = self.nodes[existing_id]

            existing_node.confidence = min(1.0, existing_node.confidence + 0.1)

            # Update existing node’s confidence, timestamp, and metadata
            existing_node.confidence = max(existing_node.confidence, node.confidence)
            existing_node.timestamp = node.timestamp
            existing_node.turn_id = node.turn_id
            existing_node.metadata.update(node.metadata)

            logger.debug(f"Updated existing node (dedup): {existing_node.id} ({existing_node.content})")
            return True
        # -----------------------------------------------------

        # Evict if needed
        if len(self.nodes) >= self.max_nodes:
            self._evict_nodes(1)

        # Add new node
        self.nodes[node.id] = node
        self.adjacency[node.id] = set()

        # Register in index
        self.node_index[node.type][content_key] = node.id

        logger.debug(f"Added node: {node.id} ({node.type.value})")
        return True

    def add_edge(self, edge: Edge) -> bool:
        """Add an edge to the graph"""
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            logger.warning(f"Cannot add edge - missing nodes: {edge.source_id} -> {edge.target_id}")
            return False

        edge.turn_id = self.current_turn
        edge_id = f"{edge.source_id}:{edge.relation.value}:{edge.target_id}"

        self.edges[edge_id] = edge
        self.adjacency[edge.source_id].add(edge.target_id)

        logger.debug(f"Added edge: {edge_id}")
        return True

    def _update_existing_node(self, new_node: Node) -> None:
        """Update existing node with new information"""
        existing = self.nodes[new_node.id]

        # Update content if more recent or higher confidence
        if (new_node.confidence > existing.confidence or 
            new_node.turn_id > existing.turn_id):
            existing.content = new_node.content
            existing.confidence = max(existing.confidence, new_node.confidence)
            existing.timestamp = new_node.timestamp
            existing.turn_id = new_node.turn_id
            existing.metadata.update(new_node.metadata)

    def _evict_nodes(self, count: int) -> None:
        """Evict nodes based on age, confidence, and type priority"""
        if len(self.nodes) <= count:
            return

        # Calculate eviction scores (lower = more likely to evict)
        node_scores = []
        for node_id, node in self.nodes.items():
            age = node.age_in_turns(self.current_turn)
            type_priority = self.type_priorities.get(node.type, 0.5)

            # Score = type_priority * confidence / (1 + age)
            score = type_priority * node.confidence / (1 + age * 0.1)
            node_scores.append((score, node_id))

        # Sort by score and evict lowest
        node_scores.sort()
        to_evict = [node_id for _, node_id in node_scores[:count]]

        for node_id in to_evict:
            self._remove_node(node_id)
            logger.debug(f"Evicted node: {node_id}")

    def _remove_node(self, node_id: str) -> None:
        """Remove a node and its associated edges"""
        # if node_id not in self.nodes:
        #     return

        # # Remove edges
        # edges_to_remove = []
        # for edge_id, edge in self.edges.items():
        #     if edge.source_id == node_id or edge.target_id == node_id:
        #         edges_to_remove.append(edge_id)

        # for edge_id in edges_to_remove:
        #     del self.edges[edge_id]

        # # Remove from adjacency
        # for neighbor_id in self.adjacency[node_id]:
        #     if node_id in self.adjacency.get(neighbor_id, set()):
        #         self.adjacency[neighbor_id].discard(node_id)

        # del self.adjacency[node_id]
        # del self.nodes[node_id]
        if node_id not in self.nodes:
            return

        node = self.nodes[node_id]
        content_key = node.content.strip().lower()

        # Remove from index
        if node.type in self.node_index and content_key in self.node_index[node.type]:
            del self.node_index[node.type][content_key]

        # Remove from graph
        del self.nodes[node_id]
        del self.adjacency[node_id]

        # Remove from other adjacency lists
        for neighbors in self.adjacency.values():
            neighbors.discard(node_id)

        logger.debug(f"Removed node: {node_id}")

    def infer_relations_for_turn(self) -> None:
        """Infer simple semantic edges between recently added nodes."""
        recent_nodes = self.get_recent_nodes(max_turns_back=0)
        if not recent_nodes:
            return

        for src in recent_nodes:
            for tgt in self.nodes.values():
                if src.id == tgt.id:
                    continue

                relation = relation_map.get(src.type, {}).get(tgt.type)
                if relation is None:
                    continue

                edge = Edge(
                    source_id=src.id,
                    target_id=tgt.id,
                    relation=relation,
                    confidence=min(src.confidence, tgt.confidence)
                )
                self.add_edge(edge)


    def advance_turn(self) -> None:
        """Advance to next turn and apply TTL eviction"""
        self.current_turn += 1

        # Apply confidence decay
        for node in self.nodes.values():
            node.decay_confidence(self.decay_rate)

        # TTL-based eviction
        expired_nodes = []
        for node_id, node in self.nodes.items():
            if node.age_in_turns(self.current_turn) > self.max_ttl_turns:
                # Skip high-priority nodes for TTL
                if node.type in [NodeType.CONSTRAINT, NodeType.INTENT]:
                    continue
                expired_nodes.append(node_id)

        for node_id in expired_nodes:
            self._remove_node(node_id)
            logger.debug(f"TTL expired node: {node_id}")

    def get_neighbors(self, node_id: str, max_depth: int = 2) -> Set[str]:
        """Get neighbors within max_depth hops"""
        if node_id not in self.nodes:
            return set()

        visited = set()
        queue = deque([(node_id, 0)])

        while queue:
            current_id, depth = queue.popleft()
            if current_id in visited or depth > max_depth:
                continue

            visited.add(current_id)

            # Add direct neighbors
            for neighbor_id in self.adjacency[current_id]:
                if neighbor_id not in visited:
                    queue.append((neighbor_id, depth + 1))

        visited.discard(node_id)  # Remove self
        return visited

    def get_nodes_by_type(self, node_type: NodeType) -> List[Node]:
        """Get all nodes of a specific type"""
        return [node for node in self.nodes.values() if node.type == node_type]

    def get_recent_nodes(self, max_turns_back: int = 3) -> List[Node]:
        """Get nodes from recent turns"""
        cutoff_turn = max(0, self.current_turn - max_turns_back)
        return [node for node in self.nodes.values() 
                if node.turn_id >= cutoff_turn]

    def find_conflicts(self) -> List[Tuple[str, str]]:
        """Find conflicting nodes based on CONFLICTS_WITH edges"""
        conflicts = []
        for edge in self.edges.values():
            if edge.relation == RelationType.CONFLICTS_WITH:
                conflicts.append((edge.source_id, edge.target_id))
        return conflicts

    def get_subgraph(self, node_ids: Set[str]) -> 'SSMGGraph':
        """Extract a subgraph containing only specified nodes"""
        subgraph = SSMGGraph(
            max_nodes=len(node_ids),
            max_ttl_turns=self.max_ttl_turns,
            decay_rate=self.decay_rate
        )
        subgraph.current_turn = self.current_turn

        # Add nodes
        for node_id in node_ids:
            if node_id in self.nodes:
                subgraph.nodes[node_id] = self.nodes[node_id]
                subgraph.adjacency[node_id] = set()

        # Add edges between included nodes
        for edge in self.edges.values():
            if edge.source_id in node_ids and edge.target_id in node_ids:
                edge_id = f"{edge.source_id}:{edge.relation.value}:{edge.target_id}"
                subgraph.edges[edge_id] = edge
                subgraph.adjacency[edge.source_id].add(edge.target_id)

        return subgraph

    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary"""
        return {
            'nodes': {
                node_id: {
                    'id': node.id,
                    'type': node.type.value,
                    'content': node.content,
                    'confidence': node.confidence,
                    'timestamp': node.timestamp.isoformat(),
                    'turn_id': node.turn_id,
                    'metadata': node.metadata
                } for node_id, node in self.nodes.items()
            },
            'edges': {
                edge_id: {
                    'source_id': edge.source_id,
                    'target_id': edge.target_id,
                    'relation': edge.relation.value,
                    'confidence': edge.confidence,
                    'timestamp': edge.timestamp.isoformat(),
                    'turn_id': edge.turn_id,
                    'metadata': edge.metadata
                } for edge_id, edge in self.edges.items()
            },
            'current_turn': self.current_turn,
            'max_nodes': self.max_nodes,
            'max_ttl_turns': self.max_ttl_turns
        }

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        return f"SSMGGraph(nodes={len(self.nodes)}, edges={len(self.edges)}, turn={self.current_turn})"

