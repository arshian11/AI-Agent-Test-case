"""
SSMG Extractor - Extracts entities, facts, intents, and relations from user utterances
Uses spaCy for NER, rule-based patterns for relations, and lightweight classifiers for intents.
"""

import spacy
import re
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

from .graph import Node, Edge, NodeType, RelationType


logger = logging.getLogger(__name__)

@dataclass
class ExtractionResult:
    """Result of extracting information from an utterance"""
    nodes: List[Node]
    edges: List[Edge]
    confidence: float
    metadata: Dict[str, Any]

class IntentClassifier:
    """Simple rule-based intent classifier"""

    def __init__(self):
        self.intent_patterns = {
            'order': ['order', 'buy', 'purchase', 'get me', 'i want', 'i need'],
            'modify': ['change', 'modify', 'update', 'edit', 'instead', 'actually'],
            'add': ['add', 'include', 'also', 'plus', 'and', 'with'],
            'remove': ['remove', 'delete', 'without', 'no', 'cancel'],
            'question': ['what', 'how', 'when', 'where', 'why', 'can you', '?'],
            'preference': ['prefer', 'like', 'dislike', 'hate', 'love', 'favorite'],
            'constraint': ['dont', "don't", 'never', 'not', 'avoid', 'allergic']
        }

    def classify(self, text: str) -> Tuple[str, float]:
        """Classify intent with confidence score"""
        text_lower = text.lower()

        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            if score > 0:
                intent_scores[intent] = score / len(patterns)

        if not intent_scores:
            return 'unknown', 0.5

        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        return best_intent[0], min(best_intent[1], 1.0)

# class RelationExtractor:
#     """Extract relations between entities using patterns and rules"""

#     def __init__(self):
#         self.relation_patterns = {
#             RelationType.PREFERS: [
#                 (r'(prefer|like|love|want)\s+(\w+)', 'user', 2),
#                 (r'(\w+)\s+is\s+(good|great|favorite)', 1, 'user'),
#             ],
#             RelationType.OWNS: [
#                 (r'(my|mine)\s+(\w+)', 'user', 2),
#                 (r'i\s+have\s+(\w+)', 'user', 1),
#             ],
#             RelationType.ASKED_ABOUT: [
#                 (r'what\s+about\s+(\w+)', 'user', 1),
#                 (r'tell\s+me\s+about\s+(\w+)', 'user', 1),
#             ]
#         }

#     def extract(self, text: str, entities: List[str]) -> List[Tuple[str, RelationType, str, float]]:
#         """Extract relations from text given entities"""
#         relations = []
#         text_lower = text.lower()

#         for relation_type, patterns in self.relation_patterns.items():
#             for pattern, source_group, target_group in patterns:
#                 matches = re.finditer(pattern, text_lower)
#                 for match in matches:
#                     try:
#                         if isinstance(source_group, str):
#                             source = source_group
#                         else:
#                             source = match.group(source_group)

#                         if isinstance(target_group, str):
#                             target = target_group
#                         else:
#                             target = match.group(target_group)

#                         confidence = 0.8  # Base confidence for pattern matches
#                         relations.append((source, relation_type, target, confidence))
#                     except (IndexError, AttributeError):
#                         continue

#         return relations

class RelationExtractor:
    """Extract relations between entities using patterns and rules"""
    
    def __init__(self):
        self.relation_patterns = {
            RelationType.PREFERS: [
                (r'want|like|prefer|love|enjoy', 'user', 2),
                (r'is good|great|excellent|nice|delicious', 1, 'user'),
                (r'i.*want|i.*like|i.*prefer|i.*love', 'user', 1),
            ],
            RelationType.OWNS: [
                (r'my|mine', 'user', 2),
                (r'i have|i own|i got', 'user', 1),
            ],
            RelationType.MENTIONS: [
                (r'about|regarding|concerning', 1, 2),
                (r'tell me about|info about', 'user', 2),
            ]
        }

    
    def extract(self, text: str, entity_contents: List[str]) -> List[Tuple[str, RelationType, str, float]]:
        """Extract relations from text given entity contents"""
        relations = []
        text_lower = text.lower()
        
        # Only create relations between entities that actually exist
        for relation_type, patterns in self.relation_patterns.items():
            for pattern, source_group, target_group in patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    try:
                        if isinstance(source_group, str):
                            source = source_group
                        else:
                            source = match.group(source_group)
                            
                        if isinstance(target_group, str):
                            target = target_group
                        else:
                            target = match.group(target_group)

                        # MAP 'user' TO FIXED SESSION USER ID
                        if source == 'user':
                            source = 'user_session'
                        if target == 'user':
                            target = 'user_session'
                        
                        # Only add relation if both entities exist in our extracted entities
                        if (source in entity_contents or source == 'user_session') and \
                           (target in entity_contents or target == 'user_session'):
                            confidence = 0.8
                            relations.append((source, relation_type, target, confidence))
                    except (IndexError, AttributeError):
                        continue
        
        return relations


class ConstraintExtractor:
    """Extract constraints and preferences from user utterances"""

    def __init__(self):
        self.constraint_patterns = [
            (r"don't\s+(?:use|add|include)\s+(\w+)", "avoid", 1),
            (r"no\s+(\w+)", "avoid", 1),  
            (r"without\s+(\w+)", "avoid", 1),
            (r"allergic\s+to\s+(\w+)", "allergic", 1),
            (r"never\s+(\w+)", "avoid", 1),
            (r"must\s+(?:have|include)\s+(\w+)", "require", 1),
            (r"need\s+(\w+)", "require", 1),
            (r"only\s+(\w+)", "only", 1)
        ]

    def extract(self, text: str) -> List[Tuple[str, str, float]]:
        """Extract constraints from text"""
        constraints = []
        text_lower = text.lower()

        for pattern, constraint_type, group_idx in self.constraint_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                try:
                    item = match.group(group_idx)
                    confidence = 0.9  # High confidence for constraint patterns
                    constraints.append((constraint_type, item, confidence))
                except IndexError:
                    continue

        return constraints

class SSMGExtractor:
    """Main extractor class for SSMG"""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logger.warning(f"Could not load {spacy_model}, using blank model")
            self.nlp = spacy.blank("en")

        self.intent_classifier = IntentClassifier()
        self.relation_extractor = RelationExtractor()
        self.constraint_extractor = ConstraintExtractor()

        # Entity types to focus on
        self.relevant_ent_types = {
            'PERSON', 'ORG', 'PRODUCT', 'FOOD', 'GPE', 
            'MONEY', 'QUANTITY', 'TIME', 'DATE'
        }

    def extract(self, text: str, turn_id: int = 0, user_id: str = "user") -> ExtractionResult:
        """Extract all information from a user utterance"""
        doc = self.nlp(text)
        
        nodes = []
        edges = []
        metadata = {
            'text': text,
            'turn_id': turn_id,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        }
        
        # Extract entities
        entities = self._extract_entities(doc, turn_id)
        nodes.extend(entities)
        
        # Extract intent
        intent_node = self._extract_intent(text, turn_id)
        if intent_node:
            nodes.append(intent_node)
        
        # Extract constraints
        constraint_nodes = self._extract_constraints(text, turn_id)
        nodes.extend(constraint_nodes)
        
        # Extract facts
        fact_nodes = self._extract_facts(doc, turn_id)
        nodes.extend(fact_nodes)
        
        # Create a mapping of node content to node IDs for edge creation
        node_id_map = {node.content.lower(): node.id for node in nodes}
        
        # Extract relations - FIX: Use actual node IDs
        relations = self.relation_extractor.extract(text, list(node_id_map.keys()))
        for source_content, relation_type, target_content, confidence in relations:
            # Map content back to actual node IDs
            source_id = node_id_map.get(source_content.lower())
            target_id = node_id_map.get(target_content.lower())
            
            # Only create edge if both nodes exist
            if source_id and target_id:
                edge = Edge(
                    source_id=source_id,
                    target_id=target_id,
                    relation=relation_type,
                    confidence=confidence,
                    turn_id=turn_id
                )
                edges.append(edge)
        
        # Calculate overall confidence
        if nodes:
            avg_confidence = sum(node.confidence for node in nodes) / len(nodes)
        else:
            avg_confidence = 0.5
            
        return ExtractionResult(
            nodes=nodes,
            edges=edges,
            confidence=avg_confidence,
            metadata=metadata
        )

    def extract_and_infer(self, text: str, turn_id: int = 0, user_id: str = "user") -> ExtractionResult:
        """Extract nodes and edges from text, then infer missing relations with logging"""
        doc = self.nlp(text)

        nodes = []
        edges = []
        metadata = {
            'text': text,
            'turn_id': turn_id,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        }

        # --- Step 1: Extract nodes ---
        entities = self._extract_entities(doc, turn_id)
        nodes.extend(entities)

        intent_node = self._extract_intent(text, turn_id)
        if intent_node:
            nodes.append(intent_node)

        constraint_nodes = self._extract_constraints(text, turn_id)
        nodes.extend(constraint_nodes)

        fact_nodes = self._extract_facts(doc, turn_id)
        nodes.extend(fact_nodes)
        # user_node = Node(
        #     id=f"user_{turn_id}",
        #     type=NodeType.ENTITY,
        #     content="user",
        #     confidence=1.0,
        #     turn_id=turn_id,
        #     metadata={"is_user": True}
        # )
        # nodes.append(user_node)
        # Map content -> node ID for edges
        node_id_map = {node.content.lower(): node.id for node in nodes}
        # node_id_map['user'] = user_node.id  # Explicit user mapping
        node_id_map['user'] = 'user_session'
        node_id_map['user_session'] = 'user_session'  # Also map the actual ID
        # --- Step 2: Extract edges from relation extractor ---
        relations = self.relation_extractor.extract(text, list(node_id_map.keys()))
        for source_content, relation_type, target_content, confidence in relations:
            source_id = node_id_map.get(source_content.lower())
            target_id = node_id_map.get(target_content.lower())
            if source_id and target_id:
                edge = Edge(
                    source_id=source_id,
                    target_id=target_id,
                    relation=relation_type,
                    confidence=confidence,
                    turn_id=turn_id
                )
                edges.append(edge)
                logger.info(f"[Turn {turn_id}] Extracted edge: {source_content} -> {target_content} ({relation_type.value})")

        # --- Step 3: Rule-based inference for missing edges ---
        # (a) Intent -> other nodes
        intent_nodes = [n for n in nodes if n.type == NodeType.INTENT]
        for intent in intent_nodes:
            for target in nodes:
                if target.id == intent.id:
                    continue

                # Infer appropriate relation type
                if target.type == NodeType.ENTITY:
                    relation = RelationType.MENTIONS
                elif target.type == NodeType.FACT:
                    relation = RelationType.ASKED_ABOUT
                elif target.type == NodeType.CONSTRAINT:
                    relation = RelationType.PREFERS
                else:
                    continue

                inferred_edge = Edge(
                    source_id=intent.id,
                    target_id=target.id,
                    relation=relation,
                    confidence=0.6,
                    turn_id=turn_id
                )
                edges.append(inferred_edge)
                logger.info(f"[Turn {turn_id}] Inferred edge (intent->{target.type.value}): "
                            f"{intent.content} -> {target.content} ({relation.value})")

        # (b) Constraint -> Entity or Fact
        constraint_nodes = [n for n in nodes if n.type == NodeType.CONSTRAINT]
        for constraint in constraint_nodes:
            for target in nodes:
                if target.id == constraint.id:
                    continue

                if target.type == NodeType.ENTITY:
                    relation = RelationType.PREFERS
                elif target.type == NodeType.FACT:
                    relation = RelationType.CONFLICTS_WITH
                else:
                    continue

                inferred_edge = Edge(
                    source_id=constraint.id,
                    target_id=target.id,
                    relation=relation,
                    confidence=0.7,
                    turn_id=turn_id
                )
                edges.append(inferred_edge)
                logger.info(f"[Turn {turn_id}] Inferred edge (constraint->{target.type.value}): "
                            f"{constraint.content} -> {target.content} ({relation.value})")

        # --- Step 4: Calculate average confidence ---
        avg_confidence = (
            sum(node.confidence for node in nodes) / len(nodes)
            if nodes else 0.5
        )

        return ExtractionResult(
            nodes=nodes,
            edges=edges,
            confidence=avg_confidence,
            metadata=metadata
        )


    def _extract_entities(self, doc, turn_id: int) -> List[Node]:
        """Extract named entities from spaCy doc"""
        entities = []

        for ent in doc.ents:
            if ent.label_ in self.relevant_ent_types:
                node = Node(
                    id=f"entity_{ent.text.lower()}_{turn_id}",
                    type=NodeType.ENTITY,
                    content=ent.text,
                    confidence=0.8,  # Base confidence for NER
                    turn_id=turn_id,
                    metadata={
                        'entity_type': ent.label_,
                        'start': ent.start_char,
                        'end': ent.end_char
                    }
                )
                entities.append(node)

        # Also extract important nouns not caught by NER
        for token in doc:
            if (token.pos_ == 'NOUN' and 
                token.is_alpha and 
                len(token.text) > 2 and
                not token.is_stop):

                # Check if already extracted as entity
                if not any(token.text.lower() in ent.content.lower() for ent in entities):
                    node = Node(
                        id=f"entity_{token.text.lower()}_{turn_id}",
                        type=NodeType.ENTITY,
                        content=token.text,
                        confidence=0.6,  # Lower confidence for simple nouns
                        turn_id=turn_id,
                        metadata={'pos': token.pos_, 'lemma': token.lemma_}
                    )
                    entities.append(node)

        return entities

    def _extract_intent(self, text: str, turn_id: int) -> Optional[Node]:
        """Extract user intent from text"""
        intent, confidence = self.intent_classifier.classify(text)

        if intent and intent != 'unknown':
            return Node(
                id=f"intent_{intent}_{turn_id}",
                type=NodeType.INTENT,
                content=intent,
                confidence=confidence,
                turn_id=turn_id,
                metadata={'raw_text': text}
            )
        return None

    def _extract_constraints(self, text: str, turn_id: int) -> List[Node]:
        """Extract constraints from text"""
        constraints = []
        extracted = self.constraint_extractor.extract(text)

        for constraint_type, item, confidence in extracted:
            node = Node(
                id=f"constraint_{constraint_type}_{item}_{turn_id}",
                type=NodeType.CONSTRAINT,
                content=f"{constraint_type}: {item}",
                confidence=confidence,
                turn_id=turn_id,
                metadata={
                    'constraint_type': constraint_type,
                    'item': item,
                    'raw_text': text
                }
            )
            constraints.append(node)

        return constraints

    def _extract_facts(self, doc, turn_id: int) -> List[Node]:
        """Extract factual statements from text"""
        facts = []

        # Simple heuristic: extract noun phrases as potential facts
        for chunk in doc.noun_chunks:
            if len(chunk.text) > 3 and not chunk.root.is_stop:
                node = Node(
                    id=f"fact_{chunk.text.lower().replace(' ', '_')}_{turn_id}",
                    type=NodeType.FACT,
                    content=chunk.text,
                    confidence=0.5,  # Low confidence for simple facts
                    turn_id=turn_id,
                    metadata={
                        'chunk_root': chunk.root.text,
                        'start': chunk.start_char,
                        'end': chunk.end_char
                    }
                )
                facts.append(node)

        return facts

    def extract_updates(self, text: str, existing_nodes: List[str], turn_id: int = 0) -> ExtractionResult:
        """Extract incremental updates given existing context"""
        # For now, same as regular extract but could be enhanced
        # to better handle references and updates
        result = self.extract(text, turn_id)

        # Add metadata about potential updates/references
        result.metadata['existing_context'] = existing_nodes
        result.metadata['is_update'] = any(
            word in text.lower() 
            for word in ['change', 'instead', 'actually', 'modify', 'update']
        )

        return result

