
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
from difflib import SequenceMatcher
# import fasttext
import nltk
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.chunk import RegexpParser
from .graph import Node, Edge, NodeType, RelationType
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

logger = logging.getLogger(__name__)

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

@dataclass
class ExtractionResult:
    """Result of extracting information from an utterance"""
    nodes: List[Node]
    edges: List[Edge]
    confidence: float
    metadata: Dict[str, Any]


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

# --- Intent Classifiers ---
class IntentClassifier:
    """Simple rule-based intent classifier (default fallback)"""
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
        text_lower = text.lower()
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in text_lower)
            if score > 0:
                intent_scores[intent] = score / len(patterns)
        if not intent_scores:
            return "unknown", 0.5
        best_intent = max(intent_scores.items(), key=lambda x: x[1])
        return best_intent[0], min(best_intent[1], 1.0)

# class FastTextIntentClassifier:
#     """Intent classifier using fastText supervised model"""
#     def __init__(self, model_path: str):
#         try:
#             self.fasttext = fasttext
#             self.model = fasttext.load_model(model_path)
#         except Exception as e:
#             logger.warning(f"Could not load fastText model: {e}")
#             self.model = None

#     def classify(self, text: str) -> Tuple[str, float]:
#         if self.model is None:
#             return 'unknown', 0.0
#         labels, probs = self.model.predict(text, k=1)
#         if labels:
#             label = labels[0].replace('__label__', '')
#             prob = float(probs[0])
#             return label, prob
#         return 'unknown', 0.0

# class TransformerIntentClassifier:
#     """Intent classifier using a HuggingFace transformer model"""
#     def __init__(self, model_name_or_path: str, label_map: dict):
#         try:
#             from transformers import AutoTokenizer, AutoModelForSequenceClassification
#             import torch
#             self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
#             self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
#             self.label_map = label_map
#             self.torch = torch
#         except Exception as e:
#             logger.warning(f"Could not load transformer model: {e}")
#             self.model = None

#     def classify(self, text: str) -> Tuple[str, float]:
#         if self.model is None:
#             return 'unknown', 0.0
#         inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
#         with self.torch.no_grad():
#             outputs = self.model(**inputs)
#             logits = outputs.logits
#             probs = self.torch.softmax(logits, dim=1).cpu().numpy()[0]
#             pred_idx = probs.argmax()
#             intent = self.label_map.get(pred_idx, 'unknown')
#             confidence = float(probs[pred_idx])
#             return intent, confidence
#         return 'unknown', 0.0


class DistilBERTIntentClassifier:
    """Intent classifier using a fine-tuned DistilBERT model and CSV label map"""
    
    def __init__(self, model_dir: str = "src/ssmg/saved_distilbert_clinc_oos", label_map_csv: str = "src/ssmg/saved_distilbert_clinc_oos/label_map.csv"):
        self.torch = torch
        try:
            # Load label map from CSV
            label_df = pd.read_csv(label_map_csv)
            self.label_map = dict(zip(label_df["index"], label_df["label"]))

            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.model.eval()  # evaluation mode

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            self.device = device

        except Exception as e:
            logger.warning(f"Could not load model or label map: {e}")
            self.model = None

    def classify(self, text: str) -> Tuple[str, float]:
        if self.model is None:
            return "unknown", 0.0

        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with self.torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = self.torch.softmax(logits, dim=1).cpu().numpy()[0]
            pred_idx = probs.argmax()
            intent = self.label_map.get(pred_idx, 'unknown')
            if intent.lower() == "oos":
                intent = "unknown"
            confidence = float(probs[pred_idx])
            return intent, confidence


# --- NLTK-based Constraint Extractor ---
class NLTKConstraintExtractor:
    """Extract constraints using NLTK chunking and POS patterns"""
    def __init__(self):
        # Define grammar for constraint phrases (e.g., 'without cheese', 'no onions', 'allergic to nuts')
        self.grammar = r'''
            CONSTRAINT: {<IN|RB|DT|JJ|VB.*>*<NN.*>+}
        '''
        self.chunker = RegexpParser(self.grammar)

    def extract(self, text: str, turn_id: int) -> List[Tuple[str, str, float]]:
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        tree = self.chunker.parse(tagged)
        constraints = []
        for subtree in tree.subtrees(): #type: ignore
            if subtree.label() == 'CONSTRAINT':
                phrase = ' '.join(word for word, tag in subtree.leaves())
                # Heuristic: classify type by keywords
                phrase_lower = phrase.lower()
                if 'without' in phrase_lower or 'no' in phrase_lower or 'exclude' in phrase_lower:
                    ctype = 'avoid'
                elif 'allergic' in phrase_lower:
                    ctype = 'allergic'
                elif 'must' in phrase_lower or 'need' in phrase_lower or 'require' in phrase_lower:
                    ctype = 'require'
                elif 'only' in phrase_lower:
                    ctype = 'only'
                else:
                    continue
                constraints.append((ctype, phrase, 0.7))
        if not constraints:
            logger.debug(f"[Turn {turn_id}] No constraints found in text: {text}")
        return constraints

# --- NLTK-based Relation Extractor ---
class NLTKRelationExtractor:
    """Extract relations using NLTK chunking and POS patterns"""
    def __init__(self):
        # Define grammar for relation patterns (e.g., 'prefer X', 'add Y', 'remove Z')
        self.grammar = r'''
            RELATION: {<PRP|NN.*|NNS|NNP|NNPS|DT|JJ|RB|VB.*>+}
        '''
        self.chunker = RegexpParser(self.grammar)

    def extract(self, text: str, entity_contents: List[str], turn_id: int) -> List[Tuple[str, RelationType, str, float]]:
        tokens = word_tokenize(text)
        tagged = pos_tag(tokens)
        tree = self.chunker.parse(tagged)
        relations = []
        for subtree in tree.subtrees(): #type: ignore
            if subtree.label() == 'RELATION':
                phrase = ' '.join(word for word, tag in subtree.leaves())
                phrase_lower = phrase.lower()
                # Heuristic: classify relation type by keywords
                if 'prefer' in phrase_lower or 'like' in phrase_lower or 'love' in phrase_lower:
                    rtype = RelationType.PREFERS
                elif 'own' in phrase_lower or 'have' in phrase_lower or 'my' in phrase_lower:
                    rtype = RelationType.OWNS
                elif 'add' in phrase_lower or 'include' in phrase_lower:
                    rtype = RelationType.MENTIONS
                elif 'remove' in phrase_lower or 'exclude' in phrase_lower or 'without' in phrase_lower:
                    rtype = RelationType.MENTIONS
                else:
                    continue
                # Try to match to entities
                for ent in entity_contents:
                    if ent in phrase_lower:
                        relations.append(('user', rtype, ent, 0.7))
        return relations

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
        # self.relation_patterns = {
        #     RelationType.PREFERS: [
        #         (r'want|like|prefer|love|enjoy', 'user', 2),
        #         (r'is good|great|excellent|nice|delicious', 1, 'user'),
        #         (r'i.*want|i.*like|i.*prefer|i.*love', 'user', 1),
        #     ],
        #     RelationType.OWNS: [
        #         (r'my|mine', 'user', 2),
        #         (r'i have|i own|i got', 'user', 1),
        #     ],
        #     RelationType.MENTIONS: [
        #         (r'about|regarding|concerning', 1, 2),
        #         (r'tell me about|info about', 'user', 2),
        #     ]
        # }
        # self.relation_patterns = {
        #     RelationType.PREFERS: [
        #         (r'i\s+(?:want|like|prefer|love|enjoy)\s+(?P<target>\w+)', 'user', 'target'),
        #         (r'(?P<source>\w+)\s+is\s+(?:good|great|excellent|nice|delicious)', 'source', 'user'),
        #     ],
        #     RelationType.OWNS: [
        #         (r'i\s+(?:have|own|got)\s+(?P<target>\w+)', 'user', 'target'),
        #         (r'my\s+(?P<target>\w+)', 'user', 'target'),
        #     ],
        #     RelationType.MENTIONS: [
        #         (r'(?:tell me|info)\s+about\s+(?P<target>\w+)', 'user', 'target'),
        #         (r'about\s+(?P<target>\w+)', 'user', 'target'),
        #     ]
        # }
        # self.relation_patterns = {
        #     RelationType.PREFERS: [
        #         (r"i\s+(?:want|like|prefer|love|enjoy)\s+(?P<target>[\w\s]+?)(?:[.,]|$)", "user", "target"),
        #         (r"(?P<source>[\w\s]+)\s+is\s+(?:good|great|excellent|nice|delicious)", "source", "user"),
        #     ],
        #     RelationType.OWNS: [
        #         (r"i\s+(?:have|own|got)\s+(?P<target>[\w\s]+?)(?:[.,]|$)", "user", "target"),
        #         (r"my\s+(?P<target>[\w\s]+?)(?:[.,]|$)", "user", "target"),
        #     ],
        #     RelationType.MENTIONS: [
        #         (r"(?:tell me|info)\s+about\s+(?P<target>[\w\s]+?)(?:[.,]|$)", "user", "target"),
        #         (r"about\s+(?P<target>[\w\s]+?)(?:[.,]|$)", "user", "target"),
        #     ]
        # }
        self.relation_patterns = {
            RelationType.PREFERS: [
                # Explicit expressions of liking or wanting
                (r"\bi\s+(?:want|like|prefer|love|enjoy)\s+(?P<target>[\w\s]+?)(?:[.,;!?]|$)", "user", "target"),
                (r"\bi\s*(?:would\s+)?(?:rather|choose)\s+(?P<target>[\w\s]+?)(?:[.,;!?]|$)", "user", "target"),

                # Comparative preferences ("prefer X over Y")
                (r"\bprefer\s+(?P<target>[\w\s]+?)\s+over\s+(?P<alt>[\w\s]+?)(?:[.,;!?]|$)", "user", "target"),
                (r"\b(?:instead\s+of|not)\s+(?P<alt>[\w\s]+?)\s+(?:i\s+want|i\s+like|choose|prefer)\s+(?P<target>[\w\s]+?)(?:[.,;!?]|$)", "user", "target"),

                # Implicit preference change (“change it to ...”, “make it ...”)
                (r"\bchange\s+(?:it|that|this|order)\s+(?:to|into)\s+(?P<target>[\w\s]+?)(?:[.,;!?]|$)", "user", "target"),
                (r"\bmake\s+(?:it|this|that)\s+(?P<target>[\w\s]+?)(?:[.,;!?]|$)", "user", "target"),

                # Positive sentiment implying preference
                (r"(?P<target>[\w\s]+?)\s+(?:is|looks|tastes)\s+(?:good|great|excellent|nice|amazing|delicious|perfect)", "user", "target"),

                # Negative sentiment implying *dislike* (inverse of prefers)
                (r"\b(?:no|without|don’t|do not|never)\s+(?:want|like|add|include)\s+(?P<target>[\w\s]+?)(?:[.,;!?]|$)", "user", "target"),
                (r"\bi\s+(?:hate|dislike|avoid)\s+(?P<target>[\w\s]+?)(?:[.,;!?]|$)", "user", "target"),
            ],

            RelationType.OWNS: [
                # Direct ownership or possession
                (r"\bi\s+(?:have|own|got)\s+(?P<target>[\w\s]+?)(?:[.,;!?]|$)", "user", "target"),
                (r"\bmy\s+(?P<target>[\w\s]+?)(?:[.,;!?]|$)", "user", "target"),

                # Indirect ownership ("belongs to me", "is mine")
                (r"(?P<target>[\w\s]+?)\s+(?:belongs\s+to|is\s+mine)(?:[.,;!?]|$)", "user", "target"),

                # Situational possession ("I’m using / carrying / wearing ...")
                (r"\bi'?m\s+(?:using|carrying|wearing|holding)\s+(?P<target>[\w\s]+?)(?:[.,;!?]|$)", "user", "target"),
            ],

            RelationType.MENTIONS: [
                # Explicit info requests
                (r"(?:tell|show|give|provide)\s+(?:me\s+)?(?:info|information|details)?\s*(?:about|on|regarding)\s+(?P<target>[\w\s]+?)(?:[.,;!?]|$)", "user", "target"),
                (r"\babout\s+(?P<target>[\w\s]+?)(?:[.,;!?]|$)", "user", "target"),

                # Indirect mentions ("add X", "include X", "remove X", "skip X")
                (r"\b(?:add|include|put)\s+(?P<target>[\w\s]+?)(?:[.,;!?]|$)", "user", "target"),
                (r"\b(?:remove|delete|skip|exclude|avoid)\s+(?P<target>[\w\s]+?)(?:[.,;!?]|$)", "user", "target"),

                # Mentions of questions about something
                (r"what('?s| is)\s+(?:the\s+)?(?P<target>[\w\s]+?)(?:[.,;!?]|$)", "user", "target"),
                (r"how\s+(?:much|many|long)\s+(?:is|are|does)\s+(?P<target>[\w\s]+?)(?:[.,;!?]|$)", "user", "target"),
            ],
        }


    
    def extract(self, text: str, entity_contents: List[str],turn_id: int) -> List[Tuple[str, RelationType, str, float]]:
        """Extract relations from text given entity contents"""
        # relations = []
        # text_lower = text.lower()
        # entity_contents = [e.lower() for e in entity_contents]
        # # Only create relations between entities that actually exist
        # for relation_type, patterns in self.relation_patterns.items():
        #     for pattern, source_group, target_group in patterns:
        #         matches = re.finditer(pattern, text_lower)
        #         for match in matches:
        #             try:
        #                 if isinstance(source_group, str):
        #                     source = source_group
        #                 else:
        #                     source = match.group(source_group)
                            
        #                 if isinstance(target_group, str):
        #                     target = target_group
        #                 else:
        #                     target = match.group(target_group)

        #                 # MAP 'user' TO FIXED SESSION USER ID
        #                 if source == 'user':
        #                     source = 'user_session'
        #                 if target == 'user':
        #                     target = 'user_session'
                        
        #                 # Only add relation if both entities exist in our extracted entities
        #                 if (source in entity_contents or source == 'user_session') and \
        #                    (target in entity_contents or target == 'user_session'):
        #                     confidence = 0.8
        #                     relations.append((source, relation_type, target, confidence))
        #                     logger.debug(f"Pattern {pattern} matched source={source}, target={target}")

        #             except (IndexError, AttributeError) as e:
        #                 logger.warning(f"Pattern {pattern} failed to extract groups: {e}")
        #                 continue
        
        # if not relations:
        #     for e in entity_contents:
        #         if "i like" in text_lower and e in text_lower:
        #             relations.append(("user_session", RelationType.PREFERS, e, 0.7))
        #     logger.warning(f"[Turn {turn_id}] No pattern-based relations found in text: {text}")
        """Extract relations from text given entity contents"""
        relations = []
        text_lower = text.lower()

        for relation_type, patterns in self.relation_patterns.items():
            # found = False  # track if any match found for this relation type

            for pattern, source_group, target_group in patterns:
                try:
                    matches = re.finditer(pattern, text_lower)
                except re.error as e:
                    logger.warning(f"[Turn {turn_id}] Invalid regex for {relation_type.value}: {pattern} ({e})")
                    continue

                for match in matches:
                    try:
                        # ---- Source extraction ----
                        if isinstance(source_group, str):
                            if source_group == 'user':
                                source = 'user_session'
                            else:
                                source = match.groupdict().get(source_group)
                        else:
                            source = match.group(source_group)

                        # ---- Target extraction ----
                        if isinstance(target_group, str):
                            if target_group == 'user':
                                target = 'user_session'
                            else:
                                target = match.groupdict().get(target_group)
                        else:
                            target = match.group(target_group)

                        # ---- Sanity checks ----
                        if not source or not target:
                            logger.debug(
                                f"Pattern {pattern} matched '{match.group(0)}' "
                                f"but failed to extract valid source/target: ({source}, {target})"
                            )
                            continue

                        # ---- Entity filtering ----
                        # if (source in entity_contents or source == 'user_session') and \
                        # (target in entity_contents or target == 'user_session'):
                        #     confidence = 0.8
                        #     relations.append((source, relation_type, target, confidence))
                        #     found = True
                        #     logger.debug(
                        #         f"[RelationExtractor] Matched {relation_type.value}: "
                        #         f"{source} -> {target} | Pattern: '{pattern}'"
                        #     )
                        source_match = None
                        target_match = None

                        if source != 'user_session':
                            source_match = self._match_entity(source, entity_contents)
                        else:
                            source_match = 'user_session'

                        if target != 'user_session':
                            target_match = self._match_entity(target, entity_contents)
                        else:
                            target_match = 'user_session'

                        if source_match and target_match:
                            confidence = 0.8
                            relations.append((source_match, relation_type, target_match, confidence))
                            found = True
                            logger.debug(
                                f"[RelationExtractor] Fuzzy-matched {relation_type.value}: "
                                f"{source_match} -> {target_match} (from {source}->{target})"
                            )

                    except (IndexError, AttributeError) as e:
                        logger.warning(
                            f"Pattern {pattern} failed to extract groups: {e}"
                        )
                        continue

            # if not found:
            #     logger.warning(f"[Turn ID: {turn_id}] No pattern-based relations found for type {relation_type.value} in text: {text}")
        return relations
    
    def _match_entity(self, candidate: str, entities: List[str], threshold: float = 0.6) -> Optional[str]:
        """Return the best-matching entity string from entity_contents for a candidate"""
        candidate = candidate.lower().strip()
        best_match = None
        best_score = 0.0

        for e in entities:
            e_clean = e.lower().strip()
            # Simple containment bonus
            if candidate in e_clean or e_clean in candidate:
                return e  # direct partial match, fast exit

            # Fuzzy match using sequence ratio
            score = SequenceMatcher(None, candidate, e_clean).ratio()
            if score > best_score:
                best_match = e
                best_score = score

        return best_match if best_score >= threshold else None


# class ConstraintExtractor:
#     """Extract constraints and preferences from user utterances"""

#     def __init__(self):
#         self.constraint_patterns = [
#             (r"don't\s+(?:use|add|include)\s+(\w+)", "avoid", 1),
#             (r"no\s+(\w+)", "avoid", 1),  
#             (r"without\s+(\w+)", "avoid", 1),
#             (r"allergic\s+to\s+(\w+)", "allergic", 1),
#             (r"never\s+(\w+)", "avoid", 1),
#             (r"must\s+(?:have|include)\s+(\w+)", "require", 1),
#             (r"need\s+(\w+)", "require", 1),
#             (r"only\s+(\w+)", "only", 1)
#         ]

#     def extract(self, text: str) -> List[Tuple[str, str, float]]:
#         """Extract constraints from text"""
#         constraints = []
#         text_lower = text.lower()

#         for pattern, constraint_type, group_idx in self.constraint_patterns:
#             matches = re.finditer(pattern, text_lower)
#             for match in matches:
#                 try:
#                     item = match.group(group_idx)
#                     confidence = 0.9  # High confidence for constraint patterns
#                     item_normalized = self._normalize_content(item)
#                     constraints.append((constraint_type, item_normalized, confidence))
#                     # constraints.append((constraint_type, item, confidence))
#                 except IndexError:
#                     continue

#         return constraints
    
#     def _normalize_content(self, content: str) -> str:
#         """Lowercase and strip punctuation for consistent matching"""
#         return re.sub(r"[^\w\s]", "", content.lower().strip())

def _normalize_content(text: str) -> str:
        """Normalize text for deduplication consistency."""
        text = text.strip().lower()
        # Remove common determiners and punctuation noise
        for token in ["the ", "a ", "an "]:
            text = text.replace(token, "")
        text = text.replace(":", "").replace(",", "").strip()
        # Lemmatize word by word
        # text = " ".join(self.lemmatizer.lemmatize(word) for word in text.split())
        return text

import re
import logging
from typing import List, Tuple, Optional
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)

class ConstraintExtractor:
    """Extract constraints (avoid, require, only, allergic) and map them to entities."""

    def __init__(self):
        # --- Multi-word, robust constraint patterns ---
        self.constraint_patterns = [
            # Avoidance
            (r"(?:don't|do not|shouldn't)\s+(?:use|add|include|put|mix|contain)\s+(?P<item>[\w\s]+?)(?:[.,]|$)", "avoid"),
            (r"\bno\s+(?P<item>[\w\s]+?)(?:[.,]|$)", "avoid"),
            (r"\bwithout\s+(?P<item>[\w\s]+?)(?:[.,]|$)", "avoid"),
            (r"\bnever\s+(?P<item>[\w\s]+?)(?:[.,]|$)", "avoid"),
            (r"\bexclude\s+(?P<item>[\w\s]+?)(?:[.,]|$)", "avoid"),

            # Allergies
            (r"\ballergic\s+to\s+(?P<item>[\w\s]+?)(?:[.,]|$)", "allergic"),

            # Requirements
            (r"\bmust\s+(?:have|include|contain)\s+(?P<item>[\w\s]+?)(?:[.,]|$)", "require"),
            (r"\bneed(?:\s+to)?\s+(?:add|include|have)?\s*(?P<item>[\w\s]+?)(?:[.,]|$)", "require"),
            (r"\brequire\s+(?P<item>[\w\s]+?)(?:[.,]|$)", "require"),

            # Exclusivity
            (r"\bonly\s+(?P<item>[\w\s]+?)(?:[.,]|$)", "only"),
        ]

    # def extract(self, text: str, entity_contents: List[str],turn_id) -> List[Tuple[str, str, str, float]]:
    #     """
    #     Extracts constraints and optionally links them to existing entities.
    #     Returns list of (source, relation_type, target, confidence)
    #     """
    #     constraints = []
    #     text_lower = text.lower()

    #     for pattern, constraint_type in self.constraint_patterns:
    #         try:
    #             matches = re.finditer(pattern, text_lower)
    #         except re.error as e:
    #             logger.warning(f"Invalid regex pattern '{pattern}': {e}")
    #             continue

    #         for match in matches:
    #             try:
    #                 item = match.group("item")
    #                 if not item:
    #                     continue

    #                 # Clean up extracted phrase
    #                 item = re.sub(r"[^\w\s]", "", item).strip()
    #                 if not item:
    #                     continue

    #                 # Fuzzy match item to known entities
    #                 target_entity = self._match_entity(item, entity_contents)
    #                 if target_entity is None:
    #                     target_entity = item  # fallback to text form

    #                 confidence = 0.9
    #                 constraints.append(("user_session", constraint_type, target_entity, confidence))

    #                 logger.debug(
    #                     f"[ConstraintExtractor] Extracted {constraint_type}: "
    #                     f"user_session -> {target_entity} (from '{item}')"
    #                 )
    #             except Exception as e:
    #                 logger.warning(f"Failed to extract constraint from '{match.group(0)}': {e}")

    #     if not constraints:
    #         logger.warning(f"[Turn ID: {turn_id}]No constraints found in text: {text}")

    #     return constraints

    # # --- Fuzzy matcher identical to RelationExtractor ---
    # def _match_entity(self, candidate: str, entities: List[str], threshold: float = 0.6) -> Optional[str]:
    #     candidate = candidate.lower().strip()
    #     best_match = None
    #     best_score = 0.0

    #     for e in entities:
    #         e_clean = e.lower().strip()

    #         if candidate in e_clean or e_clean in candidate:
    #             return e  # direct partial match (fast exit)

    #         score = SequenceMatcher(None, candidate, e_clean).ratio()
    #         if score > best_score:
    #             best_match = e
    #             best_score = score

    #     return best_match if best_score >= threshold else None
    def extract(self, text: str,turn_id: int) -> List[Tuple[str, str, float]]:
        """Extract constraints from text"""
        constraints = []
        text_lower = text.lower()

        for pattern, constraint_type in self.constraint_patterns:
            try:
                matches = re.finditer(pattern, text_lower)
            except re.error as e:
                logger.warning(f"Invalid regex in constraint pattern '{pattern}': {e}")
                continue

            for match in matches:
                try:
                    item = match.group("item")
                    if not item:
                        continue
                    # Clean and normalize
                    item = re.sub(r"[^\w\s]", "", item).strip()
                    if not item:
                        continue
                    confidence = 0.9
                    constraints.append((constraint_type, item, confidence))
                    logger.debug(f"[ConstraintExtractor] Extracted {constraint_type}: {item}")
                except Exception as e:
                    logger.warning(f"[Turn ID: {turn_id}]Failed to extract constraint from '{match.group(0)}': {e}")

        return constraints




class SSMGExtractor:
    """Main extractor class for SSMG"""

    def __init__(self, spacy_model: str = "en_core_web_sm", intent_backend: str = "transformer", 
                 constraint_backend: str = "nltk", #type: ignore
                 relation_backend: str = "nltk"): #type: ignore
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logger.warning(f"Could not load {spacy_model}, using blank model")
            self.nlp = spacy.blank("en")

        # --- Intent classifier selection ---
        # if intent_backend == "fasttext" and fasttext_model_path is not None:
        #     self.intent_classifier = FastTextIntentClassifier(fasttext_model_path)
        self.intent_backend = intent_backend
        self.constraint_backend = constraint_backend    
        self.relation_backend = relation_backend

        
        if self.intent_backend == "transformer":
            self.intent_classifier = DistilBERTIntentClassifier()
        else:
            self.intent_backend = "simple"
            self.intent_classifier = IntentClassifier()

        # --- Constraint extractor selection ---
        if self.constraint_backend == "nltk":
            self.constraint_extractor = NLTKConstraintExtractor()
        else:
            self.constraint_backend = "pattern"
            self.constraint_extractor = ConstraintExtractor()

        # --- Relation extractor selection ---
        if self.relation_backend == "nltk":
            self.relation_extractor = NLTKRelationExtractor()
        else:
            self.relation_backend = "pattern"
            self.relation_extractor = RelationExtractor()

        # Entity types to focus on
        self.relevant_ent_types = {
            'PERSON', 'ORG', 'PRODUCT', 'FOOD', 'GPE', 
            'MONEY', 'QUANTITY', 'TIME', 'DATE'
        }

        logger.debug(f"Extractor Initialised with Intent: {self.intent_backend}, Constraint: {self.constraint_backend}, Relation: {self.relation_backend}")

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
        relations = self.relation_extractor.extract(text, list(node_id_map.keys()),turn_id)
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
        if not relations:
            logger.warning(f"[Turn {turn_id}] No pattern-based relations found in text: {text}")
        # Calculate overall confidence
        if nodes:
            avg_confidence = sum(node.confidence for node in nodes) / len(nodes)
        else:
            avg_confidence = 0.5
        
        logger.info(f"[Turn ID: {turn_id}] Nodes Added: {len(nodes)} Edges Added: {len(edges)} Avg Confidence:{avg_confidence}")

        return ExtractionResult(
            nodes=nodes,
            edges=edges,
            confidence=avg_confidence,
            metadata=metadata
        )
    
    
    def extract_and_infer(self, text: str, turn_id: int = 0, user_node: Node = None) -> ExtractionResult: #type: ignore
        """Extract nodes and edges from text, then infer missing relations with logging"""
        doc = self.nlp(text)

        nodes = []
        edges = []
        metadata = {
            'text': text,
            'turn_id': turn_id,
            'user_node': user_node,
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
        existing_entities = {n.content for n in entities}
        fact_nodes = self._extract_facts(doc, turn_id,existing_entities)
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
        # node_id_map['user'] = 'user_session'
        # node_id_map['user_session'] = 'user_session'  # Also map the actual ID
        # --- Step 2: Extract edges from relation extractor ---
        relations = self.relation_extractor.extract(text, list(node_id_map.keys()),turn_id)
        # if not relations:
        #     logger.warning(f"[Turn {turn_id}] No pattern-based relations found in text: {text}")
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
                logger.debug(f"[Turn ID: {turn_id}] Extracted edge: {source_content} -> {target_content} ({relation_type.value})")

        # if not relations:
        #     # --- Step 3: Rule-based inference for missing edges ---
        #     # (a) Intent -> other nodes
        #     intent_nodes = [n for n in nodes if n.type == NodeType.INTENT]
        #     for intent in intent_nodes:
        #         for target in nodes:
        #             if target.id == intent.id:
        #                 continue

        #             # Infer appropriate relation type
        #             if target.type == NodeType.ENTITY:
        #                 relation = RelationType.MENTIONS
        #             elif target.type == NodeType.FACT:
        #                 relation = RelationType.ASKED_ABOUT
        #             elif target.type == NodeType.CONSTRAINT:
        #                 relation = RelationType.PREFERS
        #             else:
        #                 continue

        #             inferred_edge = Edge(
        #                 source_id=intent.id,
        #                 target_id=target.id,
        #                 relation=relation,
        #                 confidence=0.6,
        #                 turn_id=turn_id
        #             )
        #             edges.append(inferred_edge)
        #             logger.debug(f"[Turn ID: {turn_id}] Inferred edge (intent->{target.type.value}): "
        #                         f"{intent.content} -> {target.content} ({relation.value})")

        #     # (b) Constraint -> Entity or Fact
        #     constraint_nodes = [n for n in nodes if n.type == NodeType.CONSTRAINT]
        #     for constraint in constraint_nodes:
        #         for target in nodes:
        #             if target.id == constraint.id:
        #                 continue

        #             if target.type == NodeType.ENTITY:
        #                 relation = RelationType.PREFERS
        #             elif target.type == NodeType.FACT:
        #                 relation = RelationType.CONFLICTS_WITH
        #             else:
        #                 continue

        #             inferred_edge = Edge(
        #                 source_id=constraint.id,
        #                 target_id=target.id,
        #                 relation=relation,
        #                 confidence=0.7,
        #                 turn_id=turn_id
        #             )
        #             edges.append(inferred_edge)
        #             logger.debug(f"[Turn {turn_id}] Inferred edge (constraint->{target.type.value}): "
        #                         f"{constraint.content} -> {target.content} ({relation.value})")
        # if not relations:
        #     # --- Step 3: Rule-based inference for missing edges ---

        #     # 0️⃣ Identify the user node
        #     user_nodes = [n for n in nodes if n.type == NodeType.ENTITY and n.content.lower() == "user"]
        #     user_node = user_nodes[0] if user_nodes else None

        #     # 1️⃣ Link Intents to all relevant nodes (entities/facts/constraints) across the entire graph
        #     intent_nodes = [n for n in nodes if n.type == NodeType.INTENT]
        #     for intent in intent_nodes:
        #         for target in nodes:
        #             if target.id == intent.id:
        #                 continue

        #             if target.type == NodeType.ENTITY:
        #                 relation = RelationType.MENTIONS
        #             elif target.type == NodeType.FACT:
        #                 relation = RelationType.ASKED_ABOUT
        #             elif target.type == NodeType.CONSTRAINT:
        #                 relation = RelationType.PREFERS
        #             else:
        #                 continue

        #             edge_id = f"{intent.id}:{relation}:{target.id}"
        #             if edge_id not in {e.source_id + ":" + e.relation.value + ":" + e.target_id for e in edges}:
        #                 inferred_edge = Edge(
        #                     source_id=intent.id,
        #                     target_id=target.id,
        #                     relation=relation,
        #                     confidence=0.6,
        #                     turn_id=turn_id
        #                 )
        #                 edges.append(inferred_edge)
        #                 logger.debug(f"[Turn ID: {turn_id}] Inferred edge (intent->{target.type.value}): "
        #                             f"{intent.content} -> {target.content} ({relation.value})")

        #     # 2️⃣ Link Constraints to all relevant entities/facts across turns
        #     constraint_nodes = [n for n in nodes if n.type == NodeType.CONSTRAINT]
        #     for constraint in constraint_nodes:
        #         for target in nodes:
        #             if target.id == constraint.id:
        #                 continue

        #             if target.type == NodeType.ENTITY:
        #                 relation = RelationType.PREFERS
        #             elif target.type == NodeType.FACT:
        #                 relation = RelationType.CONFLICTS_WITH
        #             else:
        #                 continue

        #             edge_id = f"{constraint.id}:{relation}:{target.id}"
        #             if edge_id not in {e.source_id + ":" + e.relation.value + ":" + e.target_id for e in edges}:
        #                 inferred_edge = Edge(
        #                     source_id=constraint.id,
        #                     target_id=target.id,
        #                     relation=relation,
        #                     confidence=0.7,
        #                     turn_id=turn_id
        #                 )
        #                 edges.append(inferred_edge)
        #                 logger.debug(f"[Turn {turn_id}] Inferred edge (constraint->{target.type.value}): "
        #                             f"{constraint.content} -> {target.content} ({relation.value})")

        #     # 3️⃣ Link entities/facts with similar content across turns
        #     entity_fact_nodes = [n for n in nodes if n.type in {NodeType.ENTITY, NodeType.FACT}]
        #     for i, node_a in enumerate(entity_fact_nodes):
        #         for node_b in entity_fact_nodes[i+1:]:
        #             if node_a.id == node_b.id:
        #                 continue

        #             content_a = node_a.content.lower()
        #             content_b = node_b.content.lower()
        #             if content_a in content_b or content_b in content_a:
        #                 relation = RelationType.RELATED
        #                 edge_id = f"{node_a.id}:{relation}:{node_b.id}"
        #                 if edge_id not in {e.source_id + ":" + e.relation.value + ":" + e.target_id for e in edges}:
        #                     inferred_edge = Edge(
        #                         source_id=node_a.id,
        #                         target_id=node_b.id,
        #                         relation=relation,
        #                         confidence=0.5,
        #                         turn_id=turn_id
        #                     )
        #                     edges.append(inferred_edge)
        #                     logger.debug(f"[Turn {turn_id}] Inferred edge (related): {node_a.content} -> {node_b.content} ({relation.value})")

        #     # 4️⃣ Link user -> all entities/facts/constraints in current turn that aren't already linked
        #     if user_node:
        #         for target in nodes:
        #             if target.id == user_node.id:
        #                 continue

        #             # Decide relation type
        #             if target.type == NodeType.ENTITY:
        #                 relation = RelationType.PREFERS
        #             elif target.type == NodeType.FACT:
        #                 relation = RelationType.MENTIONS
        #             elif target.type == NodeType.CONSTRAINT:
        #                 relation = RelationType.PREFERS
        #             else:
        #                 continue

        #             edge_id = f"{user_node.id}:{relation}:{target.id}"
        #             if edge_id not in {e.source_id + ":" + e.relation.value + ":" + e.target_id for e in edges}:
        #                 inferred_edge = Edge(
        #                     source_id=user_node.id,
        #                     target_id=target.id,
        #                     relation=relation,
        #                     confidence=0.65,
        #                     turn_id=turn_id
        #                 )
        #                 edges.append(inferred_edge)
        #                 logger.debug(f"[Turn {turn_id}] Inferred edge (user->{target.type.value}): {user_node.content} -> {target.content} ({relation.value})")
        # --- Step 3: Rule-based inference for missing edges ---
        if not relations:
            logger.warning(f"[Turn {turn_id}] No {self.relation_backend}-based relations found in text: {text}")
            # Helper: Check if two nodes are meaningfully related
            def is_related(source: Node, target: Node) -> bool:
                s, t = source.content.lower(), target.content.lower()
                # Exact or partial match
                if s in t or t in s:
                    return True
                # Avoid self-loop and spurious constraint->entity links
                if source.type.value == NodeType.CONSTRAINT.value and t.startswith(s.split(":")[1].strip()):
                    return False
                return False

            # --- 1️⃣ Link intents ---
            intent_nodes = [n for n in nodes if n.type.value == NodeType.INTENT.value]
            for intent in intent_nodes:
                for target in nodes:
                    if target.id == intent.id:
                        continue
                    # Only link nearby turns (current or previous turn)
                    if abs(target.turn_id - intent.turn_id) > 1:
                        continue
                    if target.type.value == NodeType.ENTITY.value:
                        relation = RelationType.MENTIONS
                    elif target.type.value == NodeType.FACT.value:
                        relation = RelationType.ASKED_ABOUT
                    elif target.type.value == NodeType.CONSTRAINT.value:
                        relation = RelationType.PREFERS
                    else:
                        continue

                    if is_related(intent, target):
                        edge_id = f"{intent.id}:{relation}:{target.id}"
                        if edge_id not in {e.source_id + ":" + e.relation.value + ":" + e.target_id for e in edges}:
                            inferred_edge = Edge(
                                source_id=intent.id,
                                target_id=target.id,
                                relation=relation,
                                confidence=0.6,
                                turn_id=turn_id
                            )
                            edges.append(inferred_edge)
                            logger.debug(f"[Turn ID: {turn_id}] Inferred edge (intent->{target.type.value}): "
                                        f"{intent.content} -> {target.content} ({relation.value})")

            # --- 2️⃣ Link constraints ---
            constraint_nodes = [n for n in nodes if n.type == NodeType.CONSTRAINT]
            for constraint in constraint_nodes:
                for target in nodes:
                    if target.id == constraint.id:
                        continue
                    # Only link if related content exists
                    if not any(word in target.content.lower() for word in constraint.content.lower().split()):
                        continue

                    if target.type.value == NodeType.ENTITY.value:
                        relation = RelationType.PREFERS
                    elif target.type.value == NodeType.FACT.value:
                        relation = RelationType.CONFLICTS_WITH
                    else:
                        continue

                    edge_id = f"{constraint.id}:{relation}:{target.id}"
                    if edge_id not in {e.source_id + ":" + e.relation.value + ":" + e.target_id for e in edges}:
                        inferred_edge = Edge(
                            source_id=constraint.id,
                            target_id=target.id,
                            relation=relation,
                            confidence=0.7,
                            turn_id=turn_id
                        )
                        edges.append(inferred_edge)
                        logger.debug(f"[Turn {turn_id}] Inferred edge (constraint->{target.type.value}): "
                                    f"{constraint.content} -> {target.content} ({relation.value})")

            # --- 3️⃣ Link related entities/facts across turns ---
            # entity_fact_nodes = [n for n in nodes if n.type.value in {NodeType.ENTITY.value, NodeType.FACT.value}]
            # for i, node_a in enumerate(entity_fact_nodes):
            #     for node_b in entity_fact_nodes[i + 1:]:
            #         if node_a.id == node_b.id:
            #             continue
            #         if node_a.content.lower() in node_b.content.lower() or node_b.content.lower() in node_a.content.lower():
            #             relation = RelationType.RELATED
            #             edge_id = f"{node_a.id}:{relation}:{node_b.id}"
            #             if edge_id not in {e.source_id + ":" + e.relation.value + ":" + e.target_id for e in edges}:
            #                 inferred_edge = Edge(
            #                     source_id=node_a.id,
            #                     target_id=node_b.id,
            #                     relation=relation,
            #                     confidence=0.5,
            #                     turn_id=turn_id
            #                 )
            #                 edges.append(inferred_edge)
            #                 logger.debug(f"[Turn {turn_id}] Inferred edge (related): {node_a.content} -> {node_b.content} ({relation.value})")

            # --- 4️⃣ Link user node to nearby entities/facts/constraints ---
            # --- Identify the user node ---
            # user_nodes = [n for n in nodes if n.type.value == NodeType.ENTITY.value and (n.id.lower() == "user" or n.id.lower() == "user_session")]
            # user_node = user_nodes[0] if user_nodes else None
            
        if user_node is not None:
            for target in nodes:
                if target.id == user_node.id:
                    continue
                if target.turn_id != turn_id:
                    continue

                if target.type.value == NodeType.ENTITY.value:
                    relation = RelationType.PREFERS
                elif target.type.value == NodeType.FACT.value:
                    relation = RelationType.MENTIONS
                elif target.type.value == NodeType.CONSTRAINT.value:
                    relation = RelationType.PREFERS
                else:
                    continue

                edge_id = f"{user_node.id}:{relation}:{target.id}"
                if edge_id not in {e.source_id + ":" + e.relation.value + ":" + e.target_id for e in edges}:
                    inferred_edge = Edge(
                        source_id=user_node.id,
                        target_id=target.id,
                        relation=relation,
                        confidence=0.7,
                        turn_id=turn_id
                    )
                    edges.append(inferred_edge)
                    logger.debug(f"[Turn {turn_id}] Inferred edge (user->{target.type.value}): "
                                f"{user_node.content} -> {target.content} ({relation.value})")

        if not edges:
            logger.warning(f"[Turn ID: {turn_id}] No edges (extracted or inferred) were created for text: {text} falling back")
            for src in nodes:
                for tgt in nodes:
                    if src.id != tgt.id:
                        relation = relation_map.get(src.type, {}).get(tgt.type)
                        if relation is None:
                            continue
                        edges.append(Edge(
                            source_id=src.id,
                            target_id=tgt.id,
                            relation=relation,
                            confidence=min(src.confidence, tgt.confidence)
                        ))
        # --- Step 4: Calculate average confidence ---
        avg_confidence = (
            sum(node.confidence for node in nodes) / len(nodes)
            if nodes else 0.5
        )

        unique_edges = {}
        for e in edges:
            key = (e.source_id, e.target_id, e.relation)
            if key not in unique_edges or e.confidence > unique_edges[key].confidence:
                unique_edges[key] = e
        edges = list(unique_edges.values())

        logger.info(f"[Turn ID: {turn_id}] (Modified) Nodes Added: {len(nodes)} Edges Added: {len(edges)} Avg Confidence:{avg_confidence:5f}")

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
                    id=f"entity_{ent.text.lower().replace(' ', '_')}", #id=f"entity_{ent.text.lower()}_{turn_id}",
                    type=NodeType.ENTITY,
                    content=_normalize_content(ent.text),
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
                        id=f"entity_{token.text.lower().replace(' ', '_')}", #id=f"entity_{ent.text.lower()}_{turn_id}",
                        type=NodeType.ENTITY,
                        content=_normalize_content(token.text),
                        confidence=0.6,  # Lower confidence for simple nouns
                        turn_id=turn_id,
                        metadata={'pos': token.pos_, 'lemma': token.lemma_}
                    )
                    entities.append(node)

        return entities

    def _extract_intent(self, text: str, turn_id: int) -> Optional[Node]:
        """Extract user intent from text"""
        intent, confidence = self.intent_classifier.classify(text)
        
        if intent and intent.lower != 'unknown':
            return Node(
                id=f"intent_{intent}", #id=f"intent_{intent}_{turn_id}",
                type=NodeType.INTENT,
                content=_normalize_content(intent),
                confidence=confidence,
                turn_id=turn_id,
                metadata={'raw_text': text}
            )
        return None

    def _extract_constraints(self, text: str, turn_id: int) -> List[Node]:
        """Extract constraints from text"""
        constraints = []
        extracted = self.constraint_extractor.extract(text,turn_id)

        for constraint_type, item, confidence in extracted:
            node = Node(
                id=f"constraint_{constraint_type}_{item.lower().replace(' ', '_')}", #id=f"constraint_{constraint_type}_{item}_{turn_id}",
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

    def _extract_facts(self, doc, turn_id:int,existing_entities: Set[str] = None) -> List[Node]: #type: ignore
        """Extract factual statements from text"""
        facts = []
        existing_entities = existing_entities or set()

        for chunk in doc.noun_chunks:
            content_norm = _normalize_content(chunk.text)

            # Skip short phrases, stop words, proper nouns, or existing entities
            if (len(chunk.text) <= 3 or
                chunk.root.is_stop or
                chunk.root.pos_ == "PROPN" or
                content_norm in existing_entities):
                continue

            node = Node(
                id=f"fact_{content_norm}",
                type=NodeType.FACT,
                content=content_norm,
                confidence=0.5,
                turn_id=turn_id,
                metadata={
                    'chunk_root': chunk.root.text,
                    'start': chunk.start_char,
                    'end': chunk.end_char
                }
            )
            facts.append(node)

        return facts
        # facts = []

        # # Simple heuristic: extract noun phrases as potential facts
        # for chunk in doc.noun_chunks:
        #     if len(chunk.text) > 3 and not chunk.root.is_stop:
        #         node = Node(
        #             id=f"fact_{chunk.text.lower().replace(' ', '_')}", #id=f"fact_{chunk.text.lower().replace(' ', '_')}_{turn_id}",
        #             type=NodeType.FACT,
        #             content=_normalize_content(chunk.text),
        #             confidence=0.5,  # Low confidence for simple facts
        #             turn_id=turn_id,
        #             metadata={
        #                 'chunk_root': chunk.root.text,
        #                 'start': chunk.start_char,
        #                 'end': chunk.end_char
        #             }
        #         )
        #         facts.append(node)

        # return facts

    # def extract_updates(self, text: str, existing_nodes: List[str], turn_id: int = 0) -> ExtractionResult:
    #     """Extract incremental updates given existing context"""
    #     # For now, same as regular extract but could be enhanced
    #     # to better handle references and updates
    #     result = self.extract(text, turn_id)

    #     # Add metadata about potential updates/references
    #     result.metadata['existing_context'] = existing_nodes
    #     result.metadata['is_update'] = any(
    #         word in text.lower() 
    #         for word in ['change', 'instead', 'actually', 'modify', 'update']
    #     )

    #     return result

