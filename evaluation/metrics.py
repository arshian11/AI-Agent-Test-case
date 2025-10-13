"""
Evaluation metrics for SSMG and baselines
"""

import re
from typing import List, Optional,Dict, Any, Tuple, Set
from dataclasses import dataclass
import numpy as np
import datetime
from collections import defaultdict
import json

@dataclass
class EvaluationMetrics:
    """Container for all evaluation metrics"""
    turn_accuracy: float
    task_success_rate: float
    avg_context_tokens: float
    avg_latency: float
    reference_resolution_accuracy: float
    constraint_adherence: float
    token_efficiency: float  # Accuracy per token
    latency_efficiency: float  # Accuracy per second

# class GroundTruthLoader:
#     """Loads ground truth data for evaluation"""

#     def __init__(self):
#         # Mock ground truth - replace with actual dataset loading
#         self.ground_truth = {
#             'intents': {
#                 'i want pizza': 'order',
#                 'change to pasta': 'modify',
#                 'add garlic bread': 'add',
#                 'no onions please': 'constraint'
#             },
#             'entities': {
#                 'i want pizza': ['pizza'],
#                 'change to pasta': ['pasta'],
#                 'add garlic bread': ['garlic bread'],
#                 'no onions please': ['onions']
#             },
#             'constraints': {
#                 'no onions please': [('avoid', 'onions')],
#                 'i am allergic to nuts': [('allergic', 'nuts')]
#             }
#         }

#     def get_intent(self, utterance: str) -> str:
#         return self.ground_truth['intents'].get(utterance.lower(), 'unknown')

#     def get_entities(self, utterance: str) -> List[str]:
#         return self.ground_truth['entities'].get(utterance.lower(), [])

#     def get_constraints(self, utterance: str) -> List[Tuple[str, str]]:
#         return self.ground_truth['constraints'].get(utterance.lower(), [])

# class ReferenceResolver:
#     """Evaluates reference resolution accuracy"""

#     def __init__(self):
#         self.pronouns = {'it', 'he', 'she', 'they', 'this', 'that', 'them'}
#         self.references = {'the order', 'my order', 'the pizza', 'the pasta'}

#     def extract_references(self, text: str) -> List[str]:
#         """Extract pronouns and references from text"""
#         words = text.lower().split()
#         refs = []

#         for word in words:
#             if word in self.pronouns:
#                 refs.append(word)

#         for ref in self.references:
#             if ref in text.lower():
#                 refs.append(ref)

#         return refs

#     def evaluate_resolution(self, dialogue_turns: List[Dict], responses: List[str]) -> float:
#         """Evaluate how well references are resolved in responses"""
#         total_refs = 0
#         resolved_refs = 0

#         for i, (turn, response) in enumerate(zip(dialogue_turns, responses)):
#             user_input = turn.get('user_input', '')
#             refs = self.extract_references(user_input)

#             if not refs:
#                 continue

#             total_refs += len(refs)

#             # Simple heuristic: response should contain specific entities, not just pronouns
#             response_words = response.lower().split()
#             specific_words = [w for w in response_words if w not in self.pronouns and len(w) > 3]

#             if len(specific_words) > 2:  # Reasonable specificity
#                 resolved_refs += len(refs)

#         return resolved_refs / total_refs if total_refs > 0 else 1.0

# class ConstraintChecker:
#     """Evaluates constraint adherence in responses"""

#     def __init__(self):
#         self.constraint_patterns = {
#             'avoid': ['avoid', 'without', 'no', "don't"],
#             'allergic': ['allergic', 'allergy'],
#             'require': ['must', 'need', 'require'],
#             'only': ['only', 'just']
#         }

#     def extract_constraints_from_turns(self, turns: List[Dict]) -> Dict[str, Set[str]]:
#         """Extract constraints mentioned across all turns"""
#         constraints = defaultdict(set)

#         for turn in turns:
#             user_input = turn.get('user_input', '').lower()

#             # Look for constraint patterns
#             if any(word in user_input for word in ['no', "don't", 'without', 'avoid']):
#                 # Extract what to avoid
#                 words = user_input.split()
#                 for i, word in enumerate(words):
#                     if word in ['no', "don't", 'without', 'avoid'] and i + 1 < len(words):
#                         item = words[i + 1]
#                         constraints['avoid'].add(item)

#             if 'allergic' in user_input:
#                 words = user_input.split()
#                 for i, word in enumerate(words):
#                     if word == 'to' and i + 1 < len(words):
#                         item = words[i + 1]
#                         constraints['allergic'].add(item)

#         return constraints

#     def check_adherence(self, constraints: Dict[str, Set[str]], response: str) -> float:
#         """Check if response adheres to constraints"""
#         violations = 0
#         total_constraints = sum(len(items) for items in constraints.values())

#         if total_constraints == 0:
#             return 1.0

#         response_lower = response.lower()

#         # Check avoidance constraints
#         for item in constraints.get('avoid', set()):
#             if item in response_lower:
#                 violations += 1

#         # Check allergy constraints  
#         for item in constraints.get('allergic', set()):
#             if item in response_lower:
#                 violations += 1

#         adherence_rate = (total_constraints - violations) / total_constraints
#         return max(0.0, adherence_rate)

# class TaskSuccessEvaluator:
#     """Evaluates task completion success"""

#     def __init__(self):
#         # self.task_keywords = {
#         #     'order': ['order', 'get', 'buy', 'purchase'],
#         #     'modify': ['change', 'modify', 'update', 'instead'],
#         #     'add': ['add', 'include', 'also', 'plus'],
#         #     'cancel': ['cancel', 'remove', 'delete']
#         # }
#         self.task_keywords = {
#             "inform": ["inform", "tell", "provide", "give", "state"],
#             "request": ["request", "ask", "need", "want", "provide"],
#             "book": ["book", "reserve", "schedule", "confirm", "make reservation"],
#             "select": ["select", "choose", "pick"],
#             "recommend": ["recommend", "suggest", "advise"],
#             "no_offer": ["not available", "no offer", "none", "sorry"],
#             "no_book": ["cannot book", "not booked", "fail", "cannot reserve"],
#             "offer_book": ["offer booked", "offer reservation", "offer confirmed"],
#             "offer_booked": ["booked", "confirmed"],
#             'cancel': ['cancel', 'remove', 'delete']
#         }

#     def evaluate_task_success(self, intent: str, response: str) -> bool:
#         """Evaluate if response successfully addresses the intent"""
#         if intent == 'unknown':
#             return True  # Can't evaluate unknown intents

#         response_lower = response.lower()

#         # Check if response contains appropriate keywords for the intent
#         if intent in self.task_keywords:
#             keywords = self.task_keywords[intent]
#             return any(keyword in response_lower for keyword in keywords)

#         # Default heuristic - response should be substantive
#         return len(response.split()) > 3

# MULTIWOZ_INTENTS = [
#     'Attraction-Inform', 'Attraction-NoOffer', 'Attraction-Recommend', 'Attraction-Request', 'Attraction-Select',
#     'Booking-Book', 'Booking-Inform', 'Booking-NoBook', 'Booking-Request',
#     'Hospital-Inform', 'Hospital-Request',
#     'Hotel-Inform', 'Hotel-NoOffer', 'Hotel-Recommend', 'Hotel-Request', 'Hotel-Select',
#     'Police-Inform', 'Police-Request',
#     'Restaurant-Inform', 'Restaurant-NoOffer', 'Restaurant-Recommend', 'Restaurant-Request', 'Restaurant-Select',
#     'Taxi-Inform', 'Taxi-Request',
#     'Train-Inform', 'Train-NoOffer', 'Train-OfferBook', 'Train-OfferBooked', 'Train-Request', 'Train-Select',
#     'general-bye', 'general-greet', 'general-reqmore', 'general-thank', 'general-welcome'
# ]

# class ImprovedGroundTruthLoader:
#     """
#     Loads and manages ground truth data for multi-domain task-oriented dialogue evaluation.
#     Compatible with MultiWOZ dataset structure and supports multiple evaluation scenarios.
#     """
    
#     def __init__(self, dataset_path: Optional[str] = None):
#         self.dataset_path = dataset_path
        
#         # MultiWOZ domain mapping
#         self.domains = {
#             'attraction', 'hotel', 'restaurant', 'taxi', 'train', 'hospital', 'police', 'booking', 'general'
#         }
        
#         # Intent mapping based on provided MultiWOZ intents
#         self.intents = set(MULTIWOZ_INTENTS)
        
#         # Slot definitions for each domain
#         self.domain_slots = {
#             'restaurant': {
#                 'informable': ['food', 'pricerange', 'area', 'name'],
#                 'requestable': ['phone', 'address', 'postcode', 'food', 'pricerange', 'area']
#             },
#             'hotel': {
#                 'informable': ['pricerange', 'type', 'parking', 'book_stay', 'book_day', 'book_people', 'area', 'stars', 'internet'],
#                 'requestable': ['address', 'postcode', 'internet', 'parking', 'type', 'pricerange', 'stars', 'area', 'phone']
#             },
#             'attraction': {
#                 'informable': ['area', 'type', 'name'],
#                 'requestable': ['phone', 'address', 'postcode', 'fee', 'type', 'area', 'openhours']
#             },
#             'train': {
#                 'informable': ['destination', 'day', 'departure', 'arriveby', 'leaveat', 'book_people'],
#                 'requestable': ['trainid', 'duration', 'arriveby', 'leaveat', 'price']
#             },
#             'taxi': {
#                 'informable': ['leaveat', 'destination', 'departure', 'arriveby'],
#                 'requestable': ['type', 'phone']
#             },
#             'hospital': {
#                 'informable': ['department'],
#                 'requestable': ['address', 'phone', 'postcode']
#             },
#             'police': {
#                 'informable': [],
#                 'requestable': ['address', 'phone', 'postcode']
#             }
#         }
        
#         # Initialize with mock data (replace with actual dataset loading)
#         self._initialize_mock_data()
    
#     def _initialize_mock_data(self):
#         """Initialize with representative MultiWOZ-style data structure"""
#         self.ground_truth_intents = {
#             'i want to find a restaurant': 'Restaurant-Inform',
#             'i need a hotel in the centre': 'Hotel-Inform', 
#             'book a table for 2': 'Restaurant-Request',
#             'what is the phone number': 'Restaurant-Request',
#             'thank you goodbye': 'general-thank',
#             'hello i need help': 'general-greet',
#             'are there any chinese restaurants': 'Restaurant-Inform',
#             'no that does not work': 'Restaurant-Inform',
#             'i want something expensive': 'Restaurant-Inform'
#         }
        
#         self.ground_truth_entities = {
#             'i want to find a restaurant': [('restaurant', 'domain')],
#             'i need a hotel in the centre': [('hotel', 'domain'), ('centre', 'area')],
#             'book a table for 2': [('table', 'service'), ('2', 'book_people')],
#             'what is the phone number': [('phone', 'requestable_slot')],
#             'are there any chinese restaurants': [('chinese', 'food'), ('restaurants', 'domain')],
#             'i want something expensive': [('expensive', 'pricerange')]
#         }
        
#         self.ground_truth_dialogue_states = {
#             'turn_1': {
#                 'restaurant': {'food': 'chinese', 'area': 'centre'},
#                 'booking': {}
#             },
#             'turn_2': {
#                 'restaurant': {'food': 'chinese', 'area': 'centre', 'pricerange': 'expensive'},
#                 'booking': {'book_people': '2'}
#             }
#         }
        
#         self.ground_truth_dialogue_acts = {
#             'i want to find a restaurant': [
#                 {'intent': 'Restaurant-Inform', 'domain': 'restaurant', 'slot': 'name', 'value': 'dontcare'}
#             ],
#             'i need a hotel in the centre': [
#                 {'intent': 'Hotel-Inform', 'domain': 'hotel', 'slot': 'area', 'value': 'centre'}
#             ]
#         }
    
#     def load_dataset(self, dataset_path: str, version: str = '2.1') -> Dict[str, Any]:
#         """
#         Load MultiWOZ dataset (placeholder for actual implementation)
        
#         Args:
#             dataset_path: Path to MultiWOZ dataset
#             version: Dataset version ('2.0', '2.1', '2.2', etc.)
            
#         Returns:
#             Loaded dataset structure
#         """
#         # This would implement actual dataset loading
#         # For now, return mock structure
#         return {
#             'train': [],
#             'val': [],
#             'test': [],
#             'ontology': self.domain_slots,
#             'version': version
#         }
    
#     def get_intent(self, utterance: str, context: Optional[Dict] = None) -> str:
#         """Get intent for utterance with optional context"""
#         utterance_lower = utterance.lower().strip()
        
#         # Direct lookup first
#         if utterance_lower in self.ground_truth_intents:
#             return self.ground_truth_intents[utterance_lower]
        
#         # Pattern-based intent detection for unknown utterances
#         return self._predict_intent(utterance_lower, context)
    
#     def _predict_intent(self, utterance: str, context: Optional[Dict] = None) -> str:
#         """Predict intent using rule-based patterns"""
#         domain = self._extract_domain(utterance)
        
#         # Intent prediction patterns
#         if any(word in utterance for word in ['find', 'looking for', 'want', 'need']):
#             return f'{domain}-Inform'
#         elif any(word in utterance for word in ['book', 'reserve']):
#             return f'Booking-Book'
#         elif any(word in utterance for word in ['phone', 'address', 'postcode', 'number']):
#             return f'{domain}-Request'
#         elif any(word in utterance for word in ['recommend', 'suggest']):
#             return f'{domain}-Request'
#         elif any(word in utterance for word in ['thank', 'thanks']):
#             return 'general-thank'
#         elif any(word in utterance for word in ['hello', 'hi', 'help']):
#             return 'general-greet'
#         elif any(word in utterance for word in ['bye', 'goodbye']):
#             return 'general-bye'
#         else:
#             return f'{domain}-Inform'  # Default fallback
    
#     def _extract_domain(self, utterance: str) -> str:
#         """Extract domain from utterance"""
#         domain_keywords = {
#             'restaurant': ['restaurant', 'food', 'eat', 'dinner', 'lunch', 'cuisine'],
#             'hotel': ['hotel', 'accommodation', 'stay', 'room', 'lodge'],
#             'attraction': ['attraction', 'museum', 'park', 'entertainment', 'visit'],
#             'train': ['train', 'railway', 'departure', 'arrive'],
#             'taxi': ['taxi', 'cab', 'car', 'ride'],
#             'hospital': ['hospital', 'medical', 'doctor'],
#             'police': ['police', 'emergency']
#         }
        
#         for domain, keywords in domain_keywords.items():
#             if any(keyword in utterance for keyword in keywords):
#                 return domain.capitalize()
        
#         return 'general'
    
#     def get_entities(self, utterance: str, context: Optional[Dict] = None) -> List[Tuple[str, str]]:
#         """Extract entities from utterance"""
#         utterance_lower = utterance.lower().strip()
        
#         # Direct lookup
#         if utterance_lower in self.ground_truth_entities:
#             return self.ground_truth_entities[utterance_lower]
        
#         # Pattern-based entity extraction
#         return self._extract_entities(utterance_lower)
    
#     def _extract_entities(self, utterance: str) -> List[Tuple[str, str]]:
#         """Extract entities using pattern matching"""
#         entities = []
        
#         # Domain extraction
#         domain = self._extract_domain(utterance).lower()
#         if domain != 'general':
#             entities.append((domain, 'domain'))
        
#         # Common slot values
#         slot_patterns = {
#             'area': ['north', 'south', 'east', 'west', 'centre', 'center'],
#             'pricerange': ['cheap', 'expensive', 'moderate'],
#             'food': ['chinese', 'indian', 'italian', 'british', 'french'],
#             'type': ['hotel', 'guesthouse', 'museum', 'park', 'theatre'],
#             'stars': ['1', '2', '3', '4', '5'],
#             'day': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
#         }
        
#         for slot, values in slot_patterns.items():
#             for value in values:
#                 if value in utterance:
#                     entities.append((value, slot))
        
#         # Number extraction for people, time, etc.
#         numbers = re.findall(r'\b\d+\b', utterance)
#         for num in numbers:
#             if 'people' in utterance or 'person' in utterance:
#                 entities.append((num, 'book_people'))
#             elif any(time_word in utterance for time_word in ['time', 'oclock', ':']):
#                 entities.append((num, 'time'))
        
#         return entities
    
#     def get_dialogue_state(self, turn_id: str, context: Optional[Dict] = None) -> Dict[str, Dict[str, str]]:
#         """Get dialogue state for a specific turn"""
#         if turn_id in self.ground_truth_dialogue_states:
#             return self.ground_truth_dialogue_states[turn_id]
        
#         # Return empty state for unknown turns
#         return {domain: {} for domain in self.domains}
    
#     def get_dialogue_acts(self, utterance: str, context: Optional[Dict] = None) -> List[Dict[str, Any]]:
#         """Get dialogue acts for utterance"""
#         utterance_lower = utterance.lower().strip()
        
#         if utterance_lower in self.ground_truth_dialogue_acts:
#             return self.ground_truth_dialogue_acts[utterance_lower]
        
#         # Generate dialogue acts from intent and entities
#         intent = self.get_intent(utterance)
#         entities = self.get_entities(utterance)
        
#         acts = []
#         domain, intent_type = intent.split('-') if '-' in intent else ('general', intent)
        
#         for entity_value, slot_type in entities:
#             acts.append({
#                 'intent': intent,
#                 'domain': domain.lower(),
#                 'slot': slot_type,
#                 'value': entity_value
#             })
        
#         return acts
    
#     def get_constraints(self, utterance: str, context: Optional[Dict] = None) -> List[Tuple[str, str]]:
#         """Extract constraints from utterance"""
#         constraints = []
        
#         # Negation constraints
#         if any(neg in utterance.lower() for neg in ['no', 'not', 'avoid', 'without']):
#             # Simple extraction of what to avoid
#             words = utterance.lower().split()
#             for i, word in enumerate(words):
#                 if word in ['no', 'not', 'avoid', 'without'] and i + 1 < len(words):
#                     constraints.append(('avoid', words[i + 1]))
        
#         # Preference constraints
#         if any(pref in utterance.lower() for pref in ['prefer', 'want', 'like']):
#             entities = self.get_entities(utterance)
#             for entity_value, slot_type in entities:
#                 constraints.append(('prefer', f"{slot_type}:{entity_value}"))
        
#         return constraints
    
#     def validate_response(self, response: str, expected_intent: str, 
#                          expected_entities: List[Tuple[str, str]]) -> Dict[str, bool]:
#         """Validate system response against ground truth"""
#         response_lower = response.lower()
        
#         # Check if response addresses the intent
#         intent_addressed = self._response_addresses_intent(response_lower, expected_intent)
        
#         # Check if response contains expected entities
#         entities_mentioned = 0
#         for entity_value, slot_type in expected_entities:
#             if entity_value.lower() in response_lower:
#                 entities_mentioned += 1
        
#         entity_coverage = entities_mentioned / len(expected_entities) if expected_entities else 1.0
        
#         return {
#             'intent_addressed': intent_addressed,
#             'entity_coverage_complete': entity_coverage == 1.0,
#             'entity_coverage_rate': entity_coverage, #type: ignore
#             'response_appropriate': intent_addressed and entity_coverage > 0.5
#         }
    
#     def _response_addresses_intent(self, response: str, intent: str) -> bool:
#         """Check if response appropriately addresses the given intent"""
#         intent_type = intent.split('-')[-1] if '-' in intent else intent
        
#         intent_indicators = {
#             'Inform': ['here is', 'found', 'available', 'located'],
#             'Request': ['what', 'which', 'please provide', 'need to know'],
#             'Recommend': ['recommend', 'suggest', 'try'],
#             'Book': ['booked', 'reserved', 'confirmed'],
#             'NoOffer': ['no results', 'nothing found', 'unavailable'],
#             'greet': ['hello', 'hi', 'help'],
#             'thank': ['welcome', 'pleasure', 'problem'],
#             'bye': ['goodbye', 'bye', 'thank you']
#         }
        
#         indicators = intent_indicators.get(intent_type, [])
#         return any(indicator in response for indicator in indicators)
    
#     def get_evaluation_metrics(self) -> Dict[str, str]:
#         """Get available evaluation metrics and their descriptions"""
#         return {
#             'joint_goal_accuracy': 'Percentage of turns with perfect dialogue state prediction',
#             'slot_accuracy': 'Average accuracy across all slots',
#             'intent_accuracy': 'Percentage of correctly predicted intents',
#             'entity_f1': 'F1 score for entity extraction',
#             'inform_rate': 'Percentage of dialogues where system provides correct information',
#             'success_rate': 'Percentage of dialogues where user goal is achieved',
#             'bleu_score': 'BLEU score for response fluency',
#             'response_appropriateness': 'Percentage of contextually appropriate responses'
#         }
    
class ImprovedReferenceResolver:
    """
    Evaluates reference resolution accuracy for multi-domain task-oriented dialogues.
    Handles domain-specific entities and cross-domain references.
    """
    
    def __init__(self):
        # Domain-specific entity types and pronouns
        self.pronouns = {'it', 'he', 'she', 'they', 'this', 'that', 'them', 'one', 'ones'}
    
        # Domain-specific reference expressions (expanded and lowercased for robust matching)
        self.domain_references = {
            'restaurant': {
            'the restaurant', 'this restaurant', 'that restaurant', 'this place', 'that place',
            'the venue', 'the spot', 'my restaurant', 'our restaurant', 'the eatery'
            },
            'hotel': {
            'the hotel', 'this hotel', 'that hotel', 'this accommodation', 'that accommodation',
            'the property', 'my hotel', 'our hotel', 'the lodging', 'the inn'
            },
            'attraction': {
            'the attraction', 'this attraction', 'that attraction', 'this place', 'that venue',
            'the site', 'the location', 'the spot', 'my attraction', 'our attraction'
            },
            'train': {
            'the train', 'this train', 'that train', 'this service', 'that service',
            'the journey', 'the departure', 'my train', 'our train', 'the railway'
            },
            'taxi': {
            'the taxi', 'this taxi', 'that taxi', 'this ride', 'that ride',
            'the car', 'the vehicle', 'my taxi', 'our taxi', 'the cab'
            },
            'booking': {
            'the booking', 'this booking', 'that booking', 'this reservation', 'that reservation',
            'my booking', 'our booking', 'the appointment', 'the reservation'
            },
            'general': {
            'it', 'this', 'that', 'the option', 'the choice', 'the one', 'the thing', 'the item'
            }
        }

        # Common referential expressions across domains (expanded, lowercased, and normalized)
        self.common_references = {
            'the first one', 'the second one', 'the last one', 'another one', 'the next one',
            'the previous one', 'the same one', 'a different one', 'the other option',
            'the alternative', 'the cheapest', 'the most expensive', 'the closest', 'the nearest',
            'the best one', 'the worst one', 'the recommended one', 'the available one'
        }

        # Slot-based references (expanded, lowercased, and normalized)
        self.slot_references = {
            'price': {
            'the price', 'the cost', 'the fee', 'how much', 'the fare', 'the charge', 'the rate', 'the amount'
            },
            'area': {
            'the area', 'the location', 'the place', 'where', 'which area', 'which location', 'which part'
            },
            'food': {
            'the cuisine', 'the food type', 'what kind', 'the dish', 'the menu', 'the meal', 'the food'
            },
            'stars': {
            'the rating', 'the stars', 'how good', 'the review', 'the score', 'the quality'
            },
            'departure': {
            'departure time', 'when it leaves', 'leaving time', 'when does it depart', 'the departure'
            },
            'destination': {
            'where to', 'the destination', 'going to', 'where is it going', 'the arrival place'
            },
            'phone': {
            'the number', 'phone number', 'contact', 'contact number', 'telephone', 'how to call'
            },
            'address': {
            'the address', 'where it is', 'location', 'the street', 'the postcode', 'the zip code', 'the place address'
            }
        }
    
    def extract_references(self, text: str, context_domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """Extract references with context information"""
        text_lower = text.lower()
        references = []
        
        # Extract pronouns
        words = text_lower.split()
        for i, word in enumerate(words):
            if word in self.pronouns:
                references.append({
                    'type': 'pronoun',
                    'text': word,
                    'position': i,
                    'domain': context_domain
                })
        
        # Extract domain-specific references
        for domain, refs in self.domain_references.items():
            for ref in refs:
                if ref in text_lower:
                    references.append({
                        'type': 'domain_reference',
                        'text': ref,
                        'domain': domain,
                        'matches_context': domain == context_domain
                    })
        
        # Extract common references
        for ref in self.common_references:
            if ref in text_lower:
                references.append({
                    'type': 'common_reference',
                    'text': ref,
                    'domain': context_domain
                })
        
        # Extract slot-based references
        for slot, refs in self.slot_references.items():
            for ref in refs:
                if ref in text_lower:
                    references.append({
                        'type': 'slot_reference',
                        'text': ref,
                        'slot': slot,
                        'domain': context_domain
                    })
        
        return references
    
    def evaluate_resolution(self, dialogue_turns: List[Dict], responses: List[str]) -> Dict[str, float]:
        """
        Evaluate reference resolution with detailed metrics
        
        Args:
            dialogue_turns: List of dialogue turns with user input and system response
            responses: List of system responses to evaluate
            
        Returns:
            Dictionary with resolution metrics
        """
        total_refs = 0
        resolved_refs = 0
        domain_specific_resolved = 0
        domain_specific_total = 0
        cross_domain_resolved = 0
        cross_domain_total = 0
        
        for i, (turn, response) in enumerate(zip(dialogue_turns, responses)):
            user_input = turn.get('user_input', '')
            current_domains = turn.get('domains', ['general'])
            
            # Get dialogue history for context
            context_turns = dialogue_turns[:i]
            for current_domain in current_domains:
                refs = self.extract_references(user_input, current_domain)
                
                if not refs:
                    continue
                    
                total_refs += len(refs)
                response_lower = response.lower()
                
                for ref in refs:
                    is_resolved = self._check_resolution(ref, response_lower, context_turns)
                    
                    if is_resolved:
                        resolved_refs += 1
                        
                        # Track domain-specific resolution
                        if ref.get('domain') == current_domain:
                            domain_specific_resolved += 1
                        elif ref.get('domain') and ref['domain'] != current_domain:
                            cross_domain_resolved += 1
                    
                    # Count domain-specific and cross-domain references
                    if ref.get('domain') == current_domain:
                        domain_specific_total += 1
                    elif ref.get('domain') and ref['domain'] != current_domain:
                        cross_domain_total += 1
        
        # Calculate metrics
        overall_resolution = resolved_refs / total_refs if total_refs > 0 else 1.0
        domain_resolution = domain_specific_resolved / domain_specific_total if domain_specific_total > 0 else 1.0
        cross_domain_resolution = cross_domain_resolved / cross_domain_total if cross_domain_total > 0 else 1.0
        
        return {
            'overall_resolution_rate': overall_resolution,
            'domain_specific_resolution_rate': domain_resolution,
            'cross_domain_resolution_rate': cross_domain_resolution,
            'total_references': total_refs,
            'resolved_references': resolved_refs
        }
    
    def _check_resolution(self, reference: Dict, response: str, context_turns: List[Dict]) -> bool:
        """Check if a reference is properly resolved in the response"""
        ref_type = reference['type']
        ref_text = reference['text']
        
        # Basic heuristics for resolution checking
        if ref_type == 'pronoun':
            # For pronouns, check if response contains specific entities or information
            return self._contains_specific_info(response, reference.get('domain'))
        
        elif ref_type == 'domain_reference':
            # For domain references, check if response contains domain-specific information
            domain = reference['domain']
            return self._contains_domain_info(response, domain)
        
        elif ref_type == 'slot_reference':
            # For slot references, check if response contains the requested slot information
            slot = reference['slot']
            return self._contains_slot_info(response, slot)
        
        elif ref_type == 'common_reference':
            # For common references, check if response contains comparative or specific information
            return self._contains_specific_info(response, reference.get('domain'))
        
        return False
    
    def _contains_specific_info(self, response: str, domain: Optional[str]) -> bool:
        """Check if response contains specific, non-generic information"""
        # Look for specific entities, numbers, names, etc.
        specific_patterns = [
            r'\b\d+\b',  # Numbers
            r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)+\b',  # Proper names (multi-word)
            r'(?:£|\$|€|₹|¥|₩|₽|₺|₪|₫|₴|₦|₲|₵|₡|₱|฿|₨|₮|₭|₠|₢|₣|₤|₧|₯|₰|₳|₴|₵|₸|₺|₼|₽|₾|₿)\s?\d+(?:\.\d{1,2})?',  # Prices in various currencies
            r'\b\d{2}:\d{2}\b',  # Times (24-hour)
            r'\b\d{1,2}:\d{2}\s?(?:am|pm|a\.m\.|p\.m\.)\b',  # Times with am/pm
            r'\b[A-Z]{2}\d+\s?\d[A-Z]{2}\b',  # Postcodes (UK-style)
        ]
        
        for pattern in specific_patterns:
            if re.search(pattern, response):
                return True
                
        # Check for domain-specific keywords
        if domain:
            return self._contains_domain_info(response, domain)
            
        return len(response.split()) > 5  # Basic length heuristic
    
    def _contains_domain_info(self, response: str, domain: str) -> bool:
        """Check if response contains domain-specific information"""
        domain_keywords = {
            'restaurant': ['restaurant', 'food', 'cuisine', 'menu', 'table', 'booking', 'meal'],
            'hotel': ['hotel', 'room', 'accommodation', 'stay', 'night', 'guest', 'check'],
            'attraction': ['attraction', 'museum', 'park', 'gallery', 'entertainment', 'visit'],
            'train': ['train', 'departure', 'arrival', 'platform', 'ticket', 'journey'],
            'taxi': ['taxi', 'car', 'pick', 'drop', 'driver', 'ride'],
            'booking': ['book', 'reserve', 'appointment', 'confirmation', 'reference'],
            'general': ['no','never','hello', 'hi', 'thanks', 'thank you', 'bye', 'goodbye', 'help', 'please', 'welcome']
        }
        
        keywords = domain_keywords.get(domain, [])
        return any(keyword in response.lower() for keyword in keywords)
    
    def _contains_slot_info(self, response: str, slot: str) -> bool:
        """Check if response contains information about the requested slot"""
        # More robust slot indicators: include synonyms, plural forms, and common paraphrases
        slot_indicators = {
            'price': [
            '£', 'pound', 'pounds', 'dollar', 'dollars', 'rupee', 'rupees', 'cost', 'price', 'expensive', 'cheap', 'free', 'how much', 'fee', 'fare', 'charge', 'rate', 'amount', 'total cost', 'per night', 'per person'
            ],
            'area': [
            'north', 'northern', 'south', 'southern', 'east', 'eastern', 'west', 'western', 'centre', 'center', 'area', 'region', 'part of town', 'district', 'location', 'neighborhood', 'vicinity'
            ],
            'food': [
            'food', 'cuisine', 'chinese', 'indian', 'italian', 'british', 'french', 'dish', 'menu', 'type of food', 'kind of food', 'meal', 'serves', 'specialty', 'what food', 'what cuisine'
            ],
            'stars': [
            'star', 'stars', 'rating', 'rated', 'quality', 'how many stars', 'review', 'score', 'level', 'class', 'rank', 'grade'
            ],
            'departure': [
            'departure', 'depart', 'leave', 'leaving', 'from', 'start', 'starting point', 'origin', 'where from', 'departing', 'pick up', 'pickup'
            ],
            'destination': [
            'destination', 'arrive', 'arriving', 'to', 'where to', 'going to', 'arrival', 'end point', 'drop off', 'dropoff', 'stop', 'final stop'
            ],
            'phone': [
            'phone', 'number', 'call', 'contact', 'telephone', 'mobile', 'cell', 'how to call', 'contact number', 'phone number', 'reach at'
            ],
            'address': [
            'address', 'located', 'location', 'street', 'road', 'avenue', 'postcode', 'zip code', 'where is', 'place address', 'building', 'house number', 'where can I find', 'how to get to'
            ]
        }
        
        indicators = slot_indicators.get(slot, [])
        return any(indicator in response.lower() for indicator in indicators)
    
class ImprovedConstraintChecker:
    """
    Evaluates constraint adherence across multiple domains in task-oriented dialogues.
    Handles domain-specific constraints and preferences.
    """
    
    def __init__(self):
        # Multi-domain constraint patterns
        self.constraint_patterns = {
            'avoid': ['avoid', 'without', 'no', "don't", 'not', 'exclude'],
            'require': ['must', 'need', 'require', 'should', 'want', 'prefer'],
            'preference': ['prefer', 'like', 'better', 'rather', 'instead'],
            'restriction': ['only', 'just', 'exclusively', 'solely'],
            'comparison': ['cheaper', 'expensive', 'better', 'worse', 'closer', 'further']
        }
        
        # Domain-specific constraint types
        self.domain_constraints = {
            'restaurant': {
            'food_type': ['chinese', 'indian', 'italian', 'british', 'french'],
            'price_range': ['cheap', 'moderate', 'expensive'],
            'area': ['north', 'south', 'east', 'west', 'centre'],
            'special': ['vegetarian', 'vegan', 'halal', 'gluten-free']
            },
            'hotel': {
            'price_range': ['cheap', 'moderate', 'expensive'],
            'area': ['north', 'south', 'east', 'west', 'centre'],
            'stars': ['1', '2', '3', '4', '5'],
            'amenities': ['parking', 'wifi', 'breakfast', 'pool', 'gym'],
            'type': ['guesthouse', 'hotel']
            },
            'attraction': {
            'type': ['museum', 'park', 'theatre', 'cinema', 'gallery', 'church'],
            'area': ['north', 'south', 'east', 'west', 'centre'],
            'fee': ['free', 'paid']
            },
            'train': {
            'day': ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'],
            'time': ['morning', 'afternoon', 'evening'],
            'departure': ['cambridge', 'london', 'birmingham', 'norwich'],
            'destination': ['cambridge', 'london', 'birmingham', 'norwich']
            },
            'taxi': {
            'departure': ['location_name'],
            'destination': ['location_name'],
            'time': ['departure_time']
            },
            'general': {
            'politeness': ['please', 'thank you', 'thanks'],
            'urgency': ['asap', 'urgent', 'immediately', 'soon'],
            'preference': ['prefer', 'like', 'would rather'],
            'negation': ['no', 'not', "don't", 'without', 'avoid'],
            'other': ['anything', 'something', 'nothing', 'everything']
            }
        }
        
        # Negation patterns for constraint detection
        # These patterns will match words anywhere in the sentence, not just at the start.
        # Negation patterns: match negation words followed by any word(s) anywhere in the sentence
        self.negation_patterns = [
            r"\bno\s+(\w+)",
            r"\bnot\s+(\w+)",
            r"\bdon't\s+want\s+(\w+)",
            r"\bavoid\s+(\w+)",
            r"\bwithout\s+(\w+)",
            r"\bexcept\s+(\w+)",
            r"\bno\b.*?\b(\w+)",
            r"\bnot\b.*?\b(\w+)",
            r"\bwithout\b.*?\b(\w+)",
            r"\bavoid\b.*?\b(\w+)",
            r"\bdon't\b.*?\b(\w+)",
            r"\bexcept\b.*?\b(\w+)"
        ]

        # Preference patterns: match preference words followed by any word(s) anywhere in the sentence
        self.preference_patterns = [
            r"\bprefer\s+(\w+)",
            r"\bwould\s+like\s+(\w+)",
            r"\bwant\s+(\w+)",
            r"\bneed\s+(\w+)",
            r"\blooking\s+for\s+(\w+)",
            r"\bprefer\b.*?\b(\w+)",
            r"\bwould\b.*?\blike\b.*?\b(\w+)",
            r"\bwant\b.*?\b(\w+)",
            r"\bneed\b.*?\b(\w+)",
            r"\blooking\b.*?\bfor\b.*?\b(\w+)"
        ]
    
    def extract_constraints_from_turns(self, turns: List[Dict]) -> Dict[str, Dict[str, Set[str]]]:
        """
        Extract constraints across all dialogue turns, organized by domain and type
        
        Returns:
            Dict with structure: {domain: {constraint_type: {values}}}
        """
        constraints = defaultdict(lambda: defaultdict(set))
        
        for turn in turns:
            user_input = turn.get('user_input', '').lower()
            domains = turn.get('domains', ['general'])
            # intent = turn.get('intent', '')
            intent = turn.get('true_intent', 'unknown')
            for domain in domains:
                # Extract domain-specific constraints
                domain_constraints = self._extract_domain_constraints(user_input, domain)
                for constraint_type, values in domain_constraints.items():
                    constraints[domain][constraint_type].update(values)
                
                # Extract general constraint patterns
                general_constraints = self._extract_general_constraints(user_input)
                for constraint_type, values in general_constraints.items():
                    constraints[domain][constraint_type].update(values)
                
                # Extract constraints from intent patterns
                intent_constraints = self._extract_intent_constraints(user_input, intent)
                for constraint_type, values in intent_constraints.items():
                    constraints[domain][constraint_type].update(values)
        
        return dict(constraints)
    
    def _extract_domain_constraints(self, text: str, domain: str) -> Dict[str, Set[str]]:
        """Extract domain-specific constraints"""
        constraints = defaultdict(set)
        
        if domain not in self.domain_constraints:
            return constraints
        
        domain_spec = self.domain_constraints[domain]
        
        for constraint_type, values in domain_spec.items():
            for value in values:
                if value in text:
                    # Check if it's a negation or preference
                    if any(neg in text for neg in ['no', 'not','except', 'avoid', 'without']):
                        constraints['avoid'].add(f"{constraint_type}:{value}")
                    elif any(pref in text for pref in ['need','looking','would','prefer', 'want', 'like']):
                        constraints['prefer'].add(f"{constraint_type}:{value}")
                    else:
                        constraints['require'].add(f"{constraint_type}:{value}")
        
        return constraints
    
    def _extract_general_constraints(self, text: str) -> Dict[str, Set[str]]:
        """Extract general constraint patterns using regex"""
        constraints = defaultdict(set)
        
        # Extract negation constraints
        for pattern in self.negation_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                constraints['avoid'].add(match)
        
        # Extract preference constraints
        for pattern in self.preference_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                constraints['prefer'].add(match)
        
        return constraints
    
    def _extract_intent_constraints(self, text: str, intent: str) -> Dict[str, Set[str]]:
        """Extract constraints based on intent patterns"""
        constraints = defaultdict(set)
        
        # Intent-specific constraint extraction
        if 'no_offer' in intent.lower():
            constraints['system_limitation'].add('no_options_available')
        elif 'select' in intent.lower():
            constraints['choice_required'].add('multiple_options')
        elif 'request' in intent.lower():
            constraints['information_needed'].add('missing_slot')
        
        return constraints
    
    def check_adherence(self, constraints: Dict[str, Dict[str, Set[str]]], 
                       response: str, domains: str) -> Dict[str, float]:
        """
        Check constraint adherence with detailed metrics
        
        Args:
            constraints: Extracted constraints by domain and type
            response: System response to check
            domain: Current dialogue domain
            
        Returns:
            Dictionary with adherence metrics
        """
        # Initialize metrics
        total_constraints = 0
        violated_constraints = 0
        satisfied_constraints = 0

        response_lower = response.lower()
        for domain in domains:
            domain_constraints = constraints.get(domain, {})
            
            adherence_by_type = {}
            
            for constraint_type, constraint_set in domain_constraints.items():
                if not constraint_set:
                    continue
                    
                type_total = len(constraint_set)
                type_violations = 0
                type_satisfied = 0
                
                total_constraints += type_total
                
                for constraint in constraint_set:
                    violation_found, satisfaction_found = self._check_constraint(
                        constraint, constraint_type, response_lower, domain
                    )
                    
                    if violation_found:
                        violated_constraints += 1
                        type_violations += 1
                    elif satisfaction_found:
                        satisfied_constraints += 1
                        type_satisfied += 1
                
                # Calculate adherence rate for this constraint type
                if type_total > 0:
                    type_adherence = (type_total - type_violations) / type_total
                    adherence_by_type[constraint_type] = type_adherence
        
        # Calculate overall metrics
        overall_adherence = (total_constraints - violated_constraints) / total_constraints if total_constraints > 0 else 1.0
        satisfaction_rate = satisfied_constraints / total_constraints if total_constraints > 0 else 0.0
        
        return {
            'overall_adherence_rate': max(0.0, overall_adherence),
            'satisfaction_rate': satisfaction_rate,
            'violation_count': violated_constraints,
            'total_constraints': total_constraints,
            'adherence_by_type': adherence_by_type #type: ignore
        }
    
    def _check_constraint(self, constraint: str, constraint_type: str, 
                         response: str, domain: str) -> Tuple[bool, bool]:
        """
        Check if a specific constraint is violated or satisfied
        
        Returns:
            (violation_found, satisfaction_found)
        """
        violation_found = False
        satisfaction_found = False
        
        if constraint_type == 'avoid':
            # Check if avoided item appears in response
            if ':' in constraint:
                _, value = constraint.split(':', 1)
                violation_found = value in response
            else:
                violation_found = constraint in response
        
        elif constraint_type == 'prefer':
            # Check if preferred item appears in response
            if ':' in constraint:
                _, value = constraint.split(':', 1)
                satisfaction_found = value in response
            else:
                satisfaction_found = constraint in response
        
        elif constraint_type == 'require':
            # Check if required item appears in response
            if ':' in constraint:
                _, value = constraint.split(':', 1)
                satisfaction_found = value in response
            else:
                satisfaction_found = constraint in response
        
        elif constraint_type == 'system_limitation':
            # Check if system properly handles limitations
            if constraint == 'no_options_available':
                satisfaction_found = any(phrase in response for phrase in [
                    'no results', 'not found', 'no matches', 'unavailable'
                ])
        
        elif constraint_type == 'choice_required':
            # Check if system provides options for selection
            if constraint == 'multiple_options':
                satisfaction_found = any(phrase in response for phrase in [
                    'which one', 'would you prefer', 'choose', 'select'
                ])
        
        return violation_found, satisfaction_found

class ImprovedTaskSuccessEvaluator:
    """
    Evaluates task completion success for multi-domain task-oriented dialogues.
    Maps to MultiWOZ intent structure and handles domain-specific success criteria.
    """
    
    def __init__(self):
        # Intent-specific success patterns based on MultiWOZ structure
        self.intent_success_patterns = {
            # Information providing intents
            'inform': {
                'success_indicators': ['here is', 'found', 'available', 'located at', 'costs'],
                'required_elements': ['specific_info'],
                'domain_specific': True
            },
            
            # Request handling intents  
            'request': {
                'success_indicators': ['the address is', 'phone number is', 'price is', 'departure is'],
                'required_elements': ['requested_slot_value'],
                'domain_specific': True
            },
            
            # Recommendation intents
            'recommend': {
                'success_indicators': ['i recommend', 'suggest', 'try', 'good option'],
                'required_elements': ['entity_name', 'reason'],
                'domain_specific': True
            },
            
            # Selection intents
            'select': {
                'success_indicators': ['which one', 'would you prefer', 'choose between'],
                'required_elements': ['multiple_options'],
                'domain_specific': True
            },
            
            # No offer intents
            'no_offer': {
                'success_indicators': ['no results', 'nothing matches', 'not found', 'unavailable'],
                'required_elements': ['explanation'],
                'domain_specific': False
            },
            
            # Booking intents
            'book': {
                'success_indicators': ['booked', 'reserved', 'confirmed', 'reference number'],
                'required_elements': ['confirmation', 'reference'],
                'domain_specific': True
            },
            
            'no_book': {
                'success_indicators': ['cannot book', 'booking failed', 'not available'],
                'required_elements': ['explanation'],
                'domain_specific': False
            },
            
            'offer_book': {
                'success_indicators': ['shall i book', 'would you like to book', 'book for you'],
                'required_elements': ['booking_offer'],
                'domain_specific': True
            },
            
            'offer_booked': {
                'success_indicators': ['successfully booked', 'booking confirmed', 'reference number'],
                'required_elements': ['confirmation', 'reference'],
                'domain_specific': True
            },
            
            # General intents
            'greet': {
                'success_indicators': ['hello', 'hi', 'welcome', 'good'],
                'required_elements': ['greeting'],
                'domain_specific': False
            },
            
            'bye': {
                'success_indicators': ['goodbye', 'bye', 'thank you for', 'have a nice'],
                'required_elements': ['farewell'],
                'domain_specific': False
            },
            
            'thank': {
                'success_indicators': ['you are welcome', 'my pleasure', 'glad to help'],
                'required_elements': ['acknowledgment'],
                'domain_specific': False
            },
            
            'request_more': {
                'success_indicators': ['anything else', 'what else', 'further assistance'],
                'required_elements': ['continuation_offer'],
                'domain_specific': False
            },
            
            'welcome': {
                'success_indicators': ['you are welcome', 'no problem', 'glad to help'],
                'required_elements': ['acknowledgment'],
                'domain_specific': False
            },
            'unknown' :{
                'success_indicators': [],
                'required_elements': [],
                'domain_specific': False
            }
        }
        
        # Domain-specific success criteria
        self.domain_success_criteria = {
            'restaurant': {
            'required_slots': ['name', 'food', 'area'],
            'optional_slots': ['phone', 'address', 'pricerange', 'stars', 'type', 'openhours'],
            'booking_slots': ['time', 'day', 'people', 'reference', 'confirmation']
            },
            'hotel': {
            'required_slots': ['name', 'area'],
            'optional_slots': ['phone', 'address', 'pricerange', 'stars', 'type', 'internet', 'parking'],
            'booking_slots': ['day', 'stay', 'people', 'reference', 'confirmation']
            },
            'attraction': {
            'required_slots': ['name', 'type', 'area'],
            'optional_slots': ['phone', 'address', 'fee', 'openhours', 'postcode'],
            'booking_slots': []
            },
            'train': {
            'required_slots': ['departure', 'destination', 'day', 'leaveat'],
            'optional_slots': ['duration', 'price', 'trainid', 'arriveby', 'people'],
            'booking_slots': ['people', 'reference', 'confirmation']
            },
            'taxi': {
            'required_slots': ['departure', 'destination'],
            'optional_slots': ['leaveat', 'arriveby', 'phone', 'type', 'car', 'price'],
            'booking_slots': ['reference', 'confirmation']
            },
            'hospital': {
            'required_slots': ['department'],
            'optional_slots': ['phone', 'address', 'postcode'],
            'booking_slots': []
            },
            'police': {
            'required_slots': ['name'],
            'optional_slots': ['phone', 'address', 'postcode'],
            'booking_slots': []
            },
            'general': {
            'required_slots': [],
            'optional_slots': ['no','never','hello', 'hi', 'thanks', 'thank you', 'bye', 'goodbye', 'help', 'please', 'welcome','greeting', 'acknowledgment', 'continuation_offer'],
            'booking_slots': []
            }
        }
        
        # Evaluation metrics
        self.inform_criteria = {
            'has_entity_info': 'Contains specific entity information',
            'has_slot_values': 'Provides requested slot values',
            'matches_constraints': 'Matches user constraints'
        }
        
        self.success_criteria = {
            'task_completed': 'User goal achieved',
            'all_info_provided': 'All requested information given',
            'booking_completed': 'Booking successfully made (if requested)'
        }
    
    def evaluate_task_success(self, dialogue_turns: List[Dict], responses: List[str]) -> Dict[str, Any]:
        """
        Evaluate task success across the entire dialogue
        
        Args:
            dialogue_turns: List of dialogue turns with context
            responses: System responses to evaluate
            
        Returns:
            Comprehensive success metrics
        """
        if len(dialogue_turns) != len(responses):
            raise ValueError("Number of turns and responses must match")
        
        # Initialize metrics
        total_turns = len(dialogue_turns)
        successful_turns = 0
        inform_successes = 0
        booking_successes = 0
        
        # Track domain-specific metrics
        domain_metrics = defaultdict(lambda: {'turns': 0, 'successes': 0})
        intent_metrics = defaultdict(lambda: {'turns': 0, 'successes': 0})
        
        # User goal tracking
        user_goals = self._extract_user_goals(dialogue_turns)
        goal_completion = self._evaluate_goal_completion(dialogue_turns, responses, user_goals)
        
        # Evaluate each turn
        for i, (turn, response) in enumerate(zip(dialogue_turns, responses)):
            intent = turn.get('true_intent', 'unknown')
            domains = turn.get('domains', ['general'])
            
            # Evaluate turn-level success
            turn_success = self._evaluate_turn_success(turn, response, intent, domains)
            
            if turn_success['success']:
                successful_turns += 1
            
            if turn_success['inform_success']:
                inform_successes += 1
                
            if turn_success['booking_success']:
                booking_successes += 1
            
            # Update domain and intent metrics
            for domain in domains:
                domain_metrics[domain]['turns'] += 1
            intent_metrics[intent]['turns'] += 1
            
            if turn_success['success']:
                for domain in domains:
                    domain_metrics[domain]['successes'] += 1
                intent_metrics[intent]['successes'] += 1
        
        # Calculate rates
        overall_success_rate = successful_turns / total_turns if total_turns > 0 else 0.0
        inform_rate = inform_successes / total_turns if total_turns > 0 else 0.0
        
        # Calculate domain-specific success rates
        domain_success_rates = {}
        for domain, metrics in domain_metrics.items():
            domain_success_rates[domain] = metrics['successes'] / metrics['turns'] if metrics['turns'] > 0 else 0.0
        
        # Calculate intent-specific success rates
        intent_success_rates = {}
        for intent, metrics in intent_metrics.items():
            intent_success_rates[intent] = metrics['successes'] / metrics['turns'] if metrics['turns'] > 0 else 0.0
        
        return {
            'overall_success_rate': overall_success_rate,
            'inform_rate': inform_rate,
            'goal_completion_rate': goal_completion['completion_rate'],
            'domain_success_rates': domain_success_rates,
            'intent_success_rates': intent_success_rates,
            'total_turns': total_turns,
            'successful_turns': successful_turns,
            'goal_analysis': goal_completion,
            'booking_success_rate': booking_successes / total_turns if total_turns > 0 else 0.0
        }
    
    def _evaluate_turn_success(self, turn: Dict, response: str, intent: str, domains: str) -> Dict[str, bool]:
        """Evaluate success for a single turn"""
        response_lower = response.lower()
        
        # Parse intent type (remove domain prefix)
        intent_type = intent.split('-')[-1] if '-' in intent else intent
        
        # Get success patterns for this intent type
        patterns = self.intent_success_patterns.get(intent_type, {})
        
        success_indicators = patterns.get('success_indicators', [])
        required_elements = patterns.get('required_elements', [])
        
        # Check for success indicators
        has_success_indicators = any(indicator in response_lower for indicator in success_indicators)
        
        # Check for required elements
        has_required_elements = False
        for domain in domains:
            has_required_elements = has_required_elements or self._check_required_elements(response, required_elements, domain)
        
        # Domain-specific checks
        domain_success = False
        if patterns.get('domain_specific', False):
            for domain in domains:
                domain_success = domain_success or self._check_domain_specific_success(response, domain, intent_type)
        
        # Determine different types of success
        turn_success = has_success_indicators or has_required_elements or domain_success
        inform_success = intent_type in ['inform', 'recommend']
        booking_success = intent_type in ['book', 'offer_booked','offer_book']
        
        return {
            'success': turn_success,
            'inform_success': inform_success,
            'booking_success': booking_success,
            'has_indicators': has_success_indicators,
            'has_required_elements': has_required_elements,
            'domain_success': domain_success
        }
    
    def _check_required_elements(self, response: str, required_elements: List[str], domain: str) -> bool:
        """Check if response contains required elements"""
        response = response.lower()
        
        for element in required_elements:
            if element == 'specific_info':
                if not self._has_specific_info(response):
                    return False
            elif element == 'requested_slot_value':
                if not self._has_slot_value(response):
                    return False
            elif element == 'entity_name':
                if not self._has_entity_name(response, domain):
                    return False
            elif element == 'multiple_options':
                if not self._has_multiple_options(response):
                    return False
            elif element == 'confirmation':
                if not self._has_confirmation(response):
                    return False
            elif element == 'reference':
                if not self._has_reference_number(response):
                    return False
            elif element == 'explanation':
                if len(response.split()) < 5:  # Basic length check for explanation
                    return False
        
        return True
    
    def _has_specific_info(self, response: str) -> bool:
        """Check if response contains specific information"""
        # Look for specific patterns: numbers, times, addresses, names
        specific_patterns = [
            r'\b\d+\b',  # Numbers
            r'\b\d{2}:\d{2}\b',  # Times
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # Proper names
            r'£\d+',  # Prices
            r'\b\d+ [A-Z][a-z]+ (Street|Road|Lane|Avenue)\b'  # Addresses
        ]
        
        return any(re.search(pattern, response) for pattern in specific_patterns)
    
    def _has_slot_value(self, response: str) -> bool:
        """Check if response provides slot values"""
        # Look for common slot value patterns
        # Robust slot value patterns: match anywhere in the sentence, allow flexible phrasing and optional words
        slot_patterns = [
            r'\b(is|are)\s+\w+',  # e.g., "is chinese", "are expensive"
            r'\b(costs?|price|priced at|rate|fare|fee)\s*[:\-]?\s*£?\d+',  # e.g., "costs £20", "price: 15"
            r'\b(located|address|at|on|in)\s+([A-Za-z0-9 ,\-]+)',  # e.g., "located at 123 Main St"
            r'\b(phone|number|contact|telephone)\s*[:\-]?\s*\d+',  # e.g., "phone number is 01234"
            r'\b(open|close|opens|closes|open hours|opening hours)\s*(at|from|until)?\s*\d{1,2}(:\d{2})?\s*(am|pm)?',  # e.g., "opens at 9am"
            r'\bfor\s+\d+\s+(people|persons|guests|adults|kids|children)',  # e.g., "for 2 people"
            r'\b([A-Za-z]+)\s+(cuisine|food|type)',  # e.g., "chinese cuisine"
            r'\b(star|stars|rated)\s*\d',  # e.g., "4 star"
            r'\b(area|region|part of town|district|location)\s*[:\-]?\s*[a-z]+',  # e.g., "area: north"
            r'\b(day|date)\s*[:\-]?\s*[A-Za-z]+',  # e.g., "day: Monday"
            r'\b(time|at|leave at|arrive by)\s*[:\-]?\s*\d{1,2}(:\d{2})?\s*(am|pm)?',  # e.g., "leave at 10:30am"
        ]
        
        return any(re.search(pattern, response.lower()) for pattern in slot_patterns)
    
    def _has_entity_name(self, response: str, domain: str) -> bool:
        """Check if response contains entity names"""
        # Look for capitalized entity names (simplified heuristic)
        entity_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        matches = re.findall(entity_pattern, response)
        
        # Filter out common non-entity words
        non_entities = {'I', 'The', 'A', 'An', 'You', 'There', 'Here', 'What', 'How', 'Where'}
        entity_matches = [match for match in matches if match not in non_entities]
        
        return len(entity_matches) > 0
    
    def _has_multiple_options(self, response: str) -> bool:
        """Check if response presents multiple options"""
        option_indicators = ['which one', 'choose', 'select', 'option', 'either', 'or', 'between']
        return any(indicator in response.lower() for indicator in option_indicators)
    
    def _has_confirmation(self, response: str) -> bool:
        """Check if response contains booking confirmation"""
        confirmation_indicators = ['booked', 'reserved', 'confirmed', 'successful']
        return any(indicator in response.lower() for indicator in confirmation_indicators)
    
    def _has_reference_number(self, response: str) -> bool:
        """Check if response contains reference number"""
        # Look for reference patterns
        ref_patterns = [
            r'reference (\w+)',
            r'booking (\w+)',
            r'confirmation (\w+)',
            r'ref (\w+)'
        ]
        return any(re.search(pattern, response.lower()) for pattern in ref_patterns)
    
    def _check_domain_specific_success(self, response: str, domain: str, intent_type: str) -> bool:
        """Check domain-specific success criteria"""
        if domain not in self.domain_success_criteria:
            return False  # Default to failure for unknown domains

        criteria = self.domain_success_criteria[domain]
        response_lower = response.lower()

        # For inform intents, check if required slots are addressed
        if intent_type == 'inform':
            required_slots = criteria.get('required_slots', [])
            return self._response_addresses_slots(response_lower, required_slots, domain)

        # For booking intents, check booking-specific slots
        elif intent_type in ['book', 'offer_book', 'offer_booked']:
            booking_slots = criteria.get('booking_slots', [])
            return self._response_addresses_slots(response_lower, booking_slots, domain)

        return True

    def _response_addresses_slots(self, response: str, slots: List[str], domain: str) -> bool:
        """Check if response addresses required slots, using domain-specific keywords if available"""
        # Domain-specific slot keywords
        domain_slot_keywords = {
            'restaurant': {
                'name': ['restaurant', 'name', 'called'],
                'food': ['food', 'cuisine', 'serves', 'chinese', 'indian', 'italian', 'british', 'french'],
                'area': ['area', 'north', 'south', 'east', 'west', 'centre', 'center'],
                'pricerange': ['price', 'expensive', 'cheap', 'moderate'],
                'phone': ['phone', 'number', 'call', 'contact'],
                'address': ['address', 'located', 'street', 'road', 'postcode', 'zip code'],
                'time': ['time', 'at', 'oclock', 'pm', 'am'],
                'day': ['day', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'],
                'people': ['people', 'person', 'for', 'guests']
            },
            'hotel': {
                'name': ['hotel', 'name', 'called'],
                'area': ['area', 'north', 'south', 'east', 'west', 'centre', 'center'],
                'pricerange': ['price', 'expensive', 'cheap', 'moderate'],
                'stars': ['star', 'stars', 'rating'],
                'type': ['type', 'guesthouse', 'hotel'],
                'phone': ['phone', 'number', 'call', 'contact'],
                'address': ['address', 'located', 'street', 'road', 'postcode', 'zip code'],
                'internet': ['internet', 'wifi'],
                'parking': ['parking'],
                'day': ['day', 'date'],
                'stay': ['stay', 'nights', 'duration'],
                'people': ['people', 'person', 'guests']
            },
            'train': {
                'departure': ['departure', 'from', 'leaving', 'origin'],
                'destination': ['destination', 'to', 'arriving'],
                'day': ['day', 'date'],
                'leaveat': ['leave at', 'depart', 'departure time'],
                'arriveby': ['arrive by', 'arrival time'],
                'people': ['people', 'person', 'guests'],
                'trainid': ['train', 'train id', 'number'],
                'duration': ['duration', 'time'],
                'price': ['price', 'cost', 'fare']
            },
            'taxi': {
                'departure': ['departure', 'from', 'pickup'],
                'destination': ['destination', 'to', 'dropoff'],
                'leaveat': ['leave at', 'depart', 'pickup time'],
                'arriveby': ['arrive by', 'arrival time'],
                'phone': ['phone', 'number', 'call', 'contact'],
                'type': ['type', 'car', 'vehicle'],
                'price': ['price', 'cost', 'fare']
            },
            'attraction': {
                'name': ['attraction', 'name', 'called'],
                'type': ['type', 'museum', 'park', 'theatre', 'cinema', 'gallery', 'church'],
                'area': ['area', 'north', 'south', 'east', 'west', 'centre', 'center'],
                'phone': ['phone', 'number', 'call', 'contact'],
                'address': ['address', 'located', 'street', 'road', 'postcode', 'zip code'],
                'fee': ['fee', 'cost', 'price', 'free', 'paid']
            },
            'general': {}
        }

        # Fallback slot keywords if domain not found
        default_slot_keywords = {
            'name': ['name', 'called'],
            'food': ['food', 'cuisine', 'serves'],
            'area': ['area', 'north', 'south', 'east', 'west', 'centre', 'center'],
            'pricerange': ['price', 'expensive', 'cheap', 'moderate'],
            'phone': ['phone', 'number', 'call', 'contact'],
            'address': ['address', 'located', 'street', 'road', 'postcode', 'zip code'],
            'time': ['time', 'at', 'oclock', 'pm', 'am'],
            'day': ['day', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'],
            'people': ['people', 'person', 'for', 'guests']
        }

        slot_keywords = domain_slot_keywords.get(domain, default_slot_keywords)

        addressed_slots = 0
        for slot in slots:
            keywords = slot_keywords.get(slot, [slot])
            if any(keyword in response for keyword in keywords):
                addressed_slots += 1

        # Require at least half of the slots to be addressed
        return addressed_slots >= max(1, len(slots) * 0.5)
    
    def _extract_user_goals(self, dialogue_turns: List[Dict]) -> Dict[str, Any]:
        """Extract user goals from dialogue turns"""
        goals = defaultdict(set)
        
        for turn in dialogue_turns:
            user_input_lower = turn.get('user_input', '').lower()
            domains = turn.get('domains', ['general'])
            for domain in domains:
                # Robust goal extraction using flexible patterns and synonyms
                # user_input_lower = user_input.lower()
                # Find entity: look for "find", "looking for", "search", "need", "want", "show me", "any", "are there"
                if any(kw in user_input_lower for kw in ['find', 'looking for', 'search', 'need', 'want', 'show me', 'any ', 'are there', 'suggest', 'recommend']):
                    goals[domain].add('find_entity')
                # Make booking: look for "book", "reserve", "make a reservation", "schedule", "appointment", "table", "room"
                if any(kw in user_input_lower for kw in ['book', 'reserve', 'make a reservation', 'schedule', 'appointment', 'table', 'room', 'booking']):
                    goals[domain].add('make_booking')
                # Get contact info: look for "phone", "address", "contact", "number", "call", "where is", "location", "postcode", "zip code"
                if any(kw in user_input_lower for kw in ['phone', 'address', 'contact', 'number', 'call', 'where is', 'location', 'postcode', 'zip code']):
                    goals[domain].add('get_contact_info')
                # Get price info: look for "price", "cost", "how much", "fee", "charge", "rate"
                if any(kw in user_input_lower for kw in ['price', 'cost', 'how much', 'fee', 'charge', 'rate']):
                    goals[domain].add('get_price_info')
                # Get time info: look for "time", "when", "open", "close", "leave", "arrive", "departure", "arrival"
                if any(kw in user_input_lower for kw in ['time', 'when', 'open', 'close', 'leave', 'arrive', 'departure', 'arrival']):
                    goals[domain].add('get_time_info')
                # Get directions/location: look for "directions", "how to get", "how do I get", "route", "way", "map"
                if any(kw in user_input_lower for kw in ['directions', 'how to get', 'how do i get', 'route', 'way', 'map']):
                    goals[domain].add('get_directions')
                # General thanks/goodbye: look for "thank", "thanks", "bye", "goodbye"
                if any(kw in user_input_lower for kw in ['thank', 'thanks', 'bye', 'goodbye']):
                    goals[domain].add('polite_closing')
        
        return dict(goals)
    
    def _evaluate_goal_completion(self, dialogue_turns: List[Dict], responses: List[str], 
                                user_goals: Dict[str, Set[str]]) -> Dict[str, Any]:
        """Evaluate how well user goals were completed"""
        total_goals = sum(len(goals) for goals in user_goals.values())
        completed_goals = 0
        
        # Simple goal completion check
        all_responses = ' '.join(responses).lower()
        
        for domain, goals in user_goals.items():
            for goal in goals:
                if goal == 'find_entity':
                    # Check for entity name or any entity-related info in responses
                    if self._has_entity_name(all_responses, domain) or any(kw in all_responses for kw in ['restaurant', 'hotel', 'attraction', 'train', 'taxi', 'hospital', 'police']):
                        completed_goals += 1
                elif goal == 'make_booking':
                    # Check for booking confirmation or booking-related keywords
                    if self._has_confirmation(all_responses) or any(
                    kw in all_responses for kw in [
                        'booked', 'reserved', 'confirmed', 'reference', 'booking', 'reservation', 'scheduled', 'appointment'
                    ]
                    ):
                        completed_goals += 1
                elif goal == 'get_contact_info':
                    # Check for phone/address/contact/postcode/zip code in responses
                    if any(
                    kw in all_responses for kw in [
                        'phone', 'address', 'contact', 'number', 'call', 'postcode', 'zip code', 'location'
                    ]
                    ):
                        completed_goals += 1
                elif goal == 'get_price_info':
                    # Check for price/cost/fee/charge/rate in responses
                    if any(
                    kw in all_responses for kw in [
                        'price', 'cost', 'fee', 'charge', 'rate', 'how much', 'expensive', 'cheap', 'free'
                    ]
                    ):
                        completed_goals += 1
                elif goal == 'get_time_info':
                    # Check for time/when/open/close/leave/arrive/departure/arrival in responses
                    if any(
                    kw in all_responses for kw in [
                        'time', 'when', 'open', 'close', 'leave', 'arrive', 'departure', 'arrival', 'am', 'pm', 'oclock'
                    ]
                    ):
                        completed_goals += 1
                elif goal == 'get_directions':
                    # Check for directions/route/way/map in responses
                    if any(
                    kw in all_responses for kw in [
                        'directions', 'route', 'way', 'map', 'how to get', 'how do i get'
                    ]
                    ):
                        completed_goals += 1
                elif goal == 'polite_closing':
                    # Check for thanks/bye/goodbye in responses
                    if any(
                    kw in all_responses for kw in [
                        'thank', 'thanks', 'bye', 'goodbye'
                    ]
                    ):
                        completed_goals += 1
            
        completion_rate = completed_goals / total_goals if total_goals > 0 else 1.0
        
        return {
            'completion_rate': completion_rate,
            'total_goals': total_goals,
            'completed_goals': completed_goals,
            'goals_by_domain': user_goals
        }





# Create a comprehensive evaluation framework that integrates all improved classes
class MultiDomainDialogueEvaluator:
    """
    Comprehensive evaluation framework for multi-domain task-oriented dialogue systems.
    Integrates all evaluation components and provides unified evaluation metrics.
    """
    
    def __init__(self):
        self.reference_resolver = ImprovedReferenceResolver()
        self.constraint_checker = ImprovedConstraintChecker()
        self.task_success_evaluator = ImprovedTaskSuccessEvaluator()
        # No ground_truth_loader used
        # Evaluation history for analysis
        self.evaluation_history = []
    
    def evaluate_dialogue(self, dialogue_turns: List[Dict], system_responses: List[str]) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a complete dialogue
        
        Args:
            dialogue_turns: List of dialogue turns with user input, intent, domain, etc.
            system_responses: List of system responses to evaluate
            
        Returns:
            Comprehensive evaluation results
        """
        if len(dialogue_turns) != len(system_responses):
            raise ValueError("Number of dialogue turns must match number of system responses")
        
        # Initialize results
        results = {
            'dialogue_id': f"eval_{len(self.evaluation_history)}",
            'total_turns': len(dialogue_turns),
            'evaluation_timestamp': datetime.datetime.now().strftime('%Y%m%d_%H%M%S'),  # Would use actual timestamp
            'turn_level_results': [],
            'dialogue_level_results': {}
        }
        
        # Evaluate each turn
        for i, (turn, response) in enumerate(zip(dialogue_turns, system_responses)):
            turn_result = self._evaluate_turn(i, turn, response, dialogue_turns[:i+1])
            results['turn_level_results'].append(turn_result)

        # try:
        #     timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        #     with open(f"dialogue_eval_{timestamp}.json", "w", encoding="utf-8") as f:
        #         json.dump(results, f, indent=2, ensure_ascii=False)
        # except:
        #     try:
        #         import yaml
        #         timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        #         output_file = f'multiwoz_results_{timestamp}.yaml'
        #         with open(output_file, "w", encoding="utf-8") as f:
        #             yaml.dump(results, f)
        #     except:
        #         pass
        
        # Aggregate dialogue-level metrics
        results['dialogue_level_results'] = self._aggregate_dialogue_metrics(
            dialogue_turns, system_responses, results['turn_level_results']
        )

        try:
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            with open(f"dialogue_eval_{timestamp}.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        except:
            try:
                import yaml
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f'dialogue_eval_{timestamp}.yaml'
                with open(output_file, "w", encoding="utf-8") as f:
                    yaml.dump(results, f)
            except:
                pass

        # Add to evaluation history

        self.evaluation_history.append(results)
        
        return results
    
    def _evaluate_turn(self, turn_index: int, turn: Dict, response: str, 
                      dialogue_history: List[Dict]) -> Dict[str, Any]:
        """Evaluate a single dialogue turn"""
        user_input = turn.get('user_input', '')
        # intent = turn.get('intent', 'unknown')
        domains = turn.get('domains', ['general'])
        # Use true_intent and extraction_result.dominant_intent for intent accuracy
        true_intent = turn.get('true_intent', 'unknown')
        predicted_mapped_intents = turn.get('extraction_result', {}).get('mapped_intents', ['unknown'])
        predicted_intent = turn.get('extraction_result', {}).get('dominant_intent', 'unknown')
        
        intent_match = False
        for intent in predicted_mapped_intents:
            intent_match = intent_match or (intent.lower() == true_intent.lower())

        # Reference resolution evaluation
        ref_results = self.reference_resolver.evaluate_resolution(
            dialogue_history, [response]
        )

        # Constraint adherence evaluation
        constraints = self.constraint_checker.extract_constraints_from_turns(dialogue_history)
        constraint_results = self.constraint_checker.check_adherence(
            constraints, response, domains
        )

        # Task success evaluation for this turn
        task_results = self.task_success_evaluator._evaluate_turn_success(
            turn, response, true_intent, domains
        )

        # No ground truth validation (entity_coverage_rate set to None)
        validation_results = {'entity_coverage_rate': None}

        return {
            'turn_index': turn_index,
            'user_input': user_input,
            'system_response': response,
            'domains': domains,
            'reference_resolution': ref_results,
            'constraint_adherence': constraint_results,
            'task_success': task_results,
            'ground_truth_validation': validation_results,
            'response_length': len(response.split()),
            'contains_entity_info': None,
            'true_intent': true_intent,
            'predicted_intent': predicted_intent,
            'predicted_mapped_intents': predicted_mapped_intents,
            'intent_match': intent_match
        }
    
    def _aggregate_dialogue_metrics(self, dialogue_turns: List[Dict], 
                                   system_responses: List[str],
                                   turn_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate turn-level results into dialogue-level metrics"""

        
        # Task success metrics
        task_success_results = self.task_success_evaluator.evaluate_task_success(
            dialogue_turns, system_responses
        )
        if (len(turn_results) == 0):
            errors = {
                'reference_resolution_failures': 0,
                'constraint_violations': 0,
                'task_failures': 0,
                'entity_coverage_failures': 0,
                'common_error_patterns': {},
                'error_by_domain': {},
                'error_by_intent': {}
            }
            return {
                'task_success_metrics': task_success_results,
                'overall_reference_resolution_rate': 0.0,
                'overall_constraint_adherence_rate': 0.0,
                'average_response_length': 0.0,
                'entity_coverage_rate': None,
                'intent_distribution': {},
                'domain_distribution': {},
                'intent_accuracy': 0.0,
                'error_analysis': errors,
                'dialogue_efficiency': {
                    'total_turns': 0,
                    'successful_turns': 0,
                    'efficiency_rate': 0.0
                }
            }
        # Reference resolution metrics
        total_ref_rate = sum(tr['reference_resolution']['overall_resolution_rate'] 
                           for tr in turn_results) / len(turn_results)

        # Constraint adherence metrics
        total_constraint_rate = sum(tr['constraint_adherence']['overall_adherence_rate'] 
                                  for tr in turn_results) / len(turn_results)

        # Response quality metrics
        avg_response_length = sum(tr['response_length'] for tr in turn_results) / len(turn_results)
        # entity_coverage_rate is not available (set to None)
        entity_coverage = None

        # Intent and domain distribution
        intent_distribution = {}
        domain_distribution = {}

        for turn in dialogue_turns:
            intent = turn.get('true_intent', 'unknown')
            intent_distribution[intent] = intent_distribution.get(intent, 0) + 1
            domains = turn.get('domains', ['general'])
            for domain in domains:
                domain_distribution[domain] = domain_distribution.get(domain, 0) + 1

        # Intent accuracy (using true_intent and extraction_result.dominant_intent)
        correct_intents = sum(1 for tr in turn_results if tr.get('intent_match'))
        intent_accuracy = correct_intents / len(turn_results) if turn_results else 0.0

        # Error analysis
        error_analysis = self._analyze_errors(turn_results)

        return {
            'task_success_metrics': task_success_results,
            'overall_reference_resolution_rate': total_ref_rate,
            'overall_constraint_adherence_rate': total_constraint_rate,
            'average_response_length': avg_response_length,
            'entity_coverage_rate': entity_coverage,
            'intent_distribution': intent_distribution,
            'domain_distribution': domain_distribution,
            'intent_accuracy': intent_accuracy,
            'error_analysis': error_analysis,
            'dialogue_efficiency': {
                'total_turns': len(dialogue_turns),
                'successful_turns': sum(1 for tr in turn_results if tr['task_success']['success']),
                'efficiency_rate': sum(1 for tr in turn_results if tr['task_success']['success']) / len(turn_results) if turn_results else 0.0
            }
        }
    
    def _analyze_errors(self, turn_results: List[Dict]) -> Dict[str, Any]:
        """Analyze common error patterns in the dialogue"""
        errors = {
            'reference_resolution_failures': 0,
            'constraint_violations': 0,
            'task_failures': 0,
            'entity_coverage_failures': 0,
            'common_error_patterns': {},
            'error_by_domain': {},
            'error_by_intent': {}
        }
        
        for tr in turn_results:
            domains = tr['domains']
            intent = tr['true_intent']
            
            
            # Track errors by type
            if tr['reference_resolution']['overall_resolution_rate'] is not None and tr['reference_resolution']['overall_resolution_rate']< 0.5:
                errors['reference_resolution_failures'] += 1
                for domain in domains:
                    errors['error_by_domain'][domain] = errors['error_by_domain'].get(domain, 0) + 1
                errors['error_by_intent'][intent] = errors['error_by_intent'].get(intent, 0) + 1
            
            if tr['constraint_adherence']['overall_adherence_rate'] is not None and tr['constraint_adherence']['overall_adherence_rate'] < 0.8:
                errors['constraint_violations'] += 1
            
            if not tr['task_success']['success']:
                errors['task_failures'] += 1
            
            if tr['ground_truth_validation']['entity_coverage_rate'] is not None and tr['ground_truth_validation']['entity_coverage_rate'] < 0.5:
                errors['entity_coverage_failures'] += 1
        
        return errors
    
    def evaluate_batch(self, dialogue_batch: List[Dict]) -> Dict[str, Any]:
        """Evaluate multiple dialogues and provide aggregate statistics"""
        batch_results = []
        
        for dialogue in dialogue_batch:
            turns = dialogue.get('turns', [])
            responses = dialogue.get('responses', [])
            
            if len(turns) == len(responses):
                result = self.evaluate_dialogue(turns, responses)
                batch_results.append(result)
        
        # Aggregate batch statistics
        return self._aggregate_batch_statistics(batch_results)
    
    def _aggregate_batch_statistics(self, batch_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate statistics across multiple dialogues"""
        if not batch_results:
            return {}
        
        # Calculate means across dialogues
        total_dialogues = len(batch_results)
        
        avg_success_rate = sum(
            br['dialogue_level_results']['task_success_metrics']['overall_success_rate'] 
            for br in batch_results
        ) / total_dialogues
        
        avg_inform_rate = sum(
            br['dialogue_level_results']['task_success_metrics']['inform_rate'] 
            for br in batch_results
        ) / total_dialogues
        
        avg_reference_resolution = sum(
            br['dialogue_level_results']['overall_reference_resolution_rate'] 
            for br in batch_results
        ) / total_dialogues
        
        avg_constraint_adherence = sum(
            br['dialogue_level_results']['overall_constraint_adherence_rate'] 
            for br in batch_results
        ) / total_dialogues
        
        # Aggregate domain and intent statistics
        all_domains = set()
        all_intents = set()
        
        for br in batch_results:
            all_domains.update(br['dialogue_level_results']['domain_distribution'].keys())
            all_intents.update(br['dialogue_level_results']['intent_distribution'].keys())
        
        return {
            'batch_size': total_dialogues,
            'average_success_rate': avg_success_rate,
            'average_inform_rate': avg_inform_rate,
            'average_reference_resolution_rate': avg_reference_resolution,
            'average_constraint_adherence_rate': avg_constraint_adherence,
            'domains_covered': list(all_domains),
            'intents_covered': list(all_intents),
            'evaluation_summary': {
                'excellent_dialogues': sum(1 for br in batch_results 
                                         if br['dialogue_level_results']['task_success_metrics']['overall_success_rate'] > 0.9),
                'good_dialogues': sum(1 for br in batch_results 
                                    if 0.7 <= br['dialogue_level_results']['task_success_metrics']['overall_success_rate'] <= 0.9),
                'poor_dialogues': sum(1 for br in batch_results 
                                    if br['dialogue_level_results']['task_success_metrics']['overall_success_rate'] < 0.7)
            }
        }
    
    def get_evaluation_report(self, dialogue_id: Optional[str] = None) -> str:
        """Generate a human-readable evaluation report"""
        if dialogue_id:
            # Get specific dialogue results
            results = next((r for r in self.evaluation_history if r['dialogue_id'] == dialogue_id), None)
            if not results:
                return f"No evaluation found for dialogue ID: {dialogue_id}"
        else:
            # Get most recent evaluation
            if not self.evaluation_history:
                return "No evaluations available"
            results = self.evaluation_history[-1]
        
        # Generate report
        report = f"""
=== DIALOGUE EVALUATION REPORT ===
Dialogue ID: {results['dialogue_id']}
Total Turns: {results['total_turns']}

=== TASK SUCCESS METRICS ===
Overall Success Rate: {results['dialogue_level_results']['task_success_metrics']['overall_success_rate']:.2%}
Inform Rate: {results['dialogue_level_results']['task_success_metrics']['inform_rate']:.2%}
Goal Completion Rate: {results['dialogue_level_results']['task_success_metrics']['goal_completion_rate']:.2%}

=== QUALITY METRICS ===
Reference Resolution Rate: {results['dialogue_level_results']['overall_reference_resolution_rate']:.2%}
Constraint Adherence Rate: {results['dialogue_level_results']['overall_constraint_adherence_rate']:.2%}
Entity Coverage Rate: {results['dialogue_level_results']['entity_coverage_rate']:.2%}

=== EFFICIENCY METRICS ===
Dialogue Efficiency: {results['dialogue_level_results']['dialogue_efficiency']['efficiency_rate']:.2%}
Average Response Length: {results['dialogue_level_results']['average_response_length']:.1f} words

=== DOMAIN DISTRIBUTION ===
"""
        
        for domain, count in results['dialogue_level_results']['domain_distribution'].items():
            report += f"{domain}: {count} turns\n"
        
        report += "\n=== ERROR ANALYSIS ===\n"
        errors = results['dialogue_level_results']['error_analysis']
        report += f"Reference Resolution Failures: {errors['reference_resolution_failures']}\n"
        report += f"Constraint Violations: {errors['constraint_violations']}\n"
        report += f"Task Failures: {errors['task_failures']}\n"
        report += f"Entity Coverage Failures: {errors['entity_coverage_failures']}\n"
        
        return report



class SSMGEvaluator:
    """Main evaluator for SSMG system"""

    def __init__(self):
        # self.ground_truth = GroundTruthLoader()
        # self.reference_resolver = ReferenceResolver()
        # self.constraint_checker = ConstraintChecker()
        # self.task_evaluator = TaskSuccessEvaluator()

        self.improved_evaluator = MultiDomainDialogueEvaluator()

    def evaluate_session(self, turns: List[Dict], responses: List[str], 
                        metrics_list: List[Any]) -> dict:
        """Evaluate a complete dialogue session using MultiDomainDialogueEvaluator and return detailed metrics."""
        # Use improved evaluator for all dialogue-level metrics
        eval_result = self.improved_evaluator.evaluate_dialogue(turns, responses)
        dialogue_metrics = eval_result.get('dialogue_level_results', {})

        # Context efficiency metrics (if available)
        if metrics_list:
            import numpy as np
            avg_context_tokens = float(np.mean([m.context_tokens for m in metrics_list]))
            avg_latency = float(np.mean([m.total_time for m in metrics_list]))
        else:
            avg_context_tokens = 0.0
            avg_latency = 0.0

        # Compose result metric dictionary
        result_metric = {
            'turn_accuracy': dialogue_metrics.get('intent_accuracy', 0.0),
            'task_success_rate': dialogue_metrics.get('task_success_metrics', {}).get('overall_success_rate', 0.0),
            'inform_rate': dialogue_metrics.get('task_success_metrics', {}).get('inform_rate', 0.0),
            'goal_completion_rate': dialogue_metrics.get('task_success_metrics', {}).get('goal_completion_rate', 0.0),
            'reference_resolution_accuracy': dialogue_metrics.get('overall_reference_resolution_rate', 0.0),
            'constraint_adherence': dialogue_metrics.get('overall_constraint_adherence_rate', 0.0),
            'avg_context_tokens': avg_context_tokens,
            'avg_latency': avg_latency,
            'success_per_thousand_tokens': (1000 * dialogue_metrics.get('task_success_metrics', {}).get('overall_success_rate', 0.0) / avg_context_tokens) if avg_context_tokens > 0 else 0.0,
            'success_per_second_latency': (dialogue_metrics.get('task_success_metrics', {}).get('overall_success_rate', 0.0) / avg_latency) if avg_latency > 0 else 0.0,
            'average_response_length': dialogue_metrics.get('average_response_length', 0.0),
            'entity_coverage_rate': dialogue_metrics.get('entity_coverage_rate', None),
            'domain_success_rates': dialogue_metrics.get('task_success_metrics', {}).get('domain_success_rates', {}),
            'intent_success_rates': dialogue_metrics.get('task_success_metrics', {}).get('intent_success_rates', {}),
            'dialogue_efficiency': dialogue_metrics.get('dialogue_efficiency', {}),
            'error_analysis': dialogue_metrics.get('error_analysis', {}),
            'intent_distribution': dialogue_metrics.get('intent_distribution', {}),
            'domain_distribution': dialogue_metrics.get('domain_distribution', {}),
        }
        return result_metric

    def compare_methods(self, results: Dict[str, EvaluationMetrics]) -> Dict[str, Any]:
        """Compare multiple methods and generate comparison report"""
        comparison = {}

        metrics_names = [
            'turn_accuracy', 'task_success_rate', 'avg_context_tokens', 
            'avg_latency', 'reference_resolution_accuracy', 'constraint_adherence',
            'token_efficiency', 'latency_efficiency'
        ]

        for metric in metrics_names:
            metric_values = {}
            for method, eval_metrics in results.items():
                metric_values[method] = getattr(eval_metrics, metric)

            # Find best performing method for this metric
            if metric in ['avg_context_tokens', 'avg_latency']:
                # Lower is better
                best_method = min(metric_values.items(), key=lambda x: x[1])
            else:
                # Higher is better
                best_method = max(metric_values.items(), key=lambda x: x[1])

            comparison[metric] = {
                'values': metric_values,
                'best_method': best_method[0],
                'best_value': best_method[1]
            }

        return comparison
