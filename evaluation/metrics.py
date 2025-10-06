"""
Evaluation metrics for SSMG and baselines
"""

import re
from typing import List, Dict, Any, Tuple, Set
from dataclasses import dataclass
import numpy as np
from collections import defaultdict

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

class GroundTruthLoader:
    """Loads ground truth data for evaluation"""

    def __init__(self):
        # Mock ground truth - replace with actual dataset loading
        self.ground_truth = {
            'intents': {
                'i want pizza': 'order',
                'change to pasta': 'modify',
                'add garlic bread': 'add',
                'no onions please': 'constraint'
            },
            'entities': {
                'i want pizza': ['pizza'],
                'change to pasta': ['pasta'],
                'add garlic bread': ['garlic bread'],
                'no onions please': ['onions']
            },
            'constraints': {
                'no onions please': [('avoid', 'onions')],
                'i am allergic to nuts': [('allergic', 'nuts')]
            }
        }

    def get_intent(self, utterance: str) -> str:
        return self.ground_truth['intents'].get(utterance.lower(), 'unknown')

    def get_entities(self, utterance: str) -> List[str]:
        return self.ground_truth['entities'].get(utterance.lower(), [])

    def get_constraints(self, utterance: str) -> List[Tuple[str, str]]:
        return self.ground_truth['constraints'].get(utterance.lower(), [])

class ReferenceResolver:
    """Evaluates reference resolution accuracy"""

    def __init__(self):
        self.pronouns = {'it', 'he', 'she', 'they', 'this', 'that', 'them'}
        self.references = {'the order', 'my order', 'the pizza', 'the pasta'}

    def extract_references(self, text: str) -> List[str]:
        """Extract pronouns and references from text"""
        words = text.lower().split()
        refs = []

        for word in words:
            if word in self.pronouns:
                refs.append(word)

        for ref in self.references:
            if ref in text.lower():
                refs.append(ref)

        return refs

    def evaluate_resolution(self, dialogue_turns: List[Dict], responses: List[str]) -> float:
        """Evaluate how well references are resolved in responses"""
        total_refs = 0
        resolved_refs = 0

        for i, (turn, response) in enumerate(zip(dialogue_turns, responses)):
            user_input = turn.get('user_input', '')
            refs = self.extract_references(user_input)

            if not refs:
                continue

            total_refs += len(refs)

            # Simple heuristic: response should contain specific entities, not just pronouns
            response_words = response.lower().split()
            specific_words = [w for w in response_words if w not in self.pronouns and len(w) > 3]

            if len(specific_words) > 2:  # Reasonable specificity
                resolved_refs += len(refs)

        return resolved_refs / total_refs if total_refs > 0 else 1.0

class ConstraintChecker:
    """Evaluates constraint adherence in responses"""

    def __init__(self):
        self.constraint_patterns = {
            'avoid': ['avoid', 'without', 'no', "don't"],
            'allergic': ['allergic', 'allergy'],
            'require': ['must', 'need', 'require'],
            'only': ['only', 'just']
        }

    def extract_constraints_from_turns(self, turns: List[Dict]) -> Dict[str, Set[str]]:
        """Extract constraints mentioned across all turns"""
        constraints = defaultdict(set)

        for turn in turns:
            user_input = turn.get('user_input', '').lower()

            # Look for constraint patterns
            if any(word in user_input for word in ['no', "don't", 'without', 'avoid']):
                # Extract what to avoid
                words = user_input.split()
                for i, word in enumerate(words):
                    if word in ['no', "don't", 'without', 'avoid'] and i + 1 < len(words):
                        item = words[i + 1]
                        constraints['avoid'].add(item)

            if 'allergic' in user_input:
                words = user_input.split()
                for i, word in enumerate(words):
                    if word == 'to' and i + 1 < len(words):
                        item = words[i + 1]
                        constraints['allergic'].add(item)

        return constraints

    def check_adherence(self, constraints: Dict[str, Set[str]], response: str) -> float:
        """Check if response adheres to constraints"""
        violations = 0
        total_constraints = sum(len(items) for items in constraints.values())

        if total_constraints == 0:
            return 1.0

        response_lower = response.lower()

        # Check avoidance constraints
        for item in constraints.get('avoid', set()):
            if item in response_lower:
                violations += 1

        # Check allergy constraints  
        for item in constraints.get('allergic', set()):
            if item in response_lower:
                violations += 1

        adherence_rate = (total_constraints - violations) / total_constraints
        return max(0.0, adherence_rate)

class TaskSuccessEvaluator:
    """Evaluates task completion success"""

    def __init__(self):
        self.task_keywords = {
            'order': ['order', 'get', 'buy', 'purchase'],
            'modify': ['change', 'modify', 'update', 'instead'],
            'add': ['add', 'include', 'also', 'plus'],
            'cancel': ['cancel', 'remove', 'delete']
        }

    def evaluate_task_success(self, intent: str, response: str) -> bool:
        """Evaluate if response successfully addresses the intent"""
        if intent == 'unknown':
            return True  # Can't evaluate unknown intents

        response_lower = response.lower()

        # Check if response contains appropriate keywords for the intent
        if intent in self.task_keywords:
            keywords = self.task_keywords[intent]
            return any(keyword in response_lower for keyword in keywords)

        # Default heuristic - response should be substantive
        return len(response.split()) > 3

class SSMGEvaluator:
    """Main evaluator for SSMG system"""

    def __init__(self):
        self.ground_truth = GroundTruthLoader()
        self.reference_resolver = ReferenceResolver()
        self.constraint_checker = ConstraintChecker()
        self.task_evaluator = TaskSuccessEvaluator()

    def evaluate_session(self, turns: List[Dict], responses: List[str], 
                        metrics_list: List[Any]) -> EvaluationMetrics:
        """Evaluate a complete dialogue session"""

        # Turn-level accuracy (intent recognition)
        correct_intents = 0
        for turn in turns:
            user_input = turn.get('user_input', '')
            predicted_intent = turn.get('extraction_result', {}).get('dominant_intent', 'unknown')
            true_intent = self.ground_truth.get_intent(user_input)

            if predicted_intent == true_intent:
                correct_intents += 1

        turn_accuracy = correct_intents / len(turns) if turns else 0.0

        # Task success rate
        successful_tasks = 0
        for i, turn in enumerate(turns):
            user_input = turn.get('user_input', '')
            true_intent = self.ground_truth.get_intent(user_input)
            response = responses[i] if i < len(responses) else ""

            if self.task_evaluator.evaluate_task_success(true_intent, response):
                successful_tasks += 1

        task_success_rate = successful_tasks / len(turns) if turns else 0.0

        # Context efficiency metrics
        if metrics_list:
            avg_context_tokens = np.mean([m.context_tokens for m in metrics_list])
            avg_latency = np.mean([m.total_time for m in metrics_list])
        else:
            avg_context_tokens = 0.0
            avg_latency = 0.0

        # Reference resolution accuracy
        ref_accuracy = self.reference_resolver.evaluate_resolution(turns, responses)

        # Constraint adherence
        constraints = self.constraint_checker.extract_constraints_from_turns(turns)
        constraint_scores = []
        for response in responses:
            score = self.constraint_checker.check_adherence(constraints, response)
            constraint_scores.append(score)

        constraint_adherence = np.mean(constraint_scores) if constraint_scores else 1.0

        # Efficiency metrics
        token_efficiency = turn_accuracy / (avg_context_tokens + 1e-6)
        latency_efficiency = turn_accuracy / (avg_latency + 1e-6)

        return EvaluationMetrics(
            turn_accuracy=turn_accuracy,
            task_success_rate=task_success_rate,
            avg_context_tokens=avg_context_tokens,
            avg_latency=avg_latency,
            reference_resolution_accuracy=ref_accuracy,
            constraint_adherence=constraint_adherence,
            token_efficiency=token_efficiency,
            latency_efficiency=latency_efficiency
        )

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
