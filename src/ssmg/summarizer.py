"""
SSMG Summarizer - Converts graph state to concise structured summaries for LLM consumption
Handles relevance scoring, node selection, and template-based summary generation.
"""

from typing import List, Dict, Set, Optional, Any, Tuple
from dataclasses import dataclass
import logging
from datetime import datetime

from .graph import SSMGGraph, Node, NodeType, RelationType


logger = logging.getLogger(__name__)

@dataclass
class SummaryConfig:
    """Configuration for summary generation"""
    max_tokens: int = 200
    max_nodes: int = 16
    recency_weight: float = 0.4
    confidence_weight: float = 0.3
    relevance_weight: float = 0.3
    include_constraints: bool = True
    include_intents: bool = True

class RelevanceScorer:
    """Scores nodes for relevance to current context"""

    def __init__(self, config: SummaryConfig):
        self.config = config

    def score_nodes(self, graph: SSMGGraph, 
                   current_entities: Set[str] = None, #type: ignore
                   current_intent: str = None) -> List[Tuple[str, float]]: #type: ignore
        """Score all nodes for relevance to current context"""
        scored_nodes = []
        current_entities = current_entities or set()

        for node_id, node in graph.nodes.items():
            score = self._calculate_node_score(node, graph, current_entities, current_intent)
            scored_nodes.append((node_id, score))

        # Sort by score (descending)
        scored_nodes.sort(key=lambda x: x[1], reverse=True)
        if not scored_nodes:
            logger.warning(f"[Turn ID: {graph.current_turn}] No Scored Nodes were present for Entities:{current_entities} and Intents:{current_intent}")
        return scored_nodes

    def _calculate_node_score(self, node: Node, graph: SSMGGraph, 
                            current_entities: Set[str], current_intent: str) -> float:
        """Calculate relevance score for a single node"""

        # Base scores
        recency_score = self._recency_score(node, graph.current_turn)
        confidence_score = node.confidence
        relevance_score = self._relevance_score(node, current_entities, current_intent)

        # Type-based boosts
        type_boost = 1.0
        if node.type.value == NodeType.CONSTRAINT.value:
            type_boost = 1.5  # Constraints are very important
        elif node.type.value == NodeType.INTENT.value:
            type_boost = 1.3  # Intents are important
        elif node.type.value == NodeType.ENTITY.value:
            type_boost = 1.1  # Entities moderately important

        # Connectivity boost (well-connected nodes are more important)
        connectivity_score = len(graph.get_neighbors(node.id, max_depth=1)) * 0.1

        # Combine scores
        final_score = (
            self.config.recency_weight * recency_score +
            self.config.confidence_weight * confidence_score +
            self.config.relevance_weight * relevance_score +
            connectivity_score
        ) * type_boost

        return min(final_score, 2.0)  # Cap at 2.0

    def _recency_score(self, node: Node, current_turn: int) -> float:
        """Score based on how recent the node is"""
        age = node.age_in_turns(current_turn)
        if age == 0:
            return 1.0
        elif age <= 2:
            return 0.8
        elif age <= 5:
            return 0.5
        else:
            return 0.2

    def _relevance_score(self, node: Node, current_entities: Set[str], current_intent: str) -> float:
        """Score based on relevance to current context"""
        score = 0.0

        # Entity overlap
        node_text = node.content.lower()
        for entity in current_entities:
            if entity.lower() in node_text or node_text in entity.lower():
                score += 0.5

        # Intent match
        if current_intent and node.type == NodeType.INTENT:
            if current_intent in node.content or node.content in current_intent:
                score += 0.7

        # Constraint relevance (always high)
        if node.type == NodeType.CONSTRAINT:
            score += 0.6

        return min(score, 1.0)

class TemplateGenerator:
    """Generates structured text summaries from selected nodes"""

    def __init__(self):
        self.section_templates = {
            'CONSTRAINTS': 'User constraints: {}',
            'INTENTS': 'Current goals: {}', 
            'ENTITIES': 'Mentioned items: {}',
            'FACTS': 'Context: {}',
            'RELATIONS': 'Relationships: {}'
        }

    def generate(self, selected_nodes: List[Node], graph: SSMGGraph) -> str:
        """Generate structured summary from selected nodes"""

        # Group nodes by type
        grouped = {
            'CONSTRAINTS': [],
            'INTENTS': [],
            'ENTITIES': [],
            'FACTS': []
        }

        for node in selected_nodes:
            if node.type.value == NodeType.CONSTRAINT.value:
                grouped['CONSTRAINTS'].append(node.content)
            elif node.type.value == NodeType.INTENT.value:
                grouped['INTENTS'].append(node.content)
            elif node.type.value == NodeType.ENTITY.value:
                grouped['ENTITIES'].append(node.content)
            elif node.type.value == NodeType.FACT.value:
                grouped['FACTS'].append(node.content)

        # Generate sections
        sections = []

        if grouped['CONSTRAINTS']:
            constraints_text = ', '.join(grouped['CONSTRAINTS'])
            sections.append(f"CONSTRAINTS: {constraints_text}")

        if grouped['INTENTS']:
            intents_text = ', '.join(grouped['INTENTS'])
            sections.append(f"GOALS: {intents_text}")

        if grouped['ENTITIES']:
            entities_text = ', '.join(grouped['ENTITIES'][:8])  # Limit entities
            sections.append(f"ITEMS: {entities_text}")

        if grouped['FACTS']:
            facts_text = ', '.join(grouped['FACTS'][:4])  # Limit facts
            sections.append(f"CONTEXT: {facts_text}")

        # Add key relations
        relations = self._extract_key_relations(selected_nodes, graph)
        if relations:
            sections.append(f"RELATIONS: {relations}")
        if not sections:
            logger.warning(f"[Turn ID: {graph.current_turn}] No Sections Were Created in Summary")
        return ' | '.join(sections) if sections else "No context available."

    def _extract_key_relations(self, nodes: List[Node], graph: SSMGGraph) -> str:
        """Extract key relationships between selected nodes"""
        node_ids = {node.id for node in nodes}
        relations = []

        for edge in graph.edges.values():
            if edge.source_id in node_ids and edge.target_id in node_ids:
                source_content = graph.nodes[edge.source_id].content
                target_content = graph.nodes[edge.target_id].content
                relation_text = f"{source_content} {edge.relation.value} {target_content}"
                relations.append(relation_text)

        return ', '.join(relations[:3])  # Limit to top 3 relations

class SSMGSummarizer:
    """Main summarizer class for SSMG"""

    def __init__(self, config: SummaryConfig = None): #type: ignore
        self.config = config or SummaryConfig()
        self.scorer = RelevanceScorer(self.config)
        self.generator = TemplateGenerator()

    # def summarize(self, graph: SSMGGraph, 
    #              current_entities: Set[str] = None, #type: ignore
    #              current_intent: str = None) -> str: #type: ignore
    #     """Generate a concise summary of the graph state"""

    #     if not graph.nodes:
    #         return "No context available."

    #     # Score and select top nodes
    #     scored_nodes = self.scorer.score_nodes(graph, current_entities, current_intent)

    #     # Select top nodes up to max_nodes limit
    #     selected_node_ids = [node_id for node_id, _ in scored_nodes[:self.config.max_nodes]]
    #     selected_nodes = [graph.nodes[node_id] for node_id in selected_node_ids 
    #                      if node_id in graph.nodes]

    #     # Ensure constraints and intents are included if they exist
    #     if self.config.include_constraints:
    #         constraints = graph.get_nodes_by_type(NodeType.CONSTRAINT)
    #         for constraint in constraints[:3]:  # Include top 3 constraints
    #             if constraint not in selected_nodes:
    #                 selected_nodes.append(constraint)

    #     if self.config.include_intents:
    #         intents = graph.get_nodes_by_type(NodeType.INTENT)
    #         recent_intents = [intent for intent in intents if intent.turn_id >= graph.current_turn - 2]
    #         for intent in recent_intents[:2]:  # Include recent intents
    #             if intent not in selected_nodes:
    #                 selected_nodes.append(intent)

    #     # Generate summary
    #     summary = self.generator.generate(selected_nodes, graph)

    #     # Truncate if too long (rough token estimation: 1 token ≈ 4 chars)
    #     estimated_tokens = len(summary) // 4
    #     if estimated_tokens > self.config.max_tokens:
    #         # Truncate summary
    #         char_limit = self.config.max_tokens * 4
    #         summary = summary[:char_limit] + "..."

    #     logger.debug(f"Generated summary: {len(summary)} chars, ~{estimated_tokens} tokens")
    #     return summary
    def summarize(self, graph: SSMGGraph, 
             current_entities: Set[str] = None, #type: ignore
             current_intent: str = None) -> str: #type: ignore
        """Generate a concise summary of the graph state, robust to missing edges"""

        if not graph.nodes:
            logger.warning(f"[Turn ID: {graph.current_turn}] No Nodes present in the graph to build summary")
            return "No context available."
        

        # 1️⃣ Score all nodes
        scored_nodes = self.scorer.score_nodes(graph, current_entities, current_intent)

        # 2️⃣ Select top nodes up to max_nodes
        selected_node_ids = [node_id for node_id, _ in scored_nodes[:self.config.max_nodes]]
        selected_nodes = [graph.nodes[node_id] for node_id in selected_node_ids if node_id in graph.nodes]

        # 3️⃣ Ensure user node is always included
        # user_nodes = graph.get_nodes_by_type(NodeType.ENTITY)
        # for n in user_nodes:
        #     if n.content.lower() == "user" and n not in selected_nodes:
        #         selected_nodes.insert(0, n)

        # 4️⃣ Include recent nodes (last 3 turns) to capture context
        recent_nodes = [n for n in graph.nodes.values() if n.turn_id >= graph.current_turn - 3]
        for n in recent_nodes:
            if n not in selected_nodes:
                selected_nodes.append(n)

        # 5️⃣ Include top constraints if configured
        if self.config.include_constraints:
            constraints = graph.get_nodes_by_type(NodeType.CONSTRAINT)
            for constraint in constraints[:3]:
                if constraint not in selected_nodes:
                    selected_nodes.append(constraint)

        # 6️⃣ Include recent intents if configured
        if self.config.include_intents:
            intents = graph.get_nodes_by_type(NodeType.INTENT)
            recent_intents = [intent for intent in intents if intent.turn_id >= graph.current_turn - 2]
            for intent in recent_intents[:2]:
                if intent not in selected_nodes:
                    selected_nodes.append(intent)

        # 7️⃣ Fallback: If no nodes selected, include first 5 nodes
        if not selected_nodes:
            logger.warning("No Nodes were selected so using fallback to select frist 5 nodes")
            selected_nodes = list(graph.nodes.values())[:5]

        # 8️⃣ Generate summary using TemplateGenerator
        summary = self.generator.generate(selected_nodes, graph)

        # 9️⃣ Fallback for empty edges
        # if "RELATIONS: " not in summary and graph.edges:
        #     relations = ', '.join([f"{graph.nodes[e.source_id].content} {e.relation.value} {graph.nodes[e.target_id].content}" 
        #                         for e in list(graph.edges.values())[:3]])
        #     if relations:
        #         summary += f" | RELATIONS: {relations}"
        #     logger.warning("No RELATIONS were present in the Summary using fallback")
        valid_edges = []
        for eid, e in graph.edges.items():
            if e.source_id not in graph.nodes or e.target_id not in graph.nodes:
                logger.warning(f"⚠️ Invalid edge found: {eid} (source={e.source_id}, target={e.target_id})")
            else:
                valid_edges.append(e)

        # Then use only valid_edges for summarization
        if "RELATIONS: " not in summary and valid_edges:
            relations = ', '.join([
                f"{graph.nodes[e.source_id].content} {e.relation.value} {graph.nodes[e.target_id].content}"
                for e in valid_edges[:3]
            ])
            if relations:
                summary += f" | RELATIONS: {relations}"

        # 10️⃣ Truncate if too long (approx 1 token ≈ 4 chars)
        estimated_tokens = len(summary) // 4
        if estimated_tokens > self.config.max_tokens:
            char_limit = self.config.max_tokens * 4
            summary = summary[:char_limit] + "..."

        logger.debug(f"Modified Generated summary: {len(summary)} chars, ~{estimated_tokens} tokens")
        return summary


    def summarize_updates(self, graph: SSMGGraph, recent_nodes: List[Node]) -> str:
        """Generate a summary focusing on recent updates"""
        if not recent_nodes:
            return self.summarize(graph)

        # Create a focused summary of recent changes
        update_summary = self.generator.generate(recent_nodes, graph)

        # Get baseline context
        baseline_summary = self.summarize(graph)

        # Combine if both are short enough
        combined = f"RECENT: {update_summary} | CONTEXT: {baseline_summary}"
        if len(combined) // 4 <= self.config.max_tokens:
            return combined
        else:
            return update_summary  # Prioritize recent updates

    def get_debug_info(self, graph: SSMGGraph) -> Dict[str, Any]:
        """Get debug information about summarization process"""
        scored_nodes = self.scorer.score_nodes(graph)

        return {
            'total_nodes': len(graph.nodes),
            'total_edges': len(graph.edges),
            'current_turn': graph.current_turn,
            'top_scored_nodes': scored_nodes[:5],
            'node_type_counts': {
                node_type.value: len(graph.get_nodes_by_type(node_type))
                for node_type in NodeType
            }
        }

