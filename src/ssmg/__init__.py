"""
SSMG: Session-scoped Short-term Memory Graph
A lightweight, ephemeral semantic memory system for multi-turn dialogue.
"""

__version__ = "0.1.0"
__author__ = "SSMG Research Team"

from .graph import SSMGGraph, Node, Edge, NodeType, RelationType
from .extractor import SSMGExtractor, ExtractionResult
from .summarizer import SSMGSummarizer, SummaryConfig
from .integration import SSMGDialogueAgent, LLMInterface, DialogueSession

__all__ = [
    'SSMGGraph', 'Node', 'Edge', 'NodeType', 'RelationType',
    'SSMGExtractor', 'ExtractionResult', 
    'SSMGSummarizer', 'SummaryConfig',
    'SSMGDialogueAgent', 'LLMInterface', 'DialogueSession'
]
