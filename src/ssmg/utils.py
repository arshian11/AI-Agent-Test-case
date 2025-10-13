"""
Utility functions for SSMG
"""

import json
import logging
from typing import Dict, Any, List
from pathlib import Path
import time
from datetime import datetime
import hashlib

# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     filename='dialogue_agent.log',  # <-- logs go here
#     filemode='w'  # 'w' to overwrite each run, 'a' to append
# )
logger = logging.getLogger(__name__)

def setup_logging(level: str = "INFO", log_file: str = None):
    """Setup logging configuration"""
    log_level = getattr(logging, level.upper())

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        return {}

def save_results(results: Dict[str, Any], output_path: str):
    """Save results to JSON file with timestamp"""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Add metadata
    results['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'version': '0.1.0'
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_file}")

def generate_session_id(user_id: str = "anonymous") -> str:
    """Generate unique session ID"""
    timestamp = str(int(time.time() * 1000))
    content = f"{user_id}_{timestamp}"
    hash_obj = hashlib.md5(content.encode())
    return f"session_{hash_obj.hexdigest()[:8]}"

def estimate_tokens(text: str) -> int:
    """Rough token estimation for text"""
    # Simple heuristic: ~0.75 tokens per word
    words = len(text.split())
    return int(words * 1.33)

def truncate_text(text: str, max_tokens: int = 200) -> str:
    """Truncate text to approximate token limit"""
    estimated_tokens = estimate_tokens(text)
    if estimated_tokens <= max_tokens:
        return text

    # Truncate to approximate character limit
    char_ratio = max_tokens / estimated_tokens
    char_limit = int(len(text) * char_ratio * 0.9)  # 90% safety margin

    if char_limit < len(text):
        return text[:char_limit] + "..."
    return text

def validate_graph_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize graph configuration"""
    defaults = {
        'max_nodes': 50,
        'max_ttl_turns': 8,
        'decay_rate': 0.05
    }

    validated = defaults.copy()
    if 'graph' in config:
        validated.update(config['graph'])

    # Validate ranges
    validated['max_nodes'] = max(10, min(200, validated['max_nodes']))
    validated['max_ttl_turns'] = max(2, min(20, validated['max_ttl_turns']))
    validated['decay_rate'] = max(0.01, min(0.2, validated['decay_rate']))

    return validated

class Timer:
    """Context manager for timing operations"""

    def __init__(self, operation_name: str = "Operation"):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time #type: ignore
        logger.debug(f"{self.operation_name} completed in {elapsed:.3f}s")

    @property
    def elapsed(self) -> float:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return 0.0

def create_test_dialogue() -> List[str]:
    """Create a test dialogue for demonstrations"""
    return [
        "I want to order a pizza. But please don't use onions.",
        "Add garlic bread too.",
        "Actually, change it to pasta.",
        "Make sure no onions in the pasta either.",
        "What's the total cost?"
    ]
