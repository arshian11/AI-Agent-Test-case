"""
Baseline implementations for comparison with SSMG
"""

import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from collections import deque

@dataclass
class BaselineResult:
    """Result from baseline method"""
    response: str
    context_tokens: int
    latency: float
    method_name: str

class FullHistoryBaseline:
    """Baseline that uses full conversation history"""

    def __init__(self, llm_interface):
        self.llm = llm_interface
        self.history = []

    def process_turn(self, user_input: str) -> BaselineResult:
        start_time = time.time()

        # Add user input to history
        self.history.append(f"User: {user_input}")

        # Create full history prompt
        full_context = "\n".join(self.history)
        prompt = f"{full_context}\nAssistant:"

        # Generate response
        response, input_tokens, _ = self.llm.generate_response(prompt)

        # Add assistant response to history
        self.history.append(f"Assistant: {response}")

        latency = time.time() - start_time

        return BaselineResult(
            response=response,
            context_tokens=input_tokens,
            latency=latency,
            method_name="full_history"
        )

    def reset(self):
        self.history = []

class SlidingWindowBaseline:
    """Baseline that uses sliding window of recent turns"""

    def __init__(self, llm_interface, window_size: int = 5):
        self.llm = llm_interface
        self.window_size = window_size
        self.history = deque(maxlen=window_size * 2)  # *2 for user+assistant pairs

    def process_turn(self, user_input: str) -> BaselineResult:
        start_time = time.time()

        # Add user input to sliding window
        self.history.append(f"User: {user_input}")

        # Create context from sliding window
        context = "\n".join(self.history)
        prompt = f"{context}\nAssistant:"

        # Generate response
        response, input_tokens, _ = self.llm.generate_response(prompt)

        # Add assistant response to sliding window
        self.history.append(f"Assistant: {response}")

        latency = time.time() - start_time

        return BaselineResult(
            response=response,
            context_tokens=input_tokens,
            latency=latency,
            method_name=f"sliding_window_{self.window_size}"
        )

    def reset(self):
        self.history.clear()

class RAGBaseline:
    """Simple RAG baseline using turn-based retrieval"""

    def __init__(self, llm_interface, max_retrieved: int = 3):
        self.llm = llm_interface
        self.max_retrieved = max_retrieved
        self.turn_store = []  # Store of previous turns

    def process_turn(self, user_input: str) -> BaselineResult:
        start_time = time.time()

        # Simple retrieval based on keyword overlap
        relevant_turns = self._retrieve_relevant_turns(user_input)

        # Create context from retrieved turns
        if relevant_turns:
            context = "Previous relevant context:\n" + "\n".join(relevant_turns)
            prompt = f"{context}\n\nUser: {user_input}\nAssistant:"
        else:
            prompt = f"User: {user_input}\nAssistant:"

        # Generate response
        response, input_tokens, _ = self.llm.generate_response(prompt)

        # Store current turn
        turn_text = f"User: {user_input}\nAssistant: {response}"
        self.turn_store.append(turn_text)

        latency = time.time() - start_time

        return BaselineResult(
            response=response,
            context_tokens=input_tokens,
            latency=latency,
            method_name=f"rag_{self.max_retrieved}"
        )

    def _retrieve_relevant_turns(self, query: str) -> List[str]:
        """Simple keyword-based retrieval"""
        query_words = set(query.lower().split())

        scored_turns = []
        for turn in self.turn_store:
            turn_words = set(turn.lower().split())
            overlap = len(query_words & turn_words)
            if overlap > 0:
                scored_turns.append((overlap, turn))

        # Sort by relevance and return top k
        scored_turns.sort(reverse=True, key=lambda x: x[0])
        return [turn for _, turn in scored_turns[:self.max_retrieved]]

    def reset(self):
        self.turn_store = []
