"""
SSMG Integration - Connects SSMG with LLM APIs and manages the dialogue flow
Handles prompt assembly, LLM calls, and response processing.
"""

import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json

from .graph import SSMGGraph, Node, NodeType
from .extractor import SSMGExtractor, ExtractionResult
from .summarizer import SSMGSummarizer, SummaryConfig


logger = logging.getLogger(__name__)

# Add after existing imports at the top
import torch

@dataclass
class TurnMetrics:
    """Metrics for a single dialogue turn"""
    turn_id: int
    extraction_time: float
    summarization_time: float
    llm_time: float
    total_time: float
    input_tokens: int
    output_tokens: int
    context_tokens: int
    summary_length: int
    nodes_added: int
    edges_added: int
    nodes_evicted: int = 0

@dataclass
class DialogueSession:
    """Represents a complete dialogue session"""
    session_id: str
    start_time: datetime = field(default_factory=datetime.now)
    turns: List[Dict[str, Any]] = field(default_factory=list)
    metrics: List[TurnMetrics] = field(default_factory=list)
    user_id: str = "user"
    metadata: Dict[str, Any] = field(default_factory=dict)

class LLMInterface:
    """Abstract interface for LLM backends"""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name

    def generate_response(self, prompt: str, max_tokens: int = 150) -> Tuple[str, int, int]:
        """Generate response from LLM
        Returns: (response_text, input_tokens, output_tokens)
        """
        # Mock implementation - replace with actual LLM API calls
        word_count = len(prompt.split())
        input_tokens = int(word_count * 1.3)  # Rough estimation

        # Simple mock response based on prompt content
        if "pizza" in prompt.lower():
            response = "I'll help you with your pizza order. Based on your preferences, I can recommend options that avoid onions as requested."
        elif "pasta" in prompt.lower():
            response = "Great choice! I can help you with pasta options. I'll make sure to avoid onions as you mentioned."
        elif "constraint" in prompt.lower() or "avoid" in prompt.lower():
            response = "I understand your preferences and constraints. I'll make sure to follow them in my recommendations."
        else:
            response = "I understand. How can I help you with your request?"

        output_tokens = len(response.split())
        return response, input_tokens, output_tokens

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text"""
        # Rough estimation: 1 token ≈ 0.75 words
        return int(len(text.split()) * 1.33)
    
class LLaMAInterface(LLMInterface):
    """LLaMA interface using HuggingFace Transformers - DEFAULT INTERFACE"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf", device: str = "auto"):
        super().__init__(model_name)
        self.device = device
        self.model = None
        self.tokenizer = None
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize LLaMA model and tokenizer"""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            logger.info(f"Loading LLaMA model: {self.model_name}")
            
            # Determine device
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True, padding_side="left"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "low_cpu_mem_usage": True
            }
            if self.device == "cuda":
                model_kwargs["device_map"] = "auto"
            
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
            if self.device == "cpu":
                self.model = self.model.to("cpu")
                
        except Exception as e:
            logger.error(f"Failed to load LLaMA: {e}. Using fallback.")
            self.model = None
    
    def generate_response(self, prompt: str, max_tokens: int = 150) -> Tuple[str, int, int]:
        """Generate response using LLaMA"""
        if not self.model:
            return super().generate_response(prompt, max_tokens)
        
        # Format for LLaMA chat
        formatted_prompt = f"<s>[INST] {self._extract_user_input(prompt)} [/INST]"
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=0.7, do_sample=True
            )
        
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response.strip(), inputs["input_ids"].shape[1], len(outputs[0]) - inputs["input_ids"].shape[1]
    
    def _extract_user_input(self, prompt: str) -> str:
        """Extract user input from prompt"""
        if "User: " in prompt:
            return prompt.split("User: ")[-1].replace("Assistant:", "").strip()
        return prompt.strip()

# class WorkingDialogueInterface(LLMInterface):
#     """Improved dialogue interface with better prompt formatting"""
    
#     def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
#         super().__init__(model_name)
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.model = None
#         self.tokenizer = None
#         self._initialize_model()
        
#     def _initialize_model(self):
#         try:
#             from transformers import AutoTokenizer, AutoModelForCausalLM
            
#             logger.info(f"Loading dialogue model: {self.model_name}")
            
#             self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#             self.tokenizer.pad_token = self.tokenizer.eos_token
            
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 self.model_name,
#                 dtype=torch.float32,
#                 low_cpu_mem_usage=True,
#                 offload_folder="offload",
#                 llm_int8_enable_fp32_cpu_offload=True,
#                 load_in_4bit=True,
#                 device_map="cpu"
#             ).to(self.device)
            
#             logger.info(f"Model loaded successfully on {self.device}")
            
#         except Exception as e:
#             logger.error(f"Failed to load model: {e}. Using fallback responses.")
#             self.model = None
    
#     def generate_response(self, prompt: str, max_tokens: int = 150) -> Tuple[str, int, int]:
#         if not self.model:
#             return self._fallback_response(prompt, max_tokens)
        
#         try:
#             # Clean and format the prompt for dialogue
#             formatted_prompt = self._format_dialogue_prompt(prompt)
            
#             # Tokenize with proper settings
#             inputs = self.tokenizer(
#                 formatted_prompt,
#                 return_tensors="pt",
#                 truncation=True,
#                 max_length=512,  # Shorter context to avoid issues
#                 padding=False
#                 ).to(self.device)
            
#             input_length = inputs['input_ids'].shape[1]
            
#             # Generate with better parameters
#             with torch.no_grad():
#                 outputs = self.model.generate(
#                     **inputs,
#                     max_new_tokens=min(max_tokens, 50),  # Shorter responses
#                     do_sample=True,
#                     temperature=0.8,
#                     top_p=0.9,
#                     repetition_penalty=1.2,  # Reduce repetition
#                     no_repeat_ngram_size=3,  # Prevent 3-gram repetition
#                     pad_token_id=self.tokenizer.eos_token_id,
#                     eos_token_id=self.tokenizer.eos_token_id
#                 )
            
#             # Decode only the new tokens
#             new_tokens = outputs[0][input_length:]
#             response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
#             # Clean the response
#             response = self._clean_response(response)
            
#             return response, input_length, len(new_tokens)
            
#         except Exception as e:
#             logger.error(f"Generation error: {e}")
#             return self._fallback_response(prompt, max_tokens)
    
#     def _format_dialogue_prompt(self, prompt: str) -> str:
#         """Format prompt for better dialogue generation"""
#         # Extract context and user input
#         if "Context from previous conversation:" in prompt:
#             parts = prompt.split("User: ")
#             if len(parts) > 1:
#                 context = parts[0].replace("Context from previous conversation:", "").strip()
#                 user_input = parts[1].replace("Assistant:", "").strip()
                
#                 # Simple dialogue format
#                 return f"Context: {context}\nUser: {user_input}\nAssistant:"
        
#         # Simple format
#         return prompt.replace("Assistant:", "").strip() + "\nAssistant:"
    
#     def _clean_response(self, response: str) -> str:
#         """Clean and improve the response"""
#         # Remove common artifacts
#         response = response.strip()
        
#         # Remove instruction tokens and artifacts
#         for artifact in ["[INST]", "[/INST]", "<s>", "</s>", "[INFO]"]:
#             response = response.replace(artifact, "")
        
#         # Split into sentences and take the first complete one
#         sentences = response.split('. ')
#         if sentences:
#             clean_response = sentences[0].strip()
#             if clean_response and not clean_response.endswith('.'):
#                 clean_response += '.'
            
#             # Ensure it's a reasonable response length
#             if len(clean_response) > 10 and len(clean_response) < 200:
#                 return clean_response
        
#         # Fallback to first reasonable line
#         lines = [line.strip() for line in response.split('\n') if line.strip()]
#         for line in lines:
#             if 10 < len(line) < 200 and not any(x in line.lower() for x in ['[', 'sudo', 'php', 'file']):
#                 return line
        
#         return "I understand your request. How can I help you further?"
    
#     def _fallback_response(self, prompt: str, max_tokens: int) -> Tuple[str, int, int]:
#         """Generate contextual fallback responses"""
#         user_input = prompt.split("User:")[-1].replace("Assistant:", "").strip().lower()
        
#         # Context-aware responses
#         if any(word in user_input for word in ['pizza', 'order']):
#             response = "I'll help you with your pizza order. What would you like me to do?"
#         elif any(word in user_input for word in ['pasta', 'change']):
#             response = "I can change your order to pasta. I'll make sure to follow your preferences."
#         elif any(word in user_input for word in ['bread', 'add']):
#             response = "I'll add that to your order."
#         elif any(word in user_input for word in ['cost', 'price', 'total']):
#             response = "Let me calculate the total cost for your order."
#         elif any(word in user_input for word in ['onions', 'no', 'avoid']):
#             response = "I'll make sure to avoid onions as requested."
#         else:
#             response = "I understand. How can I help you with your request?"
        
#         return response, len(prompt.split()), len(response.split())

import requests
import os


from groq import Groq
from typing import Tuple
# import logging

# logger = logging.getLogger(__name__)
class GroqAPIInterface(LLMInterface):
    """Interface for Groq Language Model API"""

    def __init__(self, api_key: str = None, model_name: str = "llama-3.3-70b-versatile"):
        super().__init__(model_name)
        if api_key is None:
            api_key = "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"

        if not api_key:
            logger.warning("No GROQ_API_KEY found. Falling back to mock responses.")
            self.client = None
        else:
            try:
                self.client = Groq(api_key=api_key)
                logger.info(f"Groq client initialized with model {self.model_name}")
            except Exception as e:
                logger.error(f"Failed to initialize Groq client: {e}")
                self.client = None

    def generate_response(self, prompt: str, max_tokens: int = 150) -> Tuple[str, int, int]:
        """Generate response from Groq API
        Returns: (response_text, input_tokens, output_tokens)
        """
        if not self.client:
            logger.warning("Groq client not available, using fallback response.")
            return super().generate_response(prompt, max_tokens)

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                model=self.model_name,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9
            )

            # Extract the assistant's message
            message = chat_completion.choices[0].message.content
            input_tokens = chat_completion.usage.prompt_tokens
            output_tokens = chat_completion.usage.completion_tokens

            return message.strip(), input_tokens, output_tokens

        except Exception as e:
            logger.error(f"Groq API error: {e}, using fallback response.")
            return super().generate_response(prompt, max_tokens)

# class GeminiAPIInterface(LLMInterface):
#     """Interface for Gemini Language Model API"""

#     def __init__(self, api_key: str = None, model_name: str = "gemini-v1"):
#         super().__init__(model_name)
#         if api_key is None:
#             api_key = os.getenv("GEMINI_API_KEY")
#         self.api_key = "AIzaSyD_R0g1Hyzg39RUDeQ5Tm86Fbwdicslcb0"
#         self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"  # Replace with actual Gemini API URL

#     def generate_response(self, prompt: str, max_tokens: int = 150) -> Tuple[str, int, int]:
#         headers = {
#             "Authorization": f"Bearer {self.api_key}",
#             "Content-Type": "application/json"
#         }
#         data = {
#             "model": self.model_name,
#             "prompt": prompt,
#             "max_tokens": max_tokens,
#             "temperature": 0.7,
#             "top_p": 0.9,
#             "frequency_penalty": 0,
#             "presence_penalty": 0
#         }
#         try:
#             response = requests.post(self.api_url, headers=headers, json=data, timeout=30)
#             response.raise_for_status()
#             resp_json = response.json()
#             text = resp_json.get("text") or resp_json.get("generated_text")
#             # Estimate tokens as word count approximation
#             input_tokens = len(prompt.split())
#             output_tokens = len(text.split()) if text else 0
#             return text.strip(), input_tokens, output_tokens
#         except Exception as e:
#             # fallback to default mock response
#             print(f"Gemini API error: {e}, falling back")
#             return super().generate_response(prompt, max_tokens)



class SSMGDialogueAgent:
    """Main SSMG dialogue agent that orchestrates all components"""

    def __init__(self, 
                 llm_interface: LLMInterface = None,
                 graph_config: Dict[str, Any] = None,
                 summary_config: SummaryConfig = None,
                 spacy_model: str = "en_core_web_sm"):

        # Initialize components
        self.llm = llm_interface or GeminiAPIInterface()
        self.graph = SSMGGraph(**(graph_config or {}))
        self.extractor = SSMGExtractor(spacy_model)
        self.summarizer = SSMGSummarizer(summary_config or SummaryConfig())

        # Session management
        self.current_session: Optional[DialogueSession] = None
        self.system_prompt = self._build_system_prompt()

        logger.info(f"Initialized SSMG agent with {self.llm.model_name}")

    def start_session(self, session_id: str = None, user_id: str = "user") -> str:
        """Start a new dialogue session"""
        if session_id is None:
            session_id = f"session_{int(time.time())}"

        self.current_session = DialogueSession(
            session_id=session_id,
            user_id=user_id
        )

        # Reset graph for new session
        self.graph = SSMGGraph(
            max_nodes=self.graph.max_nodes,
            max_ttl_turns=self.graph.max_ttl_turns,
            decay_rate=self.graph.decay_rate
        )

        user_node = Node(
            id='user_session',  # Fixed ID, not dependent on turn
            type=NodeType.ENTITY,
            content='user',
            confidence=1.0,
            turn_id=0,  # Turn 0 indicates session-level node
            metadata={'is_user': True, 'persistent': True}
        )
        self.graph.add_node(user_node)

        logger.info(f"Started session: {session_id}")
        return session_id

    def end_session(self) -> Optional[DialogueSession]:
        """End current session and return session data"""
        if self.current_session:
            session = self.current_session
            session.end_time = datetime.now()

            # Clear ephemeral data
            self.graph = SSMGGraph()
            self.current_session = None

            logger.info(f"Ended session: {session.session_id}")
            return session
        return None

    def process_turn(self, user_input: str) -> Tuple[str, TurnMetrics]:
        """Process a single dialogue turn"""
        if not self.current_session:
            raise ValueError("No active session. Call start_session() first.")

        turn_start = time.time()
        turn_id = len(self.current_session.turns)

        # Step 1: Extract information from user input
        extract_start = time.time()
        extraction_result = self.extractor.extract(user_input, turn_id, self.current_session.user_id)
        extraction_time = time.time() - extract_start

        # Step 2: Update graph
        nodes_before = len(self.graph.nodes)

        for node in extraction_result.nodes:
            self.graph.add_node(node)

        for edge in extraction_result.edges:
            self.graph.add_edge(edge)

        # self.graph.infer_relations_for_turn()


        nodes_added = len(self.graph.nodes) - nodes_before
        edges_added = len(extraction_result.edges)

        # Advance turn (applies TTL and decay)
        self.graph.advance_turn()

        # Step 3: Generate summary
        summary_start = time.time()
        current_entities = {node.content for node in extraction_result.nodes 
                          if node.type == NodeType.ENTITY}
        current_intent = None
        for node in extraction_result.nodes:
            if node.type == NodeType.INTENT:
                current_intent = node.content
                break

        summary = self.summarizer.summarize(self.graph, current_entities, current_intent)
        summarization_time = time.time() - summary_start

        # Step 4: Generate LLM response
        llm_start = time.time()
        full_prompt = self._build_prompt(summary, user_input)
        response, input_tokens, output_tokens = self.llm.generate_response(full_prompt)
        llm_time = time.time() - llm_start

        total_time = time.time() - turn_start

        # Step 5: Record turn data
        turn_data = {
            'turn_id': turn_id,
            'user_input': user_input,
            'assistant_response': response,
            'summary': summary,
            'extraction_result': {
                'nodes': len(extraction_result.nodes),
                'edges': len(extraction_result.edges),
                'confidence': extraction_result.confidence
            },
            'graph_state': {
                'total_nodes': len(self.graph.nodes),
                'total_edges': len(self.graph.edges),
                'current_turn': self.graph.current_turn
            },
            'timestamp': datetime.now().isoformat()
        }

        self.current_session.turns.append(turn_data)

        # Step 6: Create metrics
        metrics = TurnMetrics(
            turn_id=turn_id,
            extraction_time=extraction_time,
            summarization_time=summarization_time,
            llm_time=llm_time,
            total_time=total_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            context_tokens=self.llm.estimate_tokens(summary),
            summary_length=len(summary),
            nodes_added=nodes_added,
            edges_added=edges_added
        )

        self.current_session.metrics.append(metrics)

        logger.info(f"Turn {turn_id}: {total_time:.3f}s total, {len(summary)} char summary")
        return response, metrics

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the LLM"""
        return """You are a helpful assistant with access to conversation context and user preferences. 
Use the provided context to maintain consistency and remember user constraints across the conversation.
Be concise but helpful, and always respect user preferences and constraints mentioned in the context."""

    def _build_prompt(self, summary: str, user_input: str) -> str:
        """Build the full prompt for LLM including context summary"""
        if summary and summary != "No context available.":
            prompt = f"""Context from previous conversation: {summary}

            User: {user_input}
            Assistant:"""
        else:
            prompt = f"""User: {user_input}
            Assistant:"""

        return prompt

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary statistics for current session"""
        if not self.current_session:
            return {}

        metrics = self.current_session.metrics
        if not metrics:
            return {'session_id': self.current_session.session_id, 'turns': 0}

        total_time = sum(m.total_time for m in metrics)
        total_input_tokens = sum(m.input_tokens for m in metrics)
        total_output_tokens = sum(m.output_tokens for m in metrics)
        avg_summary_length = sum(m.summary_length for m in metrics) / len(metrics)

        return {
            'session_id': self.current_session.session_id,
            'turns': len(metrics),
            'total_time': total_time,
            'avg_time_per_turn': total_time / len(metrics),
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'avg_summary_length': avg_summary_length,
            'final_graph_size': len(self.graph.nodes),
            'graph_turns': self.graph.current_turn
        }
