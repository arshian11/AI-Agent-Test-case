A comprehensive evaluation framework for Session-scoped Short-term Memory Graph (SSMG) dialogue agents, featuring multiple baseline comparisons and extensive testing capabilities.

## 🚀 Overview

This repository implements SSMG (Session-scoped Short-term Memory Graph), a lightweight, ephemeral semantic memory system for multi-turn dialogue applications. SSMG maintains structured context during conversations while ensuring privacy through session-based memory management.

### Key Features

- **Memory Graph Architecture**: Maintains entities, facts, intents, and constraints in a structured graph
- **Session-scoped Privacy**: All memory discarded at session end, no persistent storage
- **Token Efficiency**: Achieves ~90% token savings compared to full-history approaches
- **Multi-LLM Support**: Compatible with Groq, LLaMA, and other language model APIs
- **Comprehensive Evaluation**: Includes baseline comparisons and performance metrics
- **Real-world Testing**: Evaluated on MultiWOZ and custom dialogue datasets

## 📁 Repository Structure

```
AI-Agent-Test-case/
├── configs/                    # Configuration files
│   └── default_config.json    # Default SSMG parameters
├── data/                      # Data loading utilities
│   └── loaders.py            # MultiWOZ and synthetic data loaders
├── evaluation/               # Evaluation framework
│   ├── baselines.py         # Baseline implementations
│   ├── eval_runner.py       # Main evaluation runner
│   └── metrics.py           # Performance metrics
├── notebooks/               # Interactive demonstrations
│   └── demo.ipynb          # SSMG demo and comparisons
├── src/ssmg/               # Core SSMG implementation
│   ├── __init__.py         # Package initialization
│   ├── extractor.py        # Information extraction
│   ├── graph.py            # Memory graph implementation
│   ├── integration.py      # LLM integration and dialogue flow
│   ├── summarizer.py       # Context summarization
│   └── utils.py            # Utility functions
├── .gitignore              # Git ignore patterns
└── README.md               # This file
```

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- pip or conda package manager

### Quick Setup

1. **Clone the repository**:
```bash
git clone https://github.com/arshian11/AI-Agent-Test-case.git
cd AI-Agent-Test-case
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download spaCy model** (optional for enhanced NER):
```bash
python -m spacy download en_core_web_sm
```

### API Keys Setup

For enhanced performance with external APIs:

```bash
# Groq API (recommended)
export GROQ_API_KEY="your_groq_api_key"

# Or add to .env file
echo "GROQ_API_KEY=your_groq_api_key" > .env
```

## 🚀 Quick Start

### Basic Usage

```python
from src.ssmg.integration import SSMGDialogueAgent, GroqAPIInterface
from src.ssmg.summarizer import SummaryConfig

# Initialize with Groq API
llm = GroqAPIInterface(api_key="your_api_key")
agent = SSMGDialogueAgent(
    llm_interface=llm,
    graph_config={'max_nodes': 50, 'max_ttl_turns': 8},
    summary_config=SummaryConfig(max_tokens=200)
)

# Start conversation
session_id = agent.start_session("demo_session")

# Process dialogue turns
response, metrics = agent.process_turn("I want to order pizza but no onions")
print(f"Assistant: {response}")
print(f"Context tokens used: {metrics.context_tokens}")

# Continue conversation
response, metrics = agent.process_turn("Add garlic bread too")
print(f"Assistant: {response}")

# End session (clears all memory)
agent.end_session()
```

### Interactive Demo

Run the Jupyter notebook for an interactive demonstration:

```bash
jupyter notebook notebooks/demo.ipynb
```

The demo showcases:
- SSMG vs baseline comparisons
- Token efficiency analysis
- Graph structure visualization
- Constraint adherence testing

## 📊 Evaluation Framework

### Run Complete Evaluation

```bash
python evaluation/eval_runner.py \
    --config configs/default_config.json \
    --output-dir results \
    --max-dialogues 100 \
    --run-baselines
```

### Baseline Methods

The framework includes several baseline implementations:

1. **Full History**: Uses complete conversation history
2. **Sliding Window**: Maintains fixed-size context window
3. **Simple RAG**: Basic retrieval-augmented generation
4. **SSMG** (proposed): Session-scoped memory graph

### Performance Metrics

- **Turn Accuracy**: Intent recognition accuracy
- **Task Success Rate**: Goal completion percentage
- **Token Efficiency**: Context tokens per turn
- **Latency**: Response generation time
- **Constraint Adherence**: User preference compliance
- **Reference Resolution**: Pronoun/reference handling

## 🔧 Configuration

### Graph Configuration

```json
{
  "graph": {
    "max_nodes": 50,
    "max_ttl_turns": 8,
    "decay_rate": 0.05
  },
  "summarizer": {
    "max_tokens": 200,
    "max_nodes": 16,
    "recency_weight": 0.4,
    "confidence_weight": 0.3,
    "relevance_weight": 0.3
  }
}
```

### LLM Configuration

```json
{
  "llm": {
    "model_name": "llama-3.3-70b-versatile",
    "max_tokens": 150,
    "temperature": 0.7
  }
}
```

## 📈 Performance Results

Based on evaluation on MultiWOZ dataset:

| Method | Avg Context Tokens | Latency (s) | Turn Accuracy |
|--------|-------------------|-------------|---------------|
| Full History | 280.6 | 0.56 | 0.75 |
| Sliding Window | 276.2 | 0.41 | 0.73 |
| **SSMG** | **29.0** | **0.40** | **0.78** |

**Key Findings**:
- 89.7% reduction in context tokens vs full history
- Maintains superior accuracy despite reduced context
- Consistent sub-second response times
- Effective constraint persistence across turns

## 🏗️ Architecture

### Core Components

1. **SSMGGraph**: Manages nodes (entities, facts, intents, constraints) and edges (relationships)
2. **SSMGExtractor**: Extracts semantic information using spaCy NLP and rule-based patterns
3. **SSMGSummarizer**: Generates concise context summaries with relevance scoring
4. **LLM Integration**: Supports multiple LLM backends with consistent interface

### Memory Management

- **TTL-based Eviction**: Nodes expire after configurable turns
- **Confidence Decay**: Node confidence decreases over time
- **Priority-based Retention**: Constraints and intents prioritized
- **Session Isolation**: Complete memory reset between sessions







## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 Acknowledgments

- spaCy for natural language processing capabilities
- Groq for high-performance language model inference
- MultiWOZ dataset creators for evaluation benchmarks
- Open source community for inspiration and feedback
