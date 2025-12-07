# Prompts Evaluator

An AI-powered tool for evaluating LLM agent responses against test cases. An evaluator LLM converses with an agent LLM, scoring responses on accuracy, clarity, naturalness, conciseness, and engagement.

## Features

- **Multiple evaluation modes**: Static script, Pipecat pipeline, or Streamlit web UI
- **Configurable prompts**: Customize agent and evaluator system prompts
- **Test case driven**: Define input/expected output pairs in JSON
- **Multi-turn conversations**: Evaluator continues dialogue until goals are met or `FLOW_COMPLETE`

## Quick Start

### Prerequisites

- Python 3.12+
- OpenAI API key (required)
- Google/Gemini API key (optional, for some modes)

### Installation

```bash
# Using uv (recommended)
uv sync

# Or pip
pip install -e .
```

### Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key  # optional
OPENAI_MODEL=gpt-4              # optional, defaults to gpt-4
```

### Usage

**Static evaluation** (simplest):
```bash
python static_evaluator.py
```

**Pipecat pipeline**:
```bash
python pipecat_evaluator.py
```

**Web UI**:
```bash
streamlit run web_evaluator.py
```

### Configuration

Edit `prompt.json` to configure:

```json
{
  "prompt": { "messages": [{ "role": "system", "content": "Agent prompt..." }] },
  "evaluation": { "messages": [{ "role": "system", "content": "Evaluator prompt..." }] },
  "test_cases": [
    { "input": "User question", "expected": "Expected behavior" }
  ]
}
```

## Docker

```bash
docker compose up --build
```

App available at http://localhost:8000

## Evaluation Criteria

| Criterion | Weight |
|-----------|--------|
| Accuracy | 20% |
| Clarity | 20% |
| Naturalness | 20% |
| Conciseness | 20% |
| Conversation | 20% |