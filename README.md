# constructor-agent-system

XML-defined LangGraph agent using `constructor_adapter.StatefulConstructorAdapter`.

## Installation

```bash
pip install -e .
```

If `constructor_adapter` is not already installed:

```bash
pip install "git+https://github.com/GiancarloSucci/ConstructorAdapter.git"
```

## Environment

```bash
export CONSTRUCTOR_API_KEY="..."
export CONSTRUCTOR_KM_ID="..."
export CONSTRUCTOR_API_URL="https://training.constructor.app/api/platform-kmapi/v1"
```

`CONSTRUCTOR_API_URL` is optional if the adapter default is correct.

## Run

```bash
constructor-agent run \
  --config config/path.xml \
  --query "Explain how Prolog can help an LLM analyze a repository." \
  --show-trace
```

or:

```bash
constructor-agent run \
  --config config/path.xml \
  --query-file query.txt
```

## XML endpoint attributes

- `id`: unique node id in the LangGraph graph.
- `llm_alias`: alias used by `ConstructorAdapter._get_llm_from_alias()`.
- `mode`: `direct` for LLM engine only, `model` for knowledge model only.
- `role`: semantic role used in the prompt.
- `timeout`, `request_timeout`, `retry_delay`: passed to `StatefulConstructorAdapter.query()`.
