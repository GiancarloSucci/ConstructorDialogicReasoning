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

## To see all models
```bash
constructor-agent list-endpoints
```

## To see the XML blocks to copy in the configuration file path.xml
```bash
constructor-agent list-endpoints --xml
```

## To see the only the endpoints direct
```bash
constructor-agent list-endpoints --no-model --xml
```

## To see the only the endpoints model
```bash
constructor-agent list-endpoints --no-direct --xml
```

## With explicit parameters
```bash
constructor-agent list-endpoints \
  --api-url "https://training.constructor.app/api/platform-kmapi/v1" \
  --api-key "$CONSTRUCTOR_API_KEY" \
  --xml
```
## Note
In the ConstructorAdapter the endpoint is generated as (mode, llm_alias), therefore the list generates automatically candidates like
```bash
<endpoint id="direct_gemini_3_flash_preview" llm_alias="gemini-3-flash-preview" mode="direct" role="review_and_improve" timeout="300" request_timeout="15" retry_delay="3">
    <description>Endpoint using gemini-3-flash-preview in direct mode.</description>
</endpoint>
```


## Sample configuration file

<agentPath name="mixed-constructor-path">

    <endpoint
        id="knowledge_answer"
        llm_alias="gemini-3-flash-preview"
        mode="model"
        role="initial_answer">
        <description>
            Answer using the Constructor Knowledge Model.
        </description>
    </endpoint>

    <endpoint
        id="independent_critic"
        llm_alias="gpt-4.1"
        mode="direct"
        role="review_and_improve">
        <description>
            Critically review the previous answer without relying on the Knowledge Model.
        </description>
    </endpoint>

    <endpoint
        id="finalizer"
        llm_alias="gemini-3-flash-preview"
        mode="model"
        role="final_synthesis">
        <description>
            Produce the final answer, checking again against the Knowledge Model.
        </description>
    </endpoint>

</agentPath>
