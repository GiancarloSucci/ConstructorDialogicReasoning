# constructor-agent-system

XML-defined dialogic reasoning system for ConstructorPlatform.

The system supports two execution modes:

1. A LangGraph-based stateful execution path using `StatefulConstructorAdapter`.
2. A stateless dialogic adapter, `StatelessConstructorAdapterDialogic`, compatible with `StatelessConstructorAdapter`.

The XML configuration defines a sequence of `question` steps. Each step sends a prompt to a selected ConstructorPlatform LLM and may use the answer produced by previous steps.

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

`CONSTRUCTOR_API_URL` is optional if the default URL is correct.

`CONSTRUCTOR_KM_ID` is required for `mode="model"`.

For `mode="direct"`, the system can continue even if `CONSTRUCTOR_KM_ID` is missing, empty, invalid, or inaccessible. In that case, it creates a temporary empty knowledge model as a technical container for ConstructorPlatform chat sessions.

The alternative variable name `KNOWLEDGE_MODEL_ID` is also accepted.

## Execution modes

### `mode="direct"`

Uses the selected LLM engine directly through ConstructorPlatform.

### `mode="model"`

Uses the Constructor knowledge model associated with `CONSTRUCTOR_KM_ID` or `KNOWLEDGE_MODEL_ID`.

## Run with LangGraph

```bash
constructor-agent run \
  --configDialog config/path.xml \
  --prompt "Explain how Prolog can help an LLM analyze a repository." \
  --show-trace
```

The option `--config-dialog` is also accepted:

```bash
constructor-agent run \
  --config-dialog config/path.xml \
  --prompt "Explain how Prolog can help an LLM analyze a repository."
```

To read the prompt from a file:

```bash
constructor-agent run \
  --configDialog config/path.xml \
  --prompt-file query.txt
```

## XML configuration

The XML root element is `questionPath`.

Each `questionPath` contains one or more `question` elements.

```xml
<?xml version="1.0" encoding="UTF-8"?>
<questionPath name="example-question-path">

    <question
        id="first"
        llm_alias="gpt-4o-mini"
        mode="direct"
        role="initial_answer"
        timeout="900"
        request_timeout="30"
        retry_delay="5">
        <description>
            Produce the first answer.
        </description>
        <prompt>
Original user prompt:
{prompt}

Produce a precise technical answer.
Return only the answer.
        </prompt>
    </question>

</questionPath>
```

## XML question attributes

Each `question` may define the following attributes:

- `id`: unique question id in the execution path.
- `llm_alias`: ConstructorPlatform LLM alias.
- `mode`: `direct` or `model`.
- `role`: semantic role used by the default prompt.
- `timeout`: maximum time to wait for the model answer.
- `request_timeout`: timeout for each platform polling request.
- `retry_delay`: delay between polling attempts.

Each `question` may also contain:

- `description`: human-readable description of the question.
- `prompt`: optional custom prompt template.

If `prompt` is omitted, the system uses a default prompt. The first question receives the original user prompt. Later questions receive the original user prompt and the previous answer.

## Prompt placeholders

Custom prompts support these placeholders:

- `{prompt}`: original user prompt.
- `{answer}`: answer produced by the previous question.
- `{input}`: original prompt for the first question, previous answer for later questions.
- `{original_prompt}`: original user prompt.
- `{previous_answer}`: answer produced by the previous question.
- `{previous_question}`: id of the previous question.
- `{question_id}`: id of the current question.
- `{question_role}`: role of the current question.
- `{question_description}`: description of the current question.
- `{some_question_id.prompt}`: prompt actually sent to a previous question.
- `{some_question_id.answer}`: answer produced by a previous question.

For example, if a previous question has:

```xml
id="review"
```

then a later question can use:

```text
{review.prompt}
{review.answer}
```

`{finalizer.prompt}` and `{finalizer.answer}` are available only after the question with id `finalizer` has been executed. They cannot be used inside the prompt of `finalizer` itself.

## List available question candidates

```bash
constructor-agent list-questions
```

## Print XML blocks to copy into a configuration file

```bash
constructor-agent list-questions --xml
```

## Print only direct question candidates

```bash
constructor-agent list-questions --no-model --xml
```

## Print only model question candidates

```bash
constructor-agent list-questions --no-direct --xml
```

## Use explicit ConstructorPlatform parameters

```bash
constructor-agent list-questions \
  --api-url "https://training.constructor.app/api/platform-kmapi/v1" \
  --api-key "$CONSTRUCTOR_API_KEY" \
  --xml
```

## Example generated question block

```xml
<question id="direct_gemini_3_flash_preview" llm_alias="gemini-3-flash-preview" mode="direct" role="review_and_improve" timeout="900" request_timeout="30" retry_delay="5">
    <description>Question using gemini-3-flash-preview in direct mode.</description>
    <prompt>
{answer}
    </prompt>
</question>
```

## Sample configuration file

```xml
<?xml version="1.0" encoding="UTF-8"?>
<questionPath name="mixed-constructor-question-path">

    <question
        id="knowledge_answer"
        llm_alias="gemini-3-flash-preview"
        mode="model"
        role="initial_answer"
        timeout="900"
        request_timeout="30"
        retry_delay="5">
        <description>
            Answer using the Constructor Knowledge Model.
        </description>
        <prompt>
Original user prompt:
{prompt}

Produce a precise and explicit answer.
Return only the answer.
        </prompt>
    </question>

    <question
        id="independent_critic"
        llm_alias="gpt-4.1"
        mode="direct"
        role="review_and_improve"
        timeout="900"
        request_timeout="30"
        retry_delay="5">
        <description>
            Critically review the previous answer.
        </description>
        <prompt>
Original user prompt:
{prompt}

Previous answer:
{answer}

Prompt used by knowledge_answer:
{knowledge_answer.prompt}

Answer produced by knowledge_answer:
{knowledge_answer.answer}

Check correctness, completeness, and clarity.
Return a complete improved answer.
        </prompt>
    </question>

    <question
        id="finalizer"
        llm_alias="gemini-3-flash-preview"
        mode="model"
        role="final_synthesis"
        timeout="900"
        request_timeout="30"
        retry_delay="5">
        <description>
            Produce the final answer.
        </description>
        <prompt>
Original user prompt:
{prompt}

Current answer:
{answer}

Independent critic answer:
{independent_critic.answer}

Produce the final version.
Return only the final answer.
        </prompt>
    </question>

</questionPath>
```

## Stateless dialogic client

The package also provides a stateless client:

```python
from constructor_agent import ConstructorPlatformConfig, ConstructorStatelessClient

client = ConstructorStatelessClient.from_xml(
    xml_path="config/path.xml",
    platform_config=ConstructorPlatformConfig.from_environment(),
    llm_alias="gpt-4o-mini",
    mode="direct",
)

result = client.run(
    prompt="Explain how Prolog can help an LLM analyze a repository.",
    timeout=900,
    request_timeout=30,
    retry_delay=5,
)

print(result.final_answer)
```

The stateless client uses `StatelessConstructorAdapterDialogic` and does not use LangGraph.

## Architecture

Main modules:

```text
platform_config.py
    ConstructorPlatformConfig and DEFAULT_API_URL.

stateful_constructor_client.py
    StatefulConstructorClient for LangGraph execution.

stateless_constructor_adapter_dialogic.py
    StatelessConstructorAdapterDialogic.
    It extends StatelessConstructorAdapter and executes a questionPath internally.

stateless_constructor_client.py
    ConstructorStatelessClient, a thin wrapper around StatelessConstructorAdapterDialogic.

domain.py
    QuestionSpec, QuestionPathConfig, QuestionExchange, AgentState.

config_loader.py
    XmlQuestionConfigLoader for questionPath XML files.

prompts.py
    PromptFactory and placeholder substitution.

graph_builder.py
    LangGraph construction from QuestionPathConfig.

langchain_constructor_model.py
    LangChain BaseChatModel wrapper around StatefulConstructorClient.

runner.py
    ConstructorAgentRunner facade.

cli.py
    Typer command-line interface.
```

## Validation

After installation or code changes:

```bash
pip install -e .
python3.12 -m compileall src
```

Then run:

```bash
constructor-agent run \
  --configDialog config/path.xml \
  --prompt "Explain how Prolog can help an LLM analyze a repository." \
  --show-trace
```