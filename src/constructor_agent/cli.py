from __future__ import annotations
from rich.table import Table

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from constructor_agent.constructor_stateful_client import ConstructorPlatformConfig
from constructor_agent.query_provider import FileQueryProvider, InlineQueryProvider
from constructor_agent.runner import ConstructorAgentRunner

app = typer.Typer(help="Run an XML-defined LangGraph agent on ConstructorPlatform.")
console = Console()


def _read_query(query: Optional[str], query_file: Optional[Path]) -> str:
    if query and query_file:
        raise typer.BadParameter("Use either --query or --query-file, not both.")
    if query:
        return InlineQueryProvider(query).get_query()
    if query_file:
        return FileQueryProvider(query_file).get_query()
    raise typer.BadParameter("Provide either --query or --query-file.")


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="XML path configuration."),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Query text."),
    query_file: Optional[Path] = typer.Option(None, "--query-file", "-f", help="File containing the query."),
    api_url: Optional[str] = typer.Option(None, "--api-url", help="Constructor API URL."),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Constructor API key."),
    km_id: Optional[str] = typer.Option(None, "--km-id", help="Constructor knowledge model id."),
    show_trace: bool = typer.Option(False, "--show-trace", help="Show all intermediate endpoint answers."),
) -> None:
    user_query = _read_query(query, query_file)
    platform_config = ConstructorPlatformConfig(api_url=api_url, api_key=api_key, km_id=km_id)
    runner = ConstructorAgentRunner.from_xml(config, platform_config)
    result = runner.run(user_query)

    console.print(Panel(result.final_answer, title="Final answer"))
    console.print(Panel(result.explanation, title="How the answer was produced"))

    if show_trace:
        for i, exchange in enumerate(result.state.get("exchanges", []), start=1):
            console.print(
                Panel(
                    exchange.answer,
                    title=f"Step {i}: {exchange.endpoint_id} / {exchange.llm_alias} / {exchange.mode}",
                )
            )

@app.command("list-endpoints")
def list_endpoints(
    api_url: Optional[str] = typer.Option(None, "--api-url", help="Constructor API URL."),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="Constructor API key."),
    km_id: Optional[str] = typer.Option(None, "--km-id", help="Constructor knowledge model id. Not required for listing LLMs."),
    direct: bool = typer.Option(True, "--direct/--no-direct", help="Include direct LLM-engine endpoints."),
    model: bool = typer.Option(True, "--model/--no-model", help="Include knowledge-model endpoints."),
    xml: bool = typer.Option(False, "--xml", help="Print XML snippets for the endpoint candidates."),
) -> None:
    platform_config = ConstructorPlatformConfig(
        api_url=api_url,
        api_key=api_key,
        km_id=km_id,
    )

    candidates = ConstructorAgentRunner.list_constructor_endpoints(
        platform_config=platform_config,
        include_direct=direct,
        include_model=model,
    )

    if not candidates:
        console.print("[bold red]No ConstructorPlatform language models were returned.[/bold red]")
        raise typer.Exit(code=1)

    table = Table(title="ConstructorPlatform endpoint candidates")
    table.add_column("#", justify="right", style="bold")
    table.add_column("endpoint id", style="bold cyan")
    table.add_column("mode", style="magenta")
    table.add_column("llm_alias", style="bold green")
    table.add_column("name")
    table.add_column("llm id", overflow="fold")

    for index, candidate in enumerate(candidates, start=1):
        table.add_row(
            str(index),
            candidate.endpoint_id,
            candidate.mode,
            candidate.llm_alias,
            candidate.llm_name,
            candidate.llm_id,
        )

    console.print(table)

    if xml:
        snippets = [candidate.xml_snippet for candidate in candidates]
        console.print(Panel("\n\n".join(snippets), title="XML snippets"))

if __name__ == "__main__":
    app()
