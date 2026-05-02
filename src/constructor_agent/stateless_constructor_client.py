from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from constructor_agent.platform_config import ConstructorPlatformConfig
from constructor_agent.stateless_constructor_adapter_dialogic import (
    StatelessConstructorAdapterDialogic,
)


@dataclass(frozen=True)
class StatelessClientRunResult:
    final_answer: str
    configuration_path: Optional[Path]


class ConstructorStatelessClient:
    """
    Thin client around StatelessConstructorAdapterDialogic.

    This client executes the XML-defined dialogic path without LangGraph and
    without StatefulConstructorAdapter sessions.
    """

    def __init__(
        self,
        platform_config: ConstructorPlatformConfig | None = None,
        llm_alias: str = "gpt-4o-mini",
        mode: str = "direct",
        configuration_path: str | Path | None = None,
    ) -> None:
        self.platform_config = (
            platform_config or ConstructorPlatformConfig.from_environment()
        )
        self.llm_alias = llm_alias
        self.mode = mode
        self.configuration_path = (
            Path(configuration_path) if configuration_path is not None else None
        )

        self.adapter = StatelessConstructorAdapterDialogic(
            mode=self.mode,
            api_url=self.platform_config.resolved_api_url(),
            api_key=self.platform_config.resolved_api_key(),
            km_id=self.platform_config.resolved_km_id(optional=True),
            llm_alias=self.llm_alias,
        )

        if self.configuration_path is not None:
            self.load_configuration(self.configuration_path)

    @classmethod
    def from_xml(
        cls,
        xml_path: str | Path,
        platform_config: ConstructorPlatformConfig | None = None,
        llm_alias: str = "gpt-4o-mini",
        mode: str = "direct",
    ) -> "ConstructorStatelessClient":
        return cls(
            platform_config=platform_config,
            llm_alias=llm_alias,
            mode=mode,
            configuration_path=xml_path,
        )

    def loadConfiguration(self, configuration_path: str | Path) -> None:
        """
        Java-style alias, coherent with StatelessConstructorAdapterDialogic.
        """
        self.load_configuration(configuration_path)

    def load_configuration(self, configuration_path: str | Path) -> None:
        path = Path(configuration_path)

        self.adapter.loadConfiguration(path)
        self.configuration_path = path

    def query(
        self,
        prompt: str,
        timeout: int = 300,
        request_timeout: int = 15,
        retry_delay: int = 3,
    ) -> str:
        if not prompt.strip():
            raise ValueError("The prompt is empty.")

        return self.adapter.query(
            prompt.strip(),
            timeout=timeout,
            request_timeout=request_timeout,
            retry_delay=retry_delay,
        )

    def run(
        self,
        prompt: str,
        timeout: int = 300,
        request_timeout: int = 15,
        retry_delay: int = 3,
    ) -> StatelessClientRunResult:
        answer = self.query(
            prompt=prompt,
            timeout=timeout,
            request_timeout=request_timeout,
            retry_delay=retry_delay,
        )

        return StatelessClientRunResult(
            final_answer=answer,
            configuration_path=self.configuration_path,
        )


# Temporary alias if some local code still imports StatelessConstructorClient.
StatelessConstructorClient = ConstructorStatelessClient