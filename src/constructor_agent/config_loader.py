from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from constructor_agent.domain import AgentPathConfig, EndpointSpec, Mode


class XmlPathConfigLoader:
    """Loads the ordered LLM path from XML."""

    def load(self, xml_file: str | Path) -> AgentPathConfig:
        path = Path(xml_file)
        if not path.exists():
            raise FileNotFoundError(f"XML configuration file not found: {path}")

        root = ET.parse(path).getroot()
        if root.tag != "agentPath":
            raise ValueError("The root element must be <agentPath>.")

        endpoints: list[EndpointSpec] = []
        for node in root.findall("endpoint"):
            endpoint_id = self._required_attr(node, "id")
            llm_alias = node.attrib.get("llm_alias") or node.attrib.get("model")
            if not llm_alias:
                raise ValueError(f"Endpoint '{endpoint_id}' must define llm_alias or model.")

            mode = node.attrib.get("mode", "direct")
            if mode not in {"direct", "model"}:
                raise ValueError(
                    f"Endpoint '{endpoint_id}' has invalid mode '{mode}'. Use 'direct' or 'model'."
                )

            description_node = node.find("description")
            description = ""
            if description_node is not None and description_node.text:
                description = description_node.text.strip()

            endpoints.append(
                EndpointSpec(
                    id=endpoint_id,
                    llm_alias=llm_alias,
                    role=node.attrib.get("role", "review_and_improve"),
                    mode=mode,  # type: ignore[arg-type]
                    description=description,
                    timeout=int(node.attrib.get("timeout", "300")),
                    request_timeout=int(node.attrib.get("request_timeout", "15")),
                    retry_delay=int(node.attrib.get("retry_delay", "3")),
                )
            )

        config = AgentPathConfig(
            name=root.attrib.get("name", "constructor-agent-path"),
            endpoints=tuple(endpoints),
        )
        config.validate()
        return config

    @staticmethod
    def _required_attr(node: ET.Element, name: str) -> str:
        value = node.attrib.get(name)
        if not value:
            raise ValueError(f"Element <{node.tag}> must define attribute '{name}'.")
        return value
