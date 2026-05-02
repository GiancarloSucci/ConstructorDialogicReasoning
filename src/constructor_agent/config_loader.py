from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from constructor_agent.domain import QuestionPathConfig, QuestionSpec


class XmlQuestionConfigLoader:
    """Loads the ordered question path from XML."""

    def load(self, xml_file: str | Path) -> QuestionPathConfig:
        path = Path(xml_file)

        if not path.exists():
            raise FileNotFoundError(f"XML configuration file not found: {path}")

        root = ET.parse(path).getroot()

        if root.tag != "questionPath":
            raise ValueError("The root element must be <questionPath>.")

        questions: list[QuestionSpec] = []

        for node in root.findall("question"):
            questions.append(self._read_question_node(node))

        config = QuestionPathConfig(
            name=root.attrib.get("name", "constructor-question-path"),
            questions=tuple(questions),
        )

        config.validate()
        return config

    def _read_question_node(self, node: ET.Element) -> QuestionSpec:
        question_id = self._required_attr(node, "id")
        llm_alias = node.attrib.get("llm_alias") or node.attrib.get("model")

        if not llm_alias:
            raise ValueError(f"Question '{question_id}' must define llm_alias or model.")

        mode = node.attrib.get("mode", "direct")

        if mode not in {"direct", "model"}:
            raise ValueError(
                f"Question '{question_id}' has invalid mode '{mode}'. "
                "Use 'direct' or 'model'."
            )

        description_node = node.find("description")
        description = ""

        if description_node is not None and description_node.text:
            description = description_node.text.strip()

        prompt_node = node.find("prompt")
        prompt = None

        if prompt_node is not None and prompt_node.text:
            prompt = prompt_node.text.strip()

        return QuestionSpec(
            id=question_id,
            llm_alias=llm_alias,
            role=node.attrib.get("role", "review_and_improve"),
            mode=mode,  # type: ignore[arg-type]
            description=description,
            prompt=prompt,
            timeout=int(node.attrib.get("timeout", "300")),
            request_timeout=int(node.attrib.get("request_timeout", "15")),
            retry_delay=int(node.attrib.get("retry_delay", "3")),
        )

    @staticmethod
    def _required_attr(node: ET.Element, name: str) -> str:
        value = node.attrib.get(name)

        if not value:
            raise ValueError(f"Element <{node.tag}> must define attribute '{name}'.")

        return value