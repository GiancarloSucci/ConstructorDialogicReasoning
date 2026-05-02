# generate_ascii_endpoints_xml.py

from __future__ import annotations

import os
import re
import html
import requests


DEFAULT_API_URL = "https://training.constructor.app/api/platform-kmapi/v1"


def safe_identifier(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    value = value.strip("_")
    return value or "unknown"


def escape_xml(value: str) -> str:
    return html.escape(value, quote=True)


def main() -> None:
    api_url = os.getenv("CONSTRUCTOR_API_URL", DEFAULT_API_URL).rstrip("/")
    api_key = os.getenv("CONSTRUCTOR_API_KEY")

    if not api_key:
        raise RuntimeError("Missing CONSTRUCTOR_API_KEY")

    response = requests.get(
        f"{api_url}/language_models",
        headers={"X-KM-AccessKey": f"Bearer {api_key}"},
        timeout=30,
    )
    response.raise_for_status()

    models = response.json().get("results", [])

    lines: list[str] = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append('<endpointCatalog>')

    for model in models:
        alias = str(model.get("alias") or "").strip()
        name = str(model.get("name") or alias).strip()

        if not alias:
            continue

        safe_alias = safe_identifier(alias)

        for mode in ("direct", "model"):
            endpoint_id = f"{mode}_{safe_alias}"

            lines.append("    <endpoint")
            lines.append(f'        id="{escape_xml(endpoint_id)}"')
            lines.append(f'        llm_alias="{escape_xml(alias)}"')
            lines.append(f'        mode="{mode}"')
            lines.append('        role="review_and_improve"')
            lines.append('        timeout="900"')
            lines.append('        request_timeout="30"')
            lines.append('        retry_delay="5">')
            lines.append(
                f"        <description>Endpoint using {escape_xml(name)} "
                f"with alias {escape_xml(alias)} in {mode} mode.</description>"
            )
            lines.append("    </endpoint>")
            lines.append("")

    lines.append("</endpointCatalog>")

    output_path = "endpoint_catalog_ascii.xml"

    with open(output_path, "w", encoding="ascii", errors="xmlcharrefreplace") as f:
        f.write("\n".join(lines))

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
