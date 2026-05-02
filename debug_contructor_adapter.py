from __future__ import annotations

import os
import time
import requests
from constructor_adapter.constructor_adapter_base import ConstructorAdapter
from constructor_adapter import StatefulConstructorAdapter


API_URL = os.getenv(
    "CONSTRUCTOR_API_URL",
    "https://training.constructor.app/api/platform-kmapi/v1",
).rstrip("/")

API_KEY = os.getenv("CONSTRUCTOR_API_KEY")
KM_ID = os.getenv("CONSTRUCTOR_KM_ID") or os.getenv("KNOWLEDGE_MODEL_ID")
LLM_ALIAS = os.getenv("CONSTRUCTOR_LLM_ALIAS", "gemini-3-flash-preview")

def patch_constructor_adapter() -> None:

    original_check_if_km_exists = ConstructorAdapter.check_if_km_exists

    original_get_already_uploaded_files = ConstructorAdapter._get_already_uploaded_files

    def patched_check_if_km_exists(self) -> None:

        return None

    def patched_get_already_uploaded_files(self) -> None:

        if getattr(self, "mode", None) == "direct":

            self.uploaded_files = {}

            return None

        return original_get_already_uploaded_files(self)
    ConstructorAdapter._check_if_km_exists = patched_check_if_km_exists
    ConstructorAdapter._get_already_uploaded_files = patched_get_already_uploaded_files


def main() -> None:
    if not API_KEY:
        raise RuntimeError("Missing CONSTRUCTOR_API_KEY.")

    if not KM_ID:
        raise RuntimeError("Missing CONSTRUCTOR_KM_ID or KNOWLEDGE_MODEL_ID.")

    print("API_URL:", API_URL)
    print("KM_ID:", KM_ID)
    print("LLM_ALIAS:", LLM_ALIAS)
    patch_constructor_adapter()
    adapter = StatefulConstructorAdapter(
        mode="direct",
        api_url=API_URL,
        api_key=API_KEY,
        km_id=KM_ID,
        llm_alias=LLM_ALIAS,
    )

    print("Session ID:", adapter.session_id)

    question = "Reply with exactly one word: OK"
    print("Sending:", question)

    try:
        answer = adapter.query(
            question,
            timeout=180,
            request_timeout=30,
            retry_delay=5,
        )
        print("ANSWER FROM adapter.query():")
        print(answer)
    except Exception as exc:
        print("adapter.query() failed:")
        print(type(exc).__name__, exc)

        print("\nRaw session messages:")
        dump_messages(adapter)


def dump_messages(adapter) -> None:
    endpoint = (
        f"{adapter.api_url}/knowledge-models/"
        f"{adapter.km_id}/chat-sessions/"
        f"{adapter.session_id}/messages"
    )

    response = requests.get(
        endpoint,
        headers=adapter._get_headers(),
        timeout=30,
    )

    print("GET", endpoint)
    print("Status:", response.status_code)
    print("Body:", response.text[:4000])


if __name__ == "__main__":
    main()