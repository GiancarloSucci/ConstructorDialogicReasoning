#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from constructor_agent.domain import QuestionSpec
from constructor_agent.platform_config import ConstructorPlatformConfig
from constructor_agent.stateful_constructor_client import StatefulConstructorClient


@dataclass(frozen=True)
class EndpointCandidate:
    endpoint_id: str
    llm_alias: str
    mode: str


@dataclass(frozen=True)
class LlmCheckResult:
    endpoint_id: str
    llm_alias: str
    mode: str
    usable: bool
    elapsed_seconds: float
    answer_preview: str
    error_type: str
    error_message: str


def parse_endpoint_catalog(xml_path: Path) -> list[EndpointCandidate]:
    if not xml_path.exists():
        raise FileNotFoundError(f"XML file not found: {xml_path}")

    root = ET.parse(xml_path).getroot()

    endpoints: list[EndpointCandidate] = []

    for node in root.findall(".//endpoint"):
        endpoint_id = node.attrib.get("id", "").strip()
        llm_alias = node.attrib.get("llm_alias", "").strip()
        mode = node.attrib.get("mode", "direct").strip()

        if not endpoint_id or not llm_alias:
            continue

        if mode not in {"direct", "model"}:
            continue

        endpoints.append(
            EndpointCandidate(
                endpoint_id=endpoint_id,
                llm_alias=llm_alias,
                mode=mode,
            )
        )

    return endpoints


def filter_candidates(
    candidates: list[EndpointCandidate],
    mode: str,
    unique_aliases: bool,
) -> list[EndpointCandidate]:
    if mode != "all":
        candidates = [candidate for candidate in candidates if candidate.mode == mode]

    if not unique_aliases:
        return candidates

    seen: set[tuple[str, str]] = set()
    result: list[EndpointCandidate] = []

    for candidate in candidates:
        key = (candidate.llm_alias, candidate.mode)

        if key in seen:
            continue

        seen.add(key)
        result.append(candidate)

    return result


def check_candidate(
    client: StatefulConstructorClient,
    candidate: EndpointCandidate,
    timeout: int,
    request_timeout: int,
    retry_delay: int,
    prompt: str,
    verbose: bool,
) -> LlmCheckResult:
    question = QuestionSpec(
        id=candidate.endpoint_id,
        llm_alias=candidate.llm_alias,
        mode=candidate.mode,  # type: ignore[arg-type]
        role="availability_test",
        description="Short LLM availability test.",
        prompt=None,
        timeout=timeout,
        request_timeout=request_timeout,
        retry_delay=retry_delay,
    )

    start = time.time()

    try:
        if verbose:
            print(
                f"[CHECK] {candidate.endpoint_id} | "
                f"{candidate.mode} | {candidate.llm_alias}",
                flush=True,
            )

        answer = client.ask(question, prompt)
        elapsed = time.time() - start

        answer_preview = str(answer).replace("\n", " ")[:300]

        return LlmCheckResult(
            endpoint_id=candidate.endpoint_id,
            llm_alias=candidate.llm_alias,
            mode=candidate.mode,
            usable=True,
            elapsed_seconds=round(elapsed, 3),
            answer_preview=answer_preview,
            error_type="",
            error_message="",
        )

    except Exception as exc:
        elapsed = time.time() - start

        if verbose:
            print(
                f"[FAIL] {candidate.endpoint_id} | "
                f"{candidate.mode} | {candidate.llm_alias} | "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )

        return LlmCheckResult(
            endpoint_id=candidate.endpoint_id,
            llm_alias=candidate.llm_alias,
            mode=candidate.mode,
            usable=False,
            elapsed_seconds=round(elapsed, 3),
            answer_preview="",
            error_type=type(exc).__name__,
            error_message=str(exc).replace("\n", " ")[:1000],
        )


def write_outputs(results: list[LlmCheckResult], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    usable = [result for result in results if result.usable]
    not_usable = [result for result in results if not result.usable]

    with (output_dir / "llm_alias_check_results.json").open("w", encoding="utf-8") as f:
        json.dump([asdict(result) for result in results], f, indent=2)

    with (output_dir / "llm_alias_check_results.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "endpoint_id",
                "llm_alias",
                "mode",
                "usable",
                "elapsed_seconds",
                "answer_preview",
                "error_type",
                "error_message",
            ],
        )
        writer.writeheader()

        for result in results:
            writer.writerow(asdict(result))

    with (output_dir / "usable_llm_aliases.txt").open("w", encoding="utf-8") as f:
        for result in usable:
            f.write(f"{result.mode}\t{result.llm_alias}\t{result.endpoint_id}\n")

    with (output_dir / "not_working_llm_aliases.txt").open("w", encoding="utf-8") as f:
        for result in not_usable:
            f.write(
                f"{result.mode}\t{result.llm_alias}\t{result.endpoint_id}\t"
                f"{result.error_type}\t{result.error_message}\n"
            )


def print_summary(results: list[LlmCheckResult]) -> None:
    usable = [result for result in results if result.usable]
    not_usable = [result for result in results if not result.usable]

    print()
    print("Summary")
    print("=======")
    print(f"Total tested: {len(results)}")
    print(f"Usable:       {len(usable)}")
    print(f"Not working:  {len(not_usable)}")
    print()

    print("Usable llm_alias values")
    print("-----------------------")
    for result in usable:
        print(f"{result.mode}\t{result.llm_alias}\t{result.endpoint_id}")

    print()
    print("Not working llm_alias values")
    print("----------------------------")
    for result in not_usable:
        print(
            f"{result.mode}\t{result.llm_alias}\t{result.endpoint_id}\t"
            f"{result.error_type}: {result.error_message}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check which ConstructorPlatform llm_alias values work, "
            "starting from an old endpointCatalog XML file."
        )
    )

    parser.add_argument(
        "--catalog",
        required=True,
        type=Path,
        help="Path to the old endpointCatalog XML file.",
    )
    parser.add_argument(
        "--mode",
        choices=["direct", "model", "all"],
        default="direct",
        help="Which endpoint modes to test. Default: direct.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("llm_alias_check_output"),
        help="Directory where result files are written.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="Maximum wait per model in seconds. Default: 90.",
    )
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=20,
        help="HTTP request timeout for polling in seconds. Default: 20.",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=3,
        help="Delay between polling attempts in seconds. Default: 3.",
    )
    parser.add_argument(
        "--prompt",
        default="Answer exactly with the single word OK.",
        help="Short prompt used to test each model.",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="ConstructorPlatform API URL. If omitted, uses environment/default.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="ConstructorPlatform API key. If omitted, uses CONSTRUCTOR_API_KEY.",
    )
    parser.add_argument(
        "--km-id",
        default=None,
        help="Knowledge model id. If omitted, uses CONSTRUCTOR_KM_ID or KNOWLEDGE_MODEL_ID.",
    )
    parser.add_argument(
        "--all-endpoints",
        action="store_true",
        help=(
            "Test every endpoint entry. By default, repeated "
            "(llm_alias, mode) pairs are tested only once."
        ),
    )
    parser.add_argument(
        "--stop-after",
        type=int,
        default=None,
        help="Stop after N checks. Useful for quick testing.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress for each model.",
    )

    args = parser.parse_args()

    try:
        candidates = parse_endpoint_catalog(args.catalog)
        candidates = filter_candidates(
            candidates=candidates,
            mode=args.mode,
            unique_aliases=not args.all_endpoints,
        )

        if args.stop_after is not None:
            candidates = candidates[: args.stop_after]

        if not candidates:
            print("No endpoint candidates found.", file=sys.stderr)
            return 1

        config = ConstructorPlatformConfig(
            api_url=args.api_url,
            api_key=args.api_key,
            km_id=args.km_id,
        )

        client = StatefulConstructorClient(config)

        results: list[LlmCheckResult] = []

        for index, candidate in enumerate(candidates, start=1):
            print(
                f"[{index}/{len(candidates)}] "
                f"testing {candidate.mode} {candidate.llm_alias}",
                flush=True,
            )

            result = check_candidate(
                client=client,
                candidate=candidate,
                timeout=args.timeout,
                request_timeout=args.request_timeout,
                retry_delay=args.retry_delay,
                prompt=args.prompt,
                verbose=args.verbose,
            )

            results.append(result)

            status = "OK" if result.usable else "FAIL"
            print(
                f"    {status} in {result.elapsed_seconds}s: "
                f"{candidate.endpoint_id}",
                flush=True,
            )

        write_outputs(results, args.output_dir)
        print_summary(results)

        print()
        print(f"Files written in: {args.output_dir}")
        print(f"- {args.output_dir / 'usable_llm_aliases.txt'}")
        print(f"- {args.output_dir / 'not_working_llm_aliases.txt'}")
        print(f"- {args.output_dir / 'llm_alias_check_results.csv'}")
        print(f"- {args.output_dir / 'llm_alias_check_results.json'}")

        return 0

    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())