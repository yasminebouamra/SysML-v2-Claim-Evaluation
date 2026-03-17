import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError
from pydantic import BaseModel, Field


# ===================== CONFIG =====================
INPUT_DIR = "original"
OUTPUT_PATH = "claims_output/claims_output.xlsx"
MODEL = "gpt-5.2"
RUNS = 3
MAX_RETRIES = 5
RETRY_BASE_SECONDS = 2.0
# ==================================================


class Claim(BaseModel):
    id: str
    entity: str
    property: str
    relation: str
    value: str | int | float | bool | list[float] | list[str] | None = None
    type: str | None = None
    unit: str | None = None
    modality: str
    condition: str | None = None
    original_string: str
    children: list[str] = Field(default_factory=list)
    parent: list[str] = Field(default_factory=list)


class ClaimResponse(BaseModel):
    claims: list[Claim]


FIELDS_DESCRIPTION = {
    "id": "A unique identifier for the extracted claim.",
    "entity": "The primary object, system component, or subject to which the requirement applies.",
    "property": "The specific characteristic, feature, or measurable attribute of the entity.",
    "relation": "The relational operator that expresses how the property relates to the value.",
    "value": "The numerical or categorical value associated with the property constraint.",
    "type": "The type of the entity when explicitly mentioned.",
    "unit": "The measurement unit corresponding to the value, if applicable.",
    "modality": "The strength or obligation level of the requirement.",
    "condition": "Additional context or scenarios under which the requirement applies.",
    "original_string": "The exact text statement from which the claim was extracted.",
    "children": "List of IDs for claims/components directly involved in fulfilling this claim.",
    "parent": "List of IDs for higher-level claims that this claim supports.",
}

SUPPORTED_EXTENSIONS = {".txt", ".sysml"}


def enforce_no_additional_properties(schema: dict[str, Any]) -> dict[str, Any]:
    def walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                node["additionalProperties"] = False
                properties = node.get("properties")
                if isinstance(properties, dict):
                    node["required"] = list(properties.keys())
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    updated = json.loads(json.dumps(schema))
    walk(updated)
    return updated


def build_prompts(original_dir: Path) -> tuple[str, str]:
    ex_nl = (original_dir / "simple_example_req_ground_truth.json").read_text(encoding="utf-8")
    req_nl = (original_dir / "simple_example.txt").read_text(encoding="utf-8")
    ex_sysml = (original_dir / "simple_example_sysml_ground_truth.json").read_text(encoding="utf-8")
    sysml = (original_dir / "simple_example.sysml").read_text(encoding="utf-8")

    system_prompt_req = f"""
You are an expert at extracting structured information from requirement text.
Extract claims according to this schema description:
{FIELDS_DESCRIPTION}
Return only valid JSON matching the schema.
Use None or [] when a field is not explicit.
Keep naming camelCase where applicable.

Example input:
{req_nl}
Example output:
{ex_nl}
"""

    system_prompt_sysml = f"""
You are an expert in SysML v2 and structured claim extraction.
Extract claims according to this schema description:
{FIELDS_DESCRIPTION}
Return only valid JSON matching the schema.
Use None or [] when a field is not explicit.
Keep naming camelCase where applicable.

Example input:
{sysml}
Example output:
{ex_sysml}
"""

    return system_prompt_req, system_prompt_sysml


def call_extraction_model(
    client: OpenAI,
    model: str,
    system_prompt: str,
    content: str,
    source_name: str,
) -> ClaimResponse:
    strict_schema = enforce_no_additional_properties(ClaimResponse.model_json_schema())

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"Source document: {source_name}\n\nExtract claims from:\n{content}",
                    },
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "claim_response",
                        "strict": True,
                        "schema": strict_schema,
                    },
                },
                timeout=120,
            )

            message = completion.choices[0].message
            if message.content is None:
                raise RuntimeError(f"No JSON content returned for {source_name}.")

            parsed = json.loads(message.content)
            return ClaimResponse.model_validate(parsed)

        except (APIConnectionError, APITimeoutError, RateLimitError, APIError) as error:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {error}")

            sleep_seconds = RETRY_BASE_SECONDS * (2 ** (attempt - 1))
            print(f"Retry {attempt}/{MAX_RETRIES} for {source_name} - waiting {sleep_seconds:.1f}s")
            time.sleep(sleep_seconds)


def normalize_cell(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def source_output_path(base_output: Path, source_file: Path, source_type: str) -> Path:
    return base_output.with_name(
        f"{base_output.stem}_{source_file.stem}_{source_type}{base_output.suffix}"
    )


def is_supported_input_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    input_dir = Path(INPUT_DIR)
    base_output = Path(OUTPUT_PATH)
    base_output.parent.mkdir(parents=True, exist_ok=True)

    system_prompt_req, system_prompt_sysml = build_prompts(input_dir)
    client = OpenAI(api_key=api_key)

    input_files = [p for p in sorted(input_dir.glob("*")) if is_supported_input_file(p)]

    for path in input_files:
        source_type = "NL" if path.suffix.lower() == ".txt" else "SYSML"
        system_prompt = system_prompt_req if source_type == "NL" else system_prompt_sysml

        print(f"\n=== Processing file: {path.name} ({source_type}) ===")

        rows_by_run: dict[int, list[dict[str, Any]]] = {}

        for run_number in range(1, RUNS + 1):
            print(f"Run {run_number}/{RUNS} for {path.name}")

            try:
                response = call_extraction_model(
                    client=client,
                    model=MODEL,
                    system_prompt=system_prompt,
                    content=path.read_text(encoding="utf-8"),
                    source_name=f"{path.name} (run_{run_number})",
                )

                rows = []
                for claim in response.claims:
                    data = claim.model_dump()

                    row = {
                        "source_file": path.name,
                        "source_type": source_type,
                        "run": f"run_{run_number}",
                    }

                    row.update({
                        k: normalize_cell(v)
                        for k, v in data.items()
                        if k != "original_string"
                    })

                    row["original_string"] = claim.original_string
                    rows.append(row)

                rows_by_run[run_number] = rows
                print(f"→ Extracted {len(rows)} claims")

            except Exception as e:
                print(f"Error on {path.name} run_{run_number}: {e}")
                rows_by_run[run_number] = []

        output_path = source_output_path(base_output, path, source_type)

        with pd.ExcelWriter(output_path) as writer:
            for run_number in range(1, RUNS + 1):
                df = pd.DataFrame(rows_by_run.get(run_number, []))

                if not df.empty and "original_string" in df.columns:
                    df = df.set_index("original_string")

                df.to_excel(writer, sheet_name=f"run_{run_number}")

        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()