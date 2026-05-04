"""Re-parse existing response JSON files where reasoning or parsed_answer is
ERROR_PARSING, using the improved parse_response logic.

Scans all results_* subdirectories under a given root, updates files in-place,
and prints a summary of what was fixed.
"""

import json
import re
import sys
from pathlib import Path


# ── Parsing logic (mirrors collect_local_responses.parse_response) ────────────

def parse_response(raw_output: str) -> tuple[str, str]:
    # Normalise ```tag_name>...``` into proper <tag_name> (LLaMA/Qwen quirk)
    text = re.sub(r"```([a-zA-Z_]+)>", r"<\1>", raw_output)
    text = re.sub(r"```[a-zA-Z_]*\n?", "", text)
    text = re.sub(r"\n?```", "", text)

    reasoning = "ERROR_PARSING"
    decision = "ERROR_PARSING"

    # Reasoning: primary XML tags, fallback to **Clinical Reasoning:** markdown
    m = re.search(r"<clinical_reasoning>(.*?)</clinical_reasoning>", text, re.DOTALL | re.IGNORECASE)
    if m:
        reasoning = m.group(1).strip()
    else:
        m = re.search(
            r"\*{0,2}[Cc]linical\s+[Rr]easoning[^:\n]*:?\*{0,2}\s*\n(.*?)(?=\n\s*\*{0,2}[Dd]ecision|\Z)",
            text, re.DOTALL,
        )
        if m:
            reasoning = m.group(1).strip()

    # Decision: primary XML tags, then progressively looser fallbacks
    m = re.search(r"<decision>(.*?)</decision>", text, re.DOTALL | re.IGNORECASE)
    if m:
        # Strip markdown bold (**Admit** → admit) before checking
        inner = re.sub(r"\*+", "", m.group(1).strip()).strip().lower()
        if inner in ("admit", "discharge"):
            decision = inner
    if decision == "ERROR_PARSING":
        # Unclosed <decision> tag — grab first admit/discharge after it
        m = re.search(r"<decision>(.*)", text, re.DOTALL | re.IGNORECASE)
        if m:
            m2 = re.search(r"\b(admit|discharge)\b", m.group(1), re.IGNORECASE)
            if m2:
                decision = m2.group(1).lower()
    if decision == "ERROR_PARSING":
        # Standalone **Admit** / **Discharge** anywhere in text
        m = re.search(r"\*\*(admit|discharge)\*\*", text, re.IGNORECASE)
        if m:
            decision = m.group(1).lower()
    if decision == "ERROR_PARSING":
        m = re.search(r"<(admit|discharge)>", text, re.IGNORECASE)
        if m:
            decision = m.group(1).lower()
    if decision == "ERROR_PARSING":
        m = re.search(r"\(decision\)\s*\n?\s*(admit|discharge)", text, re.IGNORECASE)
        if m:
            decision = m.group(1).lower()
    if decision == "ERROR_PARSING":
        m = re.search(
            r"\*{0,2}[Dd]ecision[^:\n]*:?\*{0,2}\s*\n?\s*<?([Aa]dmit|[Dd]ischarge)>?",
            text,
        )
        if m:
            decision = m.group(1).lower()
    if decision == "ERROR_PARSING":
        last_lines = "\n".join(text.strip().splitlines()[-5:])
        matches = re.findall(r"\b(admit|discharge)\b", last_lines, re.IGNORECASE)
        if matches:
            decision = matches[-1].lower()

    return reasoning, decision


# ── Main ──────────────────────────────────────────────────────────────────────

ROOT = Path("../15_Apr/vignettes_nature_paper")

SCAN_DIRS = [
    ROOT / "results_llama32_cfd",
    ROOT / "results_qwen25_cfd",
]

total = fixed_reasoning = fixed_decision = still_broken = 0
scanned = 0
LOG_EVERY = 1000

import time
start = time.time()

for scan_dir in SCAN_DIRS:
    all_files = sorted(scan_dir.rglob("response_n=*.json"))
    print(f"\nScanning {scan_dir.name}: {len(all_files)} files...")

    for json_path in all_files:
        scanned += 1
        if scanned % LOG_EVERY == 0:
            elapsed = time.time() - start
            print(f"  [{scanned}] {elapsed/60:.1f}min elapsed | "
                  f"found {total} broken | fixed {fixed_decision} decisions, "
                  f"{fixed_reasoning} reasoning | still broken {still_broken}", flush=True)

        try:
            with open(json_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        reasoning = data.get("reasoning", "")
        decision = data.get("parsed_answer", "")

        needs_reasoning = reasoning == "ERROR_PARSING"
        needs_decision = decision == "ERROR_PARSING"

        if not needs_reasoning and not needs_decision:
            continue

        total += 1
        raw = data.get("raw_response", "")
        if not raw:
            still_broken += 1
            continue

        new_reasoning, new_decision = parse_response(raw)

        updated = False

        if needs_reasoning and new_reasoning != "ERROR_PARSING":
            data["reasoning"] = new_reasoning
            fixed_reasoning += 1
            updated = True

        if needs_decision and new_decision != "ERROR_PARSING":
            data["parsed_answer"] = new_decision
            fixed_decision += 1
            updated = True

        if updated:
            with open(json_path, "w") as f:
                json.dump(data, f, indent=4)
            print(f"  FIXED {json_path.parent.parent.name}/{json_path.parent.name}/{json_path.name}", flush=True)
        else:
            print(f"  STILL BROKEN: {json_path.parent.parent.name}/{json_path.parent.name}/{json_path.name}", flush=True)
            still_broken += 1

elapsed = time.time() - start
print(f"\nDone in {elapsed/60:.1f}min. Scanned {scanned} files.")
print(f"Files with ERROR_PARSING found : {total}")
print(f"  reasoning fixed              : {fixed_reasoning}")
print(f"  parsed_answer fixed          : {fixed_decision}")
print(f"  still ERROR_PARSING after    : {still_broken}")
