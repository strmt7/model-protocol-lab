# Reproduction recipes

These are diagnostic recipes, not implemented fixes. They use synthetic inputs
and the dependency-free package surfaces. Run them in a disposable environment
against the audited SHA, then repeat against the proposed repair. Give the
script a new caller-selected scratch directory; do not use a private model or
account. The expected failing behavior below was observed on the audited source.

## Confirm the source before reproducing

```sh
git rev-parse HEAD
git status --porcelain=v1
```

The reference source is `d8a1346343e1b50b28bf567ab6b193b38fc79608`.
Documentation-only commits can have a newer HEAD with identical package blobs;
verify that difference instead of silently applying an old finding to new code.

## RR-02 — loadable malformed reports reach unsafe consumers

Save the following as a temporary diagnostic script outside tracked source and
run it with `python SCRIPT NEW_SCRATCH_DIRECTORY`. It writes only under the
new scratch directory. It runs the real CLI twice and reports only exit and
output-shape classifications, not absolute paths or tracebacks.

```python
import copy
import json
from pathlib import Path
import subprocess
import sys

scratch = Path(sys.argv[1])
scratch.mkdir(parents=True, exist_ok=False)


def report(**sections):
    return {
        "decision": "no_candidate",
        "selected_candidate": None,
        "search": [],
        "held_out": None,
        **sections,
    }


frontier = {
    "max_new_tokens": 2,
    "candidates": {
        "a": {"fixture_ids": ["f"], "pass_steps": [[1, 0], [2, 1]]}
    },
    "decision_steps": [[1, "no_candidate", None]],
    "budget_robust": True,
    "flip_ranges": [],
}
bad_frontier = copy.deepcopy(frontier)
bad_frontier["decision_steps"] = 1
floor = {
    "native_verdict": "held_out_failed",
    "max_new_tokens": 2,
    "safe_budget_floor": None,
    "budget_robust": False,
}
bad_floor = copy.deepcopy(floor)
bad_floor["budget_robust"] = []

cases = [
    ("frontier", report(budget_frontier=frontier),
     report(budget_frontier=bad_frontier)),
    ("floor", report(held_out_budget_floor=floor),
     report(held_out_budget_floor=bad_floor)),
]
for name, old, new in cases:
    paths = []
    for side, payload in (("old", old), ("new", new)):
        path = scratch / f"{name}-{side}.json"
        path.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
        paths.append(str(path))
    child = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "model_protocol_lab",
         "diff-reports", *paths],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    try:
        json.loads(child.stdout)
        valid_stdout_json = True
    except ValueError:
        valid_stdout_json = False
    print(json.dumps({
        "case": name,
        "exit": child.returncode,
        "traceback": "Traceback (most recent call last)" in child.stderr,
        "valid_stdout_json": valid_stdout_json,
    }))
```

Observed results: both exit 1 with a traceback. The frontier case has no valid
stdout JSON; the floor case has valid stdout JSON before the summary fails.
After repair, both must follow the chosen, explicit rejection or section-local
degradation contract, and a coherent unmodified pair must still compare.

A third regression should construct a valid precision section and replace
`unavailable_candidates` with a list containing both an integer and a string.
`_precision_recompute` currently calls `sorted` before checking element types.
Use a valid section and test both input directions; a random invalid payload
that is rejected earlier would not exercise this bug.

## RR-03 — producer-generated negation changes disappear

This synthetic backend implements an explicit two-number decision rule. It is
not a model and is not evidence of model competence. It lets the real probe
producer create a controlled effect without any external dependency.

```python
import json
import re

from model_protocol_lab import diffreports, experiments
from model_protocol_lab.core import Format, Generation


class ArithmeticBackend:
    fingerprint = "synthetic-negation-control"

    def __init__(self, drop_negation):
        self.drop_negation = drop_negation
        self.calls = 0

    def encode(self, text):
        return tuple(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(ids).decode("utf-8", "replace")

    def generate(self, input_ids, *, max_new_tokens, seed, cancel):
        self.calls += 1
        prompt = self.decode(input_ids)
        pairs = re.findall(r"\n([0-9]+)\n([0-9]+)", prompt)
        assert pairs
        a, b = map(int, pairs[-1])
        larger = "is smaller?" not in prompt
        if self.drop_negation and "not the smaller?" in prompt:
            larger = False
        answer = "first" if ((a > b) == larger) else "second"
        output = tuple((answer + chr(7)).encode("utf-8"))
        output += (0,) * max(0, max_new_tokens - len(output))
        return Generation(token_ids=output, reason="limit")


fmt = Format(
    name="synthetic", bos="", user_prefix="u:",
    assistant_prefix="a:", turn_end="</s>",
)
sections = []
for drop in (False, True):
    backend = ArithmeticBackend(drop)
    stage = experiments.negation_stage(
        backend, (fmt,), boundary=(7,), max_new_tokens=16,
        max_context=32768, seed=0,
    )
    section = experiments.negation_analysis(stage, (fmt.name,))
    assert section["available"] is True
    sections.append(section)
    print(json.dumps({
        "drop": drop,
        "calls": backend.calls,
        "headline": section["candidates"][fmt.name]["headline"]["decision"],
    }))


def wrap(section):
    return {
        "decision": "no_candidate", "selected_candidate": None,
        "search": [], "held_out": None, "negation": section,
    }


assert sections[0] != sections[1]
diff = diffreports.diff_reports(wrap(sections[0]), wrap(sections[1]))
print(json.dumps({
    "identical": diff["summary"]["identical"],
    "negation_compared": "negation" in diff,
}))
```

Observed: 64 generation calls per backend; the headline changes from
`below_threshold` to `negation_drop_observed`; the final comparison reports
`identical: true`, `negation_compared: false`. The repaired public CLI must
also expose the difference and missing-section asymmetry, not just a private
new helper.

## RR-04 — numeric equality accepts a boolean

```python
from model_protocol_lab.diffreports import _recomputed_equal

print(_recomputed_equal(False, 0.0))
print(_recomputed_equal({"rate": False}, {"rate": 0.0}))
print(_recomputed_equal(True, 1))
```

Observed: `True`, `True`, `False`. The first two violate the intended JSON
numeric-versus-boolean distinction. The full regression must use a coherent
admitted precision artifact as well as this primitive, because some validators
already guard their fields before reaching the helper.

## RR-07 and RR-08 — setup and platform checks

On Windows, use the repository-pinned toolchain in a new environment:

```sh
uv pip install --dry-run --python PYTHON --require-hashes \
  -r tools/requirements/cocoindex.txt
uv pip install --dry-run --python PYTHON --require-hashes \
  -r tools/requirements/dev-hf.txt
python -m mypy model_protocol_lab
python -m pytest --collect-only -q -p no:cacheprovider
```

`PYTHON` is the caller's explicit environment interpreter, not a hardcoded
installation path. Both lock resolutions fail on the unmarked Linux-only
`nvidia-cufile` requirement. Pinned mypy 2.4.1 reports 34 errors in 3 source
files on Windows. A dev-only environment produces 15 test collection errors;
its partial count is not the HF-enabled full-suite count.

After repair, execute actual installs plus `pip check` and layer-specific tests
on each supported target. A resolver dry run alone does not prove an installed
adapter works.

## RR-01 — prove scanner sensitivity

Use Gitleaks **8.30.1** downloaded from its official release and verified against
that release's checksum manifest. In a new scratch directory, create one
never-issued provider-shaped canary locally; do not commit or publish its value.
Scan the same input once with `.gitleaks.toml` and once with a separate config
containing the following:

```toml
[extend]
useDefault = true
```

```sh
gitleaks dir NEW_CANARY_DIRECTORY --config .gitleaks.toml --redact
gitleaks dir NEW_CANARY_DIRECTORY --config DEFAULT_CONFIG --redact
```

Observed: repository config exits 0 with no detection; default-rule control
exits 1 and detects the canary. The exact same binary/input pairing is important:
a random canary that no upstream rule recognizes would prove nothing. Delete
the scratch token and retain only redacted classifications. The final repair
must prove the configured **history** invocation is sensitive too.

## RR-05 and RR-06 — read-only control-plane checks

```sh
gh api repos/strmt7/model-protocol-lab/branches/main/protection
gh run view 42835070138 --repo strmt7/model-protocol-lab --log-failed
```

The audited protection record has no required status checks, and the exact-head
findings guard fails on missing DeepSource access. Inspect `release.yml`'s
`publish.needs` and the build steps: no full test/security receipt is a required
input. Do not push a tag to reproduce that design gap. A safe regression uses a
local policy verifier or a no-publication workflow with mocked negative inputs.

## What these recipes do not establish

They are minimal reproductions of particular defects, not the complete release
gate. They do not prove every malformed report is safe, every scientific probe
is correct, a real checkpoint can succeed, or a wheel is independent of the
checkout. [PLAN.md](PLAN.md) specifies the broader acceptance work.
