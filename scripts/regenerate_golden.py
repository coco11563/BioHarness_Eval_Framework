"""Regenerate every golden CSV from the shipped run snapshot.

1. Legacy stored-score goldens (``golden/headline.csv`` and
   ``golden/headline_per_dataset/*.csv``), emitted exactly as
   ``verify.py --protocol legacy-rouge`` derives them.
2. Headline continuous-v2 goldens, via ``scripts/aggregate_continuous_v2.py``.

Run after any change to ``output/pipeline/*.jsonl``, then update the sha256
values in MANIFEST.toml.

    python scripts/regenerate_golden.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "src"))
sys.path.insert(0, str(HERE / "scripts"))

import aggregate_continuous_v2  # noqa: E402

import verify  # noqa: E402
from framework_eval.eval import (  # noqa: E402
    aggregate,
    aggregate_per_type,
    emit_headline_csv,
    emit_per_type_csv,
)


def main() -> int:
    manifest = verify.tomllib.loads(verify.MANIFEST_PATH.read_text(encoding="utf-8"))
    method = manifest["run"]["method_id"]
    by_cfg = verify._load_run_results(HERE / manifest["run"]["run_dir_relative_path"])

    rows = [aggregate(method, cfg, results) for cfg, results in by_cfg]
    rows.append(aggregate(method, "_overall", [r for _, rs in by_cfg for r in rs]))
    (HERE / "golden" / "headline.csv").write_text(emit_headline_csv(rows), encoding="utf-8")
    per_dataset = HERE / "golden" / "headline_per_dataset"
    for cfg, results in by_cfg:
        text = emit_per_type_csv(aggregate_per_type(method, cfg, [], results))
        (per_dataset / f"{cfg}.csv").write_text(text, encoding="utf-8")
    print(
        f"legacy stored-score goldens: _overall binary={rows[-1].binary_accuracy:.6f} "
        f"continuous={rows[-1].continuous_mean:.6f}"
    )

    sys.argv = [sys.argv[0]]
    return aggregate_continuous_v2.main()


if __name__ == "__main__":
    raise SystemExit(main())
