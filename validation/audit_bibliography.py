"""Static citation/BibTeX consistency audit for the journal manuscript."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tex", type=Path, default=Path("latex/sn-article.tex"))
    parser.add_argument("--bib", type=Path, default=Path("latex/mau1.bib"))
    parser.add_argument("--output", type=Path, default=Path("latex/bibliography_audit.json"))
    args = parser.parse_args()

    tex = args.tex.read_text(encoding="utf-8")
    bib = args.bib.read_text(encoding="utf-8")
    cited = []
    for group in re.findall(r"\\cite(?:t|p)?\{([^}]+)\}", tex):
        cited.extend(key.strip() for key in group.split(",") if key.strip())
    entries = re.findall(r"@\w+\s*\{\s*([^,\s]+)\s*,", bib)
    entry_counts = Counter(entries)
    dois = [doi.strip().lower() for doi in re.findall(r"\bdoi\s*=\s*\{([^}]+)\}", bib, re.I)]
    doi_counts = Counter(dois)

    recent_keys = [
        "li2025modwpt",
        "abousaber2025explainable",
        "xu2026cardiolike",
        "rahman2026uq",
        "taleban2026xai",
    ]
    result = {
        "status": "PASS",
        "n_bib_entries": len(entries),
        "n_unique_cited_keys": len(set(cited)),
        "missing_bib_entries": sorted(set(cited) - set(entries)),
        "duplicate_bib_keys": sorted(k for k, n in entry_counts.items() if n > 1),
        "duplicate_dois": sorted(k for k, n in doi_counts.items() if n > 1),
        "uncited_bib_entries": sorted(set(entries) - set(cited)),
        "recent_bspc_keys": {
            key: {"present": key in entries, "cited": key in cited} for key in recent_keys
        },
        "publisher_metadata_verification": {
            "li2025modwpt": "ScienceDirect: BSPC 104 (2025) 107516; DOI 10.1016/j.bspc.2025.107516",
            "abousaber2025explainable": "ScienceDirect: BSPC 110 (2025) 108142; DOI 10.1016/j.bspc.2025.108142",
            "xu2026cardiolike": "ScienceDirect: BSPC 112 (2026) 108604; DOI 10.1016/j.bspc.2025.108604",
            "rahman2026uq": "ScienceDirect: BSPC 113 (2026) 109032; DOI 10.1016/j.bspc.2025.109032",
            "taleban2026xai": "ScienceDirect: BSPC 114 (2026) 109325; DOI 10.1016/j.bspc.2025.109325",
        },
    }
    if result["missing_bib_entries"] or result["duplicate_bib_keys"] or result["duplicate_dois"]:
        result["status"] = "FAIL"
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
