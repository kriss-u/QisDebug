"""Set CITATION.cff's date-released to today (UTC) during a release build."""

import datetime
import pathlib
import re

CITATION = pathlib.Path(__file__).resolve().parents[2] / "CITATION.cff"


def main() -> None:
    """Rewrite the date-released line in place, failing loudly if it's missing."""
    today = datetime.datetime.now(datetime.timezone.utc).date().isoformat()
    text = CITATION.read_text(encoding="utf-8")
    updated, count = re.subn(
        r"^date-released: .*$", f'date-released: "{today}"', text, flags=re.MULTILINE
    )
    if count != 1:
        raise SystemExit(f"expected one date-released line in {CITATION}, found {count}")
    CITATION.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    main()
