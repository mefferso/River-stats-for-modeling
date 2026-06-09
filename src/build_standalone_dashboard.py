#!/usr/bin/env python3
"""Build a standalone forecast dashboard HTML with embedded JSON payload."""

from __future__ import annotations

import argparse
from pathlib import Path

FETCH_START = "    fetch('latest_forecasts.json')"
FETCH_END = "      });"


def replace_forecast_fetch_block(html: str, payload: str) -> str:
    """Replace the dashboard's runtime JSON fetch block with embedded JSON.

    The dashboard template can evolve as the front-end gains new fields.  The
    previous implementation matched one exact JavaScript block, which broke as
    soon as manual_models support was added.  Instead, find the fetch block by
    its stable start/end markers and replace the whole block.
    """

    start = html.find(FETCH_START)
    if start == -1:
        raise SystemExit("Could not find latest_forecasts.json fetch block in dashboard template")

    end = html.find(FETCH_END, start)
    if end == -1:
        raise SystemExit("Could not find end of latest_forecasts.json fetch block in dashboard template")
    end += len(FETCH_END)

    replacement = f"""const payload = {payload};
    forecasts = payload.forecasts || [];
    manualModels = payload.manual_models || {{}};
    document.getElementById('generated').textContent = `Generated UTC: ${{payload.generated_utc || 'unknown'}} · ${{forecasts.length}} LIDs`;
    populateManualChoices();
    render();"""

    return html[:start] + replacement + html[end:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed latest_forecasts.json into a standalone HTML dashboard.")
    parser.add_argument("--template", default="docs/index.html")
    parser.add_argument("--json", default="docs/latest_forecasts.json")
    parser.add_argument("--output", default="docs/forecast_dashboard.html")
    args = parser.parse_args()

    template = Path(args.template)
    json_path = Path(args.json)
    output = Path(args.output)

    html = template.read_text(encoding="utf-8")
    payload = json_path.read_text(encoding="utf-8")
    standalone_html = replace_forecast_fetch_block(html, payload)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(standalone_html, encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
