#!/usr/bin/env python3
"""Build a standalone forecast dashboard HTML with embedded JSON payload."""

from __future__ import annotations

import argparse
from pathlib import Path

FETCH_BLOCK = """fetch('latest_forecasts.json')
      .then(r => r.json())
      .then(payload => {
        forecasts = payload.forecasts || [];
        document.getElementById('generated').textContent = `Generated UTC: ${payload.generated_utc || 'unknown'} · ${forecasts.length} LIDs`;
        render();
      })
      .catch(err => {
        document.getElementById('generated').textContent = 'No latest_forecasts.json found yet. Run the forecast workflow first.';
        console.error(err);
      });"""


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
    replacement = f"""const payload = {payload};
    forecasts = payload.forecasts || [];
    document.getElementById('generated').textContent = `Generated UTC: ${{payload.generated_utc || 'unknown'}} · ${{forecasts.length}} LIDs`;
    render();"""

    if FETCH_BLOCK not in html:
        raise SystemExit("Could not find forecast fetch block in dashboard template")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html.replace(FETCH_BLOCK, replacement), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
