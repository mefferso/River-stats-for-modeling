#!/usr/bin/env python3
"""Build a standalone forecast dashboard HTML with embedded JSON payload."""

from __future__ import annotations

import argparse
from pathlib import Path

FETCH_BLOCK = """fetch('latest_forecasts.json')
      .then(r => r.json())
      .then(payload => {
        forecasts = payload.forecasts || [];
        manualModels = payload.manual_models || {};
        document.getElementById('generated').textContent = `Generated UTC: ${payload.generated_utc || 'unknown'} · ${forecasts.length} LIDs`;
        populateManualChoices();
        render();
      })
      .catch(err => {
        document.getElementById('generated').textContent = 'No latest_forecasts.json found yet. Run the forecast workflow first.';
        console.error(err);
      });"""


def replacement_block(payload: str) -> str:
    """Return JavaScript that uses an embedded latest_forecasts payload."""
    return f"""const payload = {payload};
    forecasts = payload.forecasts || [];
    manualModels = payload.manual_models || {{}};
    document.getElementById('generated').textContent = `Generated UTC: ${{payload.generated_utc || 'unknown'}} · ${{forecasts.length}} LIDs`;
    populateManualChoices();
    render();"""


def replace_forecast_fetch_block(html: str, payload: str) -> str:
    """Replace the dashboard JSON fetch block with embedded JSON.

    Prefer exact replacement when the template still matches FETCH_BLOCK.
    If the template changes spacing or adjacent dashboard logic, fall back to
    replacing from the latest_forecasts.json fetch call through its catch block.
    """
    replacement = replacement_block(payload)
    if FETCH_BLOCK in html:
        return html.replace(FETCH_BLOCK, replacement, 1)

    fetch_idx = html.find("fetch('latest_forecasts.json')")
    if fetch_idx == -1:
        fetch_idx = html.find('fetch("latest_forecasts.json")')
    if fetch_idx == -1:
        raise RuntimeError("Could not find latest_forecasts.json fetch call in dashboard template")

    # Preserve the indentation before the fetch call.
    line_start = html.rfind("\n", 0, fetch_idx) + 1
    start = line_start

    script_idx = html.find("</script>", fetch_idx)
    if script_idx == -1:
        raise RuntimeError("Could not find closing script tag after forecast fetch block")

    # The forecast fetch block is the last executable block before </script>.
    end = html.rfind(";", fetch_idx, script_idx)
    if end == -1:
        raise RuntimeError("Could not find end of forecast fetch block")
    end += 1

    indent = html[start:fetch_idx]
    indented_replacement = "\n".join(
        (indent + line if line else line) for line in replacement.splitlines()
    )
    return html[:start] + indented_replacement + html[end:]


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
