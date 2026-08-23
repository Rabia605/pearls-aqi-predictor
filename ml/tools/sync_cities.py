"""Generate the web app's city list from cities.json.

`cities.json` at the project root is the single source of truth. The web app is
built from `web/` (that's Vercel's root directory), so it can't reliably import a
file above that folder — instead we generate a typed TypeScript module inside
`web/` and commit it. Run this whenever cities.json changes:

    python -m ml.tools.sync_cities
"""
from __future__ import annotations

import json

from ml.common.cities import CITIES
from ml.common.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "web" / "src" / "lib" / "cities.generated.ts"

HEADER = """// GENERATED FILE — do not edit by hand.
// Source of truth: /cities.json · regenerate with `python -m ml.tools.sync_cities`

export type City = {
  id: string;
  name: string;
  country: string;
  timezone: string;
  featured: boolean;
};

export const CITIES: City[] = """


def main() -> None:
    payload = [
        {
            "id": c.id,
            "name": c.name,
            "country": c.country,
            "timezone": c.timezone,
            "featured": c.featured,
        }
        for c in CITIES
    ]
    body = json.dumps(payload, indent=2)
    OUT.write_text(f"{HEADER}{body};\n", encoding="utf-8")
    print(f"  Wrote {len(payload)} cities -> {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
