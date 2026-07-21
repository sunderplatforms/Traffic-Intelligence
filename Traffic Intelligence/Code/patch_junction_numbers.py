"""One-off patch: extend disambiguate_junction() in all three junction
scripts to also treat bare junction numbers (e.g. "5", "6") as generic,
qualifying them by road name — same logic already applied to road names
and "LA Boundary"."""

import re

files_and_indent = {
    "junction_network.py": "",
    "junction_predictor.py": "",
    "junction_cli.py": "    ",  # this one's function is indented inside main()
}

old_snippet = '''is_generic = (
{indent}    canonical_road_name(label) in known_road_names_canonical
{indent}    or label.lower() in GENERIC_JUNCTION_LABELS
{indent})'''

new_snippet = '''is_generic = (
{indent}    canonical_road_name(label) in known_road_names_canonical
{indent}    or label.lower() in GENERIC_JUNCTION_LABELS
{indent}    or label.isdigit()
{indent})'''

for filename, indent in files_and_indent.items():
    with open(filename) as f:
        content = f.read()

    old = old_snippet.format(indent=indent)
    new = new_snippet.format(indent=indent)

    count = content.count(old)
    if count != 1:
        print(f"WARNING: {filename} — expected 1 match, found {count}. Skipping — check manually.")
        continue

    content = content.replace(old, new)
    with open(filename, "w") as f:
        f.write(content)
    print(f"Patched {filename}")
