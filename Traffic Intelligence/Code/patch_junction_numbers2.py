"""One-off patch v2: finds the GENERIC_JUNCTION_LABELS line by content
(robust to whitespace differences) and inserts the bare-junction-number
check right after it, matching that line's own indentation."""

files = ["junction_network.py", "junction_predictor.py", "junction_cli.py"]
target_substring = "or label.lower() in GENERIC_JUNCTION_LABELS"

for filename in files:
    with open(filename) as f:
        lines = f.readlines()

    matches = [i for i, line in enumerate(lines) if target_substring in line]

    if len(matches) != 1:
        print(f"WARNING: {filename} — expected 1 matching line, found {len(matches)}. Skipping.")
        continue

    idx = matches[0]
    target_line = lines[idx]

    if "isdigit" in lines[idx + 1]:
        print(f"SKIPPED {filename} — already patched.")
        continue

    indent = target_line[:len(target_line) - len(target_line.lstrip())]
    new_line = f"{indent}or label.isdigit()\n"

    lines.insert(idx + 1, new_line)

    with open(filename, "w") as f:
        f.writelines(lines)

    print(f"Patched {filename} at line {idx + 1}")
