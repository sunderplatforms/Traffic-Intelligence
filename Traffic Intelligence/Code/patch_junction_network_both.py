"""Patches BOTH occurrences of the GENERIC_JUNCTION_LABELS line in
junction_network.py (disambiguate_junction AND is_generic_label both
need the same bare-junction-number check for consistency)."""

filename = "junction_network.py"
target_substring = "or label.lower() in GENERIC_JUNCTION_LABELS"

with open(filename) as f:
    lines = f.readlines()

matches = [i for i, line in enumerate(lines) if target_substring in line]
print(f"Found {len(matches)} matching line(s) at: {matches}")

# Insert from the bottom up so earlier indices don't shift
inserted = 0
for idx in sorted(matches, reverse=True):
    if "isdigit" in lines[idx + 1]:
        print(f"  Line {idx} already patched — skipping")
        continue
    target_line = lines[idx]
    indent = target_line[:len(target_line) - len(target_line.lstrip())]
    new_line = f"{indent}or label.isdigit()\n"
    lines.insert(idx + 1, new_line)
    inserted += 1
    print(f"  Patched at line {idx}")

with open(filename, "w") as f:
    f.writelines(lines)

print(f"\nDone — inserted {inserted} new line(s) into {filename}")
