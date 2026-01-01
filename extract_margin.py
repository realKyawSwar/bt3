#!/usr/bin/env python
import re

with open(r'.venv/Lib/site-packages/backtesting/backtesting.py', 'r') as f:
    lines = f.readlines()

# Find the line numbers with "margin"
margin_lines = []
for i, line in enumerate(lines):
    if 'margin' in line.lower() or 'leverage' in line.lower():
        margin_lines.append(i)

# Show key sections
print("=== KEY MARGIN INITIALIZATION ===")
for line_num in range(560, 580):  # Broker.__init__ area
    print(f"{line_num+1}: {lines[line_num]}", end="")

print("\n=== MARGIN AVAILABLE PROPERTY ===")
for line_num in range(735, 765):
    if line_num < len(lines):
        print(f"{line_num+1}: {lines[line_num]}", end="")

print("\n=== ORDER SIZE CALCULATION ===")
for line_num in range(870, 910):
    if line_num < len(lines):
        print(f"{line_num+1}: {lines[line_num]}", end="")
