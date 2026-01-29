import sys
with open('main.tex', 'r') as f:
    lines = f.readlines()
with open('baseline_table.tex', 'r') as f:
    baseline_lines = f.readlines()
# find line index of \end{table} (should be line 315 zero-indexed?)
for i, line in enumerate(lines):
    if line.strip() == r'\end{table}':
        end_table_idx = i
        break
# find next non-empty line after that
j = end_table_idx + 1
while j < len(lines) and lines[j].strip() == '':
    j += 1
# j points to line with \subsection{Hyper‑parameter Settings}
# insert after end_table_idx, before j
# we want to keep empty line after table, then our content, then empty line before subsection
new_lines = lines[:end_table_idx+1]  # include \end{table}
new_lines.append('\n')  # keep empty line
new_lines.extend(baseline_lines)
new_lines.append('\n')  # empty line before subsection
new_lines.extend(lines[j:])  # rest of file
with open('main_new.tex', 'w') as f:
    f.writelines(new_lines)
print('Inserted baseline table into main_new.tex')
