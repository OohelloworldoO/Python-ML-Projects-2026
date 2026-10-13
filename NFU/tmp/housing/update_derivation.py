from pathlib import Path
import json
base = Path('Logistic_Regression')
p = base / 'python.ipynb'
nb = json.loads(p.read_text(encoding='utf-8'))
text = (base / 'handwritten_derivation.md').read_text(encoding='utf-8-sig')
matched = 0
for cell in nb['cells']:
    source = ''.join(cell['source'])
    if cell['cell_type'] == 'markdown' and '## 一、手寫推導' in source:
        cell['source'] = text.splitlines(keepends=True)
        matched += 1
assert matched == 1, matched
p.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
assert '$' not in text and '\\' not in text
print('Updated handwritten_derivation.md and the notebook derivation cell; no LaTeX delimiters or commands remain in the derivation.')
