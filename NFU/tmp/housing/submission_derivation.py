from pathlib import Path
import json
p=Path('Logistic_Regression/python.ipynb')
nb=json.loads(p.read_text(encoding='utf-8'))
s=Path('Logistic_Regression/handwritten_derivation.md').read_text(encoding='utf-8-sig')
count=0
for c in nb['cells']:
    if c['cell_type']=='markdown' and ''.join(c['source']).startswith('# 第一題：'):
        c['source']=s.splitlines(keepends=True)
        count+=1
assert count==1
assert '$' not in s and '\\' not in s
p.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print('Saved submission derivation and synchronized notebook Markdown; code unchanged.')
