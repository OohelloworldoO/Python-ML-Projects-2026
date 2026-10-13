from pathlib import Path
import json
p=Path('Logistic_Regression/python.ipynb')
nb=json.loads(p.read_text(encoding='utf-8'))
text=Path('Logistic_Regression/handwritten_derivation.md').read_text(encoding='utf-8-sig')
count=0
for cell in nb['cells']:
    if cell['cell_type']=='markdown' and ''.join(cell['source']).startswith('# 第一題：'):
        cell['source']=text.splitlines(keepends=True)
        count+=1
assert count==1
assert '$' not in text and '\\' not in text
p.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
print('Updated both derivation copies; code and saved outputs preserved.')
