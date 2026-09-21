"""Local layout/data checks and page previews for final.tex."""
from pathlib import Path
from collections import defaultdict
import csv
import json
import pymupdf as fitz
from PIL import Image, ImageDraw

HERE=Path(__file__).resolve().parent
OUT=HERE/'final_checks'
OUT.mkdir(exist_ok=True)
doc=fitz.open(HERE/'final.pdf')
print('Pages:',len(doc))
for i,p in enumerate(doc):
    text=p.get_text()
    spans=[s for b in p.get_text('dict')['blocks'] if 'lines' in b for l in b['lines'] for s in l['spans']]
    outside=[s['text'] for s in spans if s['bbox'][0]<45 or s['bbox'][2]>552]
    print(f'{i+1:02}: {len(text):4} chars; outside body width: {outside}')
    p.get_pixmap(matrix=fitz.Matrix(1.4,1.4),alpha=False).save(OUT/f'page_{i+1:02}.png')
for start in range(0,len(doc),6):
    canvas=Image.new('RGB',(900,1320),'#bbbbbb')
    draw=ImageDraw.Draw(canvas)
    for offset,p in enumerate(doc[start:start+6]):
        pix=p.get_pixmap(matrix=fitz.Matrix(.48,.48),alpha=False)
        im=Image.frombytes('RGB',(pix.width,pix.height),pix.samples)
        x=(offset%3)*300; y=(offset//3)*660
        canvas.paste(im,(x,y+25)); draw.text((x+8,y+6),f'Page {start+offset+1}',fill='black')
    canvas.save(OUT/f'contact_{start+1:02}.png')
rows=list(csv.DictReader((HERE.parent/'metadata/manifest.csv').open()))
hashpeople=defaultdict(set)
for r in rows: hashpeople[r['md5']].add(r['person_id'])
print('Cross-person duplicate hashes:',sum(len(v)>1 for v in hashpeople.values()))
for e in ['e0','e1','e2']:
    results=[json.loads((HERE.parent/f'runs/e1e2/folds/fold{f}/{e}_person_results.json').read_text()) for f in range(1,7)]
    n=sum(r['test_n'] for r in results)
    sums=[sum(r[k] for r in results) for k in ['test_correct_n','test_rejected_correct_n','test_misidentified_accepted_n']]
    print(e, 'n=',n,'correct/rejected-right/misidentified=',sums,'rejected-wrong=',n-sum(sums))
assert all('??' not in p.get_text() for p in doc)
assert 'SCRFD' not in ''.join(p.get_text() for p in doc)
print('Resolved-reference and detector-naming checks passed; page limit lifted by user.')
