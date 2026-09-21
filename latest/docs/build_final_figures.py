"""Rebuild report figures from saved data; no training or source-file changes."""
from pathlib import Path
import csv
import json
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import io
import pymupdf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / 'final_assets'
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'Arial', 'font.size':11, 'axes.titlesize':12,
                     'axes.labelsize':11, 'xtick.labelsize':10, 'ytick.labelsize':10,
                     'legend.fontsize':10, 'pdf.fonttype':42, 'savefig.dpi':200})
rows = list(csv.DictReader((ROOT/'metadata/manifest.csv').open()))
setup = json.loads((ROOT/'metadata/e1e2_person_folds.json').read_text())
occs = ['none','clearglass','sunglass','cap','scarf','mask','mask_clearglass','mask_sunglass']
labels = ['None','Clear glasses','Sunglasses','Cap','Scarf','Mask','Mask + clear','Mask + sun']
colors = ['#777777','#1671a5','#b54436']

def save(fig, name):
    fig.savefig(OUT/(name+'.pdf'))
    plt.close(fig)

fig, ax = plt.subplots(figsize=(6.75,2.6),layout='constrained')
counts=Counter(r['occlusion'] for r in rows)
b=ax.barh(labels[::-1],[counts[o] for o in occs[::-1]],color='#aacbdd',edgecolor='black',linewidth=.5)
ax.bar_label(b,padding=4,fontsize=10)
ax.set_xlim(0,1000); ax.set_xlabel('Retained aligned images'); ax.spines[['top','right']].set_visible(False)
save(fig,'composition')

fig, axs = plt.subplots(4,4,figsize=(6.75,4.3),layout='constrained')
for j,(o,label) in enumerate(zip(occs,labels)):
    candidates=[r for r in rows if r['person_id']=='p11' and r['occlusion']==o and r['lighting']=='regular']
    r=max(candidates,key=lambda x:float(x['det_score']))
    for row,base,field in [(2*(j//4), ROOT.parent/'processed_data', 'image_path'),(2*(j//4)+1,ROOT/'crops','crop_path')]:
        ax=axs[row,j%4]
        ax.imshow(ImageOps.exif_transpose(Image.open(base/'p11'/Path(r[field]).name)))
        ax.set_title(label+(' (raw)' if row%2==0 else ' (aligned)'),fontsize=10,pad=2)
        ax.axis('off')
save(fig,'samples')

have={(r['person_id'],Path(r['image_path']).name) for r in rows}
failures=[p for p in sorted((ROOT.parent/'processed_data').glob('*/*'))
          if p.suffix.lower() in ['.jpg','.jpeg'] and (p.parent.name,p.name) not in have]
assert len(failures)==12, len(failures)
selected_failures=[failures[i-1] for i in [1,2,5,6,7]]
fig,axs=plt.subplots(1,5,figsize=(6.75,2.25),layout='constrained')
for ax,p,num in zip(axs.flat,selected_failures,[1,2,5,6,7]):
    ax.imshow(ImageOps.exif_transpose(Image.open(p))); ax.axis('off')
    occ=next(o for o in sorted(occs,key=len,reverse=True) if '_'+o+'_' in p.stem)
    ax.set_title(f'{num}: {p.parent.name}\n'+labels[occs.index(occ)],fontsize=10,pad=2)
save(fig,'retinaface_rejected')

preds=list(csv.DictReader((ROOT/'runs/e1e2/analysis/e0_predictions.csv').open()))
rejected=[r for r in preds if r['outcome']=='rejected']
low=sorted([r for r in rejected if r['person_id']==r['pred_person']],key=lambda r:float(r['score']))[:6]
wrong=sorted([r for r in rejected if r['pred_person']!=r['person_id']],key=lambda r:-float(r['score']))[:6]
fig,axs=plt.subplots(2,6,figsize=(6.75,2.25),layout='constrained')
for ax,r in zip(axs.flat,low+wrong):
    ax.imshow(Image.open(ROOT/'crops'/r['person_id']/Path(r['crop_path']).name))
    ax.set_xticks([]);ax.set_yticks([])
    ax.set_xlabel(f"{r['person_id']} > {r['pred_person']}\ns = {float(r['score']):.2f}",fontsize=9.5,labelpad=2)
save(fig,'arcface_rejected')

# Preserve the exact four embedded source photographs, without retouching.
prior=pymupdf.open(HERE/'prev_report.pdf')
qc=[b for b in prior[6].get_text('dict')['blocks'] if b['type']==1]
assert len(qc)==4
fig,axs=plt.subplots(1,4,figsize=(6.75,1.5),layout='constrained')
for ax,b,label in zip(axs,qc,['Too blurry','Too dark','Fully covered','Mis-cropped']):
    ax.imshow(Image.open(io.BytesIO(b['image'])))
    ax.axis('off');ax.set_title(label,fontsize=11,pad=2)
save(fig,'quality_rejected')

fig, ax = plt.subplots(figsize=(6.75,2.3),layout='constrained')
ax.hist([float(r['det_score']) for r in rows],bins=35,color='#aacbdd',edgecolor='black',linewidth=.4)
ax.set_xlabel('Detection confidence'); ax.set_ylabel('Retained images')
ax.spines[['top','right']].set_visible(False)
save(fig,'det_scores')

fig,ax=plt.subplots(figsize=(6.75,2.5),layout='constrained')
for i,e in enumerate(['e0','e1','e2']):
    results=[json.loads((ROOT/f'runs/e1e2/folds/fold{f}/{e}_person_results.json').read_text()) for f in range(1,7)]
    vals=[]
    for o in occs:
        total=sum(r['per_occlusion'][o]['n'] for r in results)
        good=sum(r['per_occlusion'][o]['correct_n'] for r in results)
        vals.append(100*good/total)
    ax.plot(range(8),vals,'o-',label=e.upper(),color=colors[i],markersize=4)
ax.set_xticks(range(8),['None','Clear\nglasses','Sun\nglasses','Cap','Scarf','Mask','Mask\n+ clear','Mask\n+ sun'])
ax.set_ylabel('Accepted and correct (%)'); ax.set_ylim(25,104); ax.legend(ncol=3,loc='lower left')
ax.grid(axis='y',alpha=.25)
save(fig,'occlusion')

for e in ['e0','e1','e2']:
    data=list(csv.reader((ROOT/f'runs/e1e2/analysis/{e}_confusion_overall.csv').open()))
    cm=np.array([[int(x) for x in r[1:]] for r in data[1:]])
    mat=cm/cm.sum(axis=1,keepdims=True)
    fig,ax=plt.subplots(figsize=(6.75,5.65),layout='constrained')
    im=ax.imshow(mat,cmap='Blues',vmin=0,vmax=1,aspect='auto',interpolation='nearest')
    ax.set_xticks(range(31),[f'{i:02}' for i in range(1,31)]+['R'],rotation=90,fontsize=10)
    ax.set_yticks(range(30),[f'{i:02}' for i in range(1,31)],fontsize=10)
    ax.set_xlabel('Predicted identity (01 = p01, ..., 30 = p30; R = rejected)')
    ax.set_ylabel('True identity'); ax.axvline(29.5,color='black',linewidth=1)
    cb=fig.colorbar(im,ax=ax,fraction=.025,pad=.02)
    cb.set_label('Fraction of each true-identity row',fontsize=10)
    save(fig,e+'_confusion')

for e in ['e1','e2']:
    fig,ax=plt.subplots(figsize=(6.75,3.8),layout='constrained')
    for f in range(1,7):
        h=json.loads((ROOT/f'runs/e1e2/learning_curve/fold{f}/{e}_person_model_history.json').read_text())['history']
        for key,style in [('train_loss','-'),('val_loss','--')]:
            hs=[x for x in h if x[key] is not None]
            ax.plot([x['epoch'] for x in hs],[x[key] for x in hs],linestyle=style,
                    color=plt.cm.tab10(f-1),label=f'Fold {f}' if key=='train_loss' else None,linewidth=1.35)
    ax.set_xlabel('Epoch');ax.set_ylabel('Loss');ax.grid(alpha=.2)
    ax.set_title(e.upper()+': training (solid) and validation (dashed)',fontsize=11)
    ax.legend(ncol=3,fontsize=9,loc='upper right',columnspacing=.8,handlelength=1.4)
    save(fig,e+'_loss')

persons=[f'p{i:02}' for i in range(1,31)]
roles=np.empty((6,30),dtype=int)
for k,fold in enumerate(setup['folds']):
    for j,p in enumerate(persons):
        roles[k,j]=0 if p in fold['train'] else 1 if p in fold['val'] else 2
fig,ax=plt.subplots(figsize=(6.75,3.2),layout='constrained')
palette=['#B8DBB0','#F3DB83','#E4A3A0']
ax.imshow(roles,cmap=ListedColormap(palette),vmin=-.5,vmax=2.5,aspect='auto')
ax.set_xticks(range(30),persons,rotation=90,fontsize=10)
ax.set_yticks(range(6),[f'Fold {k}' for k in range(1,7)],fontsize=11)
ax.set_xticks(np.arange(-.5,30,1),minor=True); ax.set_yticks(np.arange(-.5,6,1),minor=True)
ax.grid(which='minor',color='white',linewidth=1);ax.tick_params(which='minor',length=0)
ax.set_xlabel('Participant identity (the whole person receives one role per fold)')
ax.legend(handles=[Patch(facecolor=c,label=l) for c,l in zip(palette,['Train (20)','Validation (5)','Test (5)'])],
          ncol=3,loc='upper center',bbox_to_anchor=(.5,1.23),frameon=False,fontsize=10)
save(fig,'fold_assignment')

fig,axs=plt.subplots(1,2,figsize=(6.75,2.25),layout='constrained')
for ax,e in zip(axs,['e1','e2']):
    for f in range(1,7):
        d=json.loads((ROOT/f'runs/e1e2/folds/fold{f}/{e}_person_model_history.json').read_text())
        h=d['history']; line,=ax.plot([x['epoch'] for x in h],[100*x['val_acc'] for x in h],label=str(f))
        selected=next(x for x in h if x['epoch']==d['best_epoch'])
        ax.plot(selected['epoch'],100*selected['val_acc'],'o',color=line.get_color(),markersize=4)
    ax.set_title(e.upper()); ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
axs[0].set_ylabel('Validation accuracy (%)')
axs[1].legend(title='Fold',ncol=3,fontsize=8,title_fontsize=9,loc='lower right',columnspacing=.7,handlelength=1)
save(fig,'epochs')
print('Created',len(list(OUT.glob('*.pdf'))),'vector figures in',OUT)
