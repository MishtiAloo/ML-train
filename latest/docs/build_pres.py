"""Create the current-project deck while retaining the supplied deck's theme."""
from pathlib import Path
from io import BytesIO
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.xmlchemy import OxmlElement
from PIL import Image, ImageOps, ImageDraw
import pymupdf

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
OUT=HERE/'pres_assets'; OUT.mkdir(exist_ok=True)
P=Presentation(HERE/'Occlusion-Robust_Face_Identification_ArcFace_v4.pptx')
photos=[(x.image.blob,x.crop_left,x.crop_top,x.crop_right,x.crop_bottom) for x in P.slides[3].shapes if x.shape_type==13]
v4_inference_art=next(x.image.blob for x in P.slides[10].shapes if x.shape_type==13)
# Retain the original cover, slide layouts, masters, and Office theme package.
for sid in list(P.slides._sldIdLst)[1:]:
    P.part.drop_rel(sid.rId); P.slides._sldIdLst.remove(sid)
C={'ink':'1B1B27','navy':'203C80','muted':'566174','line':'D6DDE9','pale':'EFF4FF','white':'FFFFFF','teal':'008E80','red':'B54436','gray':'F5F5F5'}
def rgb(c): return RGBColor.from_string(C.get(c,c))
def text(s,x,y,w,h,txt,size=13,color='ink',bold=False,font='Arimo',align=None):
    sh=s.shapes.add_textbox(Inches(x),Inches(y),Inches(w),Inches(h))
    tf=sh.text_frame; tf.clear(); tf.word_wrap=True
    tf.margin_left=tf.margin_right=Inches(.02); tf.margin_top=tf.margin_bottom=0
    for i,line in enumerate(txt.split('\n')):
        p=tf.paragraphs[0] if i==0 else tf.add_paragraph(); p.text=line
        p.font.name=font; p.font.size=Pt(size); p.font.bold=bold; p.font.color.rgb=rgb(color)
        p.space_after=Pt(5)
        if align is not None: p.alignment=align
    return sh
def rect(s,x,y,w,h,fill='pale',line='line',radius=True):
    sh=s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE, Inches(x),Inches(y),Inches(w),Inches(h))
    sh.fill.solid(); sh.fill.fore_color.rgb=rgb(fill); sh.line.color.rgb=rgb(line); sh.line.width=Pt(.6)
    sh._element.spPr.append(OxmlElement('a:effectLst'))
    if radius: sh.adjustments[0]=.07
    return sh
def card(s,x,y,w,h,title,body,size=14):
    rect(s,x,y,w,h,'FBFCFF')
    compact=h<1.2
    text(s,x+.16,y+(.08 if compact else .13),w-.32,.34,title,15 if compact else 17,'navy',True)
    offset=.43 if compact else .66
    text(s,x+.16,y+offset,w-.32,max(.26,h-offset-.05),body,size,'ink')
NODE_PALETTE={'E0E0E0','97ECF8','FFD0AC','FFA0A0','FFF17B','ACA7FF','A4FFA1','F4C5FF'}
def node(s,x,y,w,h,title,body='',fill='FFD0AC',size=13):
    assert fill in NODE_PALETTE, fill
    shape=rect(s,x,y,w,h,fill); shape.name='Diagram node: '+title
    text(s,x+.06,y+.10,w-.12,.40 if body else h-.17,title,size,'navy',True,align=PP_ALIGN.CENTER)
    if body: text(s,x+.06,y+.52,w-.12,h-.55,body,size-1,'ink',align=PP_ALIGN.CENTER)
def arrow(s,x1,y1,x2,y2,color='navy'):
    # Thin native line with an explicit triangular arrowhead (editable).
    sh=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1),Inches(y1),Inches(x2),Inches(y2)); sh.line.color.rgb=rgb(color); sh.line.width=Pt(1.5)
    tip=s.shapes.add_shape(MSO_SHAPE.ISOSCELES_TRIANGLE, Inches(x2-.045),Inches(y2-.045),Inches(.09),Inches(.09))
    tip.rotation=90 if x2>x1 else 270 if x2<x1 else 180 if y2>y1 else 0
    tip.fill.solid(); tip.fill.fore_color.rgb=rgb(color); tip.line.fill.background()
    tip._element.spPr.append(OxmlElement('a:effectLst'))
def slide(title,sub):
    s=P.slides.add_slide(P.slide_layouts[0])
    s._element.set('showMasterSp','0')
    # Remove inherited placeholders, retaining the original master/theme.
    for sh in list(s.shapes): sh._element.getparent().remove(sh._element)
    s.background.fill.solid(); s.background.fill.fore_color.rgb=rgb('white')
    text(s,.54,.30,8.92,.48,title,27,font='Raleway')
    text(s,.54,.84,8.92,.37,sub,11.5,'muted')
    sh=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(.3),Inches(5.29),Inches(9.68),Inches(5.29)); sh.line.color.rgb=rgb('C9D6EE'); sh.line.width=Pt(.6)
    text(s,.3,5.36,2,.17,'20 Sep 2026',9,'4A5782'); text(s,9.35,5.36,.33,.17,f'{len(P.slides)-1:02}',9,'4A5782',align=PP_ALIGN.RIGHT)
    return s
def band(s,txt,y=4.66,h=.43,size=12):
    rect(s,.54,y,8.92,h,'gray','gray',False); text(s,.70,y+.08,8.60,h-.1,txt,size,'muted')
def table(s,headers,rows,widths,y=1.35,rowh=.48,size=12):
    x=.54
    for i,vals in enumerate([headers]+rows):
        for val,w in zip(vals,widths):
            rect(s,x,y+i*rowh,w,rowh,'1B2A54' if i==0 else ('EFF4FF' if i%2 else 'FBFCFF'),'white',False)
            text(s,x+.08,y+i*rowh+.10,w-.16,rowh-.12,str(val),size,'white' if i==0 else 'ink',i==0)
            x+=w
        x=.54
def asset_pdf(name):
    dest=OUT/(name+'.png')
    doc=pymupdf.open(HERE/'final_assets'/f'{name}.pdf'); doc[0].get_pixmap(matrix=pymupdf.Matrix(3,3),alpha=False).save(dest); doc.close(); return dest
def picfit(s,path,x,y,w,h):
    im=Image.open(path); iw,ih=im.size; scale=min(w/iw,h/ih); pw,ph=iw*scale,ih*scale
    return s.shapes.add_picture(str(path),Inches(x+(w-pw)/2),Inches(y+(h-ph)/2),width=Inches(pw),height=Inches(ph))
def plot(name,fn,wh=(9,3.7)):
    plt.rcParams.update({'font.family':'Arial','font.size':13,'axes.spines.top':False,'axes.spines.right':False,'axes.labelsize':13,'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':11})
    fig,ax=plt.subplots(figsize=wh,layout='constrained'); fn(fig,ax); path=OUT/(name+'.png'); fig.savefig(path,dpi=220,facecolor='white'); plt.close(fig); return path

# Original portrait and cover geometry stay unchanged.
cover=P.slides[0]
for sh in cover.shapes:
    if sh.has_text_frame and 'Occlusion-Robust' in sh.text:
        sh.top=Inches(.55); sh.height=Inches(1.37)
        sh.text_frame.paragraphs[0].runs[0].text='Occlusion-Robust Face Identification with Partially Fine-Tuned ArcFace'
        for p in sh.text_frame.paragraphs[1:]: p.text=''
        # Replace the complete title without inheriting stale runs.
        sh.text_frame.clear(); p=sh.text_frame.paragraphs[0]; p.text='Occlusion-Robust Face Identification with Partially Fine-Tuned ArcFace'; p.font.name='Raleway'; p.font.size=Pt(25); p.font.color.rgb=rgb('ink')
    elif sh.has_text_frame and 'September' in sh.text:
        sh.top=Inches(2.02); sh.height=Inches(1.05)
        for p in sh.text_frame.paragraphs:
            p.space_before=Pt(0); p.space_after=Pt(5); p.line_spacing=1.0
            for r in p.runs:
                r.text=r.text.replace('08 September 2026','20 September 2026'); r.font.size=Pt(9)
text(cover,4.29,3.18,5.16,.31,'Team members (rolls)',15,'navy',True)
for j,roll in enumerate(['2107031','2107044','2107046','2107047','2107057','2107059']):
    text(cover,4.31,3.64+j*.25,5.10,.24,roll,12,'ink')

s=slide('Research Question','Does occlusion robustness transfer to identities never seen during fine-tuning?')
card(s,.54,1.38,4.3,2.91,'The task','Identify an enrolled person from an occluded face.\nCompare three adaptation scopes using the same gallery and decision threshold.\nA correct label below the threshold is still a rejection.',14)
card(s,5.05,1.38,4.41,2.91,'The experiment','E0 · Frozen pretrained ArcFace\nE1 · Fine-tune the final residual block\nE2 · Fine-tune the block + embedding head\n20 train / 5 validation / 5 test identities per fold.',14)
band(s,'Person-disjoint fine-tuning evaluation; all 30 identities remain enrolled in the gallery.')

s=slide('Dataset & Occlusion Conditions','In-person capture, signed consent, pseudonymous identities p01–p30.')
for x,big,title,detail in [( .54,'30','Identities','6 collectors × 5 people'),(2.8,'6,000','Raw images','200 per person'),(5.06,'5,988','Aligned crops','99.8% detection'),(7.32,'8 × 2','Conditions','occlusion × lighting')]:
    rect(s,x,1.32,2.1,1.34,'FBFCFF'); text(s,x+.08,1.48,1.94,.49,big,28,'navy',True,align=PP_ALIGN.CENTER); text(s,x+.08,2.01,1.94,.3,title,14,'ink',True,align=PP_ALIGN.CENTER); text(s,x+.05,2.37,2,.2,detail,10,'muted',align=PP_ALIGN.CENTER)
card(s,.54,2.89,4.3,1.6,'Capture protocol','Phone-camera collection in regular and low light. Single-sitting capture; indoor conditions.',14)
card(s,5.05,2.89,4.41,1.6,'Evaluation pool','5,818 queries after 170 gallery / near-burst exclusions. Fixed gallery: 30 identities.',14)
link=text(s,.60,4.76,8.85,.28,'Dataset folder · consent and access restrictions still apply',12,'navy')
link.text_frame.paragraphs[0].runs[0].hyperlink.address='https://drive.google.com/drive/u/1/folders/1XN55OGpozy5Rhbs5QRFMatlCMTAo8sPu'

s=slide('Eight Occlusion Conditions','Representative participant images reused from the previous presentation.')
labels=['None','Clear glasses','Sunglasses','Cap','Scarf','Mask','Mask + clear','Mask + sunglasses']
# Preserve the source montage ordering by reading its labels, not guessing image identity.
src=Presentation(HERE/'Occlusion-Robust_Face_Identification_ArcFace_v4.pptx').slides[3]
for sh in src.shapes:
    if sh.shape_type==13:
        new=s.shapes.add_picture(BytesIO(sh.image.blob),sh.left,sh.top,sh.width,sh.height)
        new.crop_left=sh.crop_left; new.crop_right=sh.crop_right; new.crop_top=sh.crop_top; new.crop_bottom=sh.crop_bottom
for sh in src.shapes:
    if sh.has_text_frame and 2.65<sh.top/914400<5.2:
        text(s,sh.left/914400,sh.top/914400,sh.width/914400,sh.height/914400,sh.text,11,'ink',True,align=PP_ALIGN.CENTER)

s=slide('Why This Dataset?','Pgu-Face is the base dataset paper—not a directly comparable recognition benchmark.')
table(s,['Property','Pgu-Face (2016)','Our dataset'],[
['Participants','224','30'],['Images / person','4 (896 total)','200 raw (6,000 total)'],['Covering variation','Facial hair + sunglasses','8 types, including compound masks'],['Capture conditions','Up to 2 sessions, ≥6 days apart','Regular + low light; one sitting'],['Evaluation artifacts','Partially covered-face collection','112 × 112 crops, labels, six-fold split']], [2.05,3.05,3.82],rowh=.52,size=11.5)
band(s,'More within-person variation and coverings; fewer identities and no across-session evaluation.',y=4.65,size=12)

s=slide('Different—and Better for This Question','Broader occlusion stress testing, rather than a claim of universally better data.')
card(s,.54,1.38,4.30,3.03,'What the new data enables','~6.7× as many raw images overall.\n50× as many captures per identity.\nMask + glasses combinations probe severe, compound occlusion.\nTwo light levels and condition labels support targeted error analysis.',14)
card(s,5.05,1.38,4.41,3.03,'What Pgu-Face still offers','224 identities versus our 30: greater identity diversity.\nSeparated capture sessions probe time-related change.\nDifferent data and protocols mean published accuracies cannot be ranked directly.',14)
band(s,'The advantage is alignment with our occlusion-transfer study—not broader population coverage.')

s=slide('Curation, Quality & Provenance','Retain the documented dataset; disclose limitations rather than inventing cleanup counts.')
card(s,.54,1.38,4.30,2.98,'Processing checks','12 raw images failed detection.\n5,988 crops: 3,050 regular / 2,938 low light.\n429 duplicate MD5 occurrences retained, all within the same identity.\nPerson-disjoint folds keep a person’s copies together.',13.5)
card(s,5.05,1.38,4.41,2.98,'Coverage & consent','Scarf coverage: 24 of 30 people; some other condition gaps.\nA small, unflagged digitally added covering subset was reported previously.\nAcademic, non-commercial, withdrawable consent; faces are still identifiable.',13.5)
band(s,'No claim of perfectly balanced coverage, fully natural coverings, or an unrestricted public license.')

s=slide('Detection Failures: Actual Examples','Five documented rejected raw images: examples 1, 2, 5, 6 and 7 from the report.')
picfit(s,asset_pdf('retinaface_rejected'),.54,1.4,8.92,2.8)
band(s,'RetinaFace detects 5,988 of 6,000 raw images. These failures never enter the recognition evaluation.')

s=slide('Preprocessing Pipeline','Frozen RetinaFace detection and landmark alignment produce the recognizer’s inputs.')
for x,title,body in [( .54,'Raw photo','Read orientation\nRGB image'),(2.83,'RetinaFace','Face box + score\n5 landmarks'),(5.12,'Alignment','Similarity transform\n112 × 112 RGB'),(7.41,'Manifest','Identity, covering, light\nCrop + quality fields')]:
    node(s,x,1.85,2.05,1.35,title,body,fill='97ECF8' if title=='Raw photo' else 'A4FFA1' if title=='Manifest' else 'FFD0AC',size=13)
    if x<7: arrow(s,x+2.05,2.53,x+2.28,2.53)
card(s,.54,3.65,8.92,.91,'Recognition normalization','(pixel − 127.5) / 127.5; channel-first tensor N × 3 × 112 × 112.',13)

s=slide('Person-Disjoint Six-Fold Evaluation','Seed 42 · 20 train identities, 5 validation identities, 5 held-out test identities per fold.')
picfit(s,asset_pdf('fold_assignment'),.54,1.27,8.92,3.55)
text(s,.65,4.89,8.7,.22,'Every identity is tested once. Validation uses the next five-person block cyclically.',11,'muted')

s=slide('One Fixed Gallery, Fair Comparisons','Enrollment is not fine-tuning: held-out test identities have a gallery reference only.')
card(s,.54,1.38,4.30,2.95,'Build once with E0','Choose the highest-detection-score “none” crop per identity.\nCompute and L2-normalize its pretrained 512-D embedding.\nLock these 30 reference vectors across E0, E1, and E2.',14)
card(s,5.05,1.38,4.41,2.95,'Prevent shortcut evaluation','Exclude gallery photos and near-burst queries (170 images total).\nFine-tuning uses only the 20 training identities’ fixed vectors.\nValidation and testing match against all 30 gallery identities.',14)
band(s,'5,818 pooled test queries. No held-out person contributes fine-tuning gradients.')

s=slide('Two Models, Two Distinct Roles','Detector stays frozen. Only selected ArcFace recognition layers are adapted.')
node(s,.7,1.7,3.7,1.8,'1 · RetinaFace','Raw RGB → box, score, 5 landmarks\n4,225,835 parameters\nDetection and alignment only',fill='E0E0E0',size=16)
arrow(s,4.4,2.6,5.35,2.6)
node(s,5.35,1.7,3.95,1.8,'2 · ArcFace / iResNet-50','Aligned RGB → 512-D embedding\n43,590,976 parameters\nFixed-gallery cosine identification',fill='F4C5FF',size=16)
band(s,'The following diagrams reflect the exported model graphs used in the report.',y=4.35)

s=slide('RetinaFace: Actual Architecture','Frozen det_10g.onnx · typical detector input 1 × 3 × 640 × 640 RGB.')
node(s,.54,2.18,1.2,1.35,'Input RGB','640 × 640\nDynamic H, W',fill='97ECF8',size=11)
node(s,1.99,2.18,1.35,1.35,'Backbone','Convolutional\nfeatures',fill='E0E0E0',size=11)
arrow(s,1.74,2.85,1.99,2.85)
node(s,3.59,2.18,1.3,1.35,'3-level FPN','Multi-scale\nfusion',fill='E0E0E0',size=11)
arrow(s,3.34,2.85,3.59,2.85)
def plainline(s,x1,y1,x2,y2):
    sh=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(x1),Inches(y1),Inches(x2),Inches(y2)); sh.line.color.rgb=rgb('navy'); sh.line.width=Pt(1.2); sh._element.spPr.append(OxmlElement('a:effectLst'))
plainline(s,4.89,2.85,5.02,2.85); plainline(s,5.02,1.88,5.02,4.04)
for y,title,body in [(1.40,'P3 · stride 8','80 × 80'),(2.48,'P4 · stride 16','40 × 40'),(3.56,'P5 · stride 32','20 × 20')]:
    shape=rect(s,5.14,y,1.9,.96,'E0E0E0'); shape.name='Diagram node: '+title
    text(s,5.20,y+.07,1.78,.25,title,12,'navy',True,align=PP_ALIGN.CENTER)
    text(s,5.20,y+.37,1.78,.54,body,11,'ink',align=PP_ALIGN.CENTER)
    arrow(s,5.02,y+.48,5.14,y+.48); arrow(s,7.04,y+.48,7.32,y+.48)
    shape=rect(s,7.32,y,2.14,.96,'A4FFA1'); shape.name='Diagram node: '+title+' outputs'
    text(s,7.38,y+.09,2.02,.80,'Face scores\nBounding boxes\nFive facial landmarks',11,'ink')
text(s,7.30,1.12,2.18,.22,'OUTPUTS',11,'navy',True,align=PP_ALIGN.CENTER)
text(s,.60,3.97,4.35,.61,'Nine output tensors per image:\nscores, boxes and landmarks at each level.',12,'muted')
band(s,'Outputs → box decoding / filtering → 5-landmark alignment → 112 × 112 RGB crop.',y=4.72)

s=slide('ArcFace: Actual iResNet-50 Architecture','w600k_r50.onnx · 24 residual blocks; block internals intentionally omitted.')
# Two rows keep all tensor shapes legible without shrinking the labels.
xs=[.54,2.83,5.12,7.41]
top=[('Input RGB','N × 3 × 112 × 112'),('Stem','Convolution + PReLU'),('Stage 1','3 residual blocks'),('Stage 2','4 residual blocks')]
bottom=[('Stage 3','14 residual blocks'),('Stage 4','3 residual blocks\nLast block: layer4.2'),('Embedding head','BN → flatten\nFC → final BN'),('Output embedding','N × 512\nExternal L2 normalization')]
for row,items in [(1.45,top),(3.28,bottom)]:
    for j,(title,body) in enumerate(items):
        fill=('97ECF8' if j==0 else 'E0E0E0') if row<3 else ['E0E0E0','F4C5FF','F4C5FF','A4FFA1'][j]
        node(s,xs[j],row,2.05,1.20,title,body,fill=fill,size=12)
        if j<3: arrow(s,xs[j]+2.05,row+.60,xs[j+1],row+.60)
# Row transition uses a clear gutter, not boxes.
for a,b,c,d in [(8.44,2.65,8.44,2.98),(8.44,2.98,1.57,2.98),(1.57,2.98,1.57,3.28)]:
    sh=s.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(a),Inches(b),Inches(c),Inches(d)); sh.line.color.rgb=rgb('navy'); sh.line.width=Pt(1.2)
arrow(s,1.57,3.11,1.57,3.28)
band(s,'E1: final block of stage 4 (layer4.2). E2: that block + embedding head. Earlier layers stay frozen.',size=11.5)

s=slide('E0 / E1 / E2: What Changes?','The reference vectors and evaluation rule are identical; the trainable scope changes.')
table(s,['Experiment','Trainable layers','Trainable parameters'],[
['E0 · pretrained','None','0'],['E1 · block only','Final stage-4 residual block (layer4.2)','4,720,640'],['E2 · block + head','Same block + BN / FC / final BN','17,568,256']], [2.1,4.37,2.45],rowh=.66,size=13)
card(s,.54,4.15,8.92,.88,'Important distinction','The 20-vector training classifier is a fixed buffer—not a learned classification head.',12.5)

s=slide('Training Pipeline','20 training identities per fold; gradients affect only the selected E1 or E2 layers.')
items=[('Aligned crops','Training identities\nAugment + normalize'),('ArcFace','Selected layers trainable\n512-D L2 embeddings'),('Cosine logits','20 locked E0 references\nAngular margin + scale'),('Loss + update','Smoothed cross-entropy\nAdamW on selected layers')]
for j,(title,body) in enumerate(items):
    node(s,xs[j],1.58,2.05,1.42,title,body,fill=['97ECF8','F4C5FF','FFD0AC','ACA7FF'][j],size=12)
    if j<3: arrow(s,xs[j]+2.05,2.29,xs[j+1],2.29)
card(s,.54,3.50,4.30,1.18,'Validation','All 30 gallery vectors; select by validation accuracy, then confidence gap.',12.5)
card(s,5.05,3.50,4.41,1.18,'Baseline is eligible','Epoch 0 is a candidate. E2 keeps E0 on folds 3, 4 and 5.',12.5)

s=slide('Training Recipe','Same optimization schedule and data protocol for both fine-tuned configurations.')
table(s,['Setting','Configuration'],[
['Optimizer','AdamW · learning rate 3 × 10⁻⁵ · weight decay 10⁻⁴'],['Schedule','Cosine decay (Tmax = 20); maximum 20 epochs; patience 5'],['Batch / seed','Train 32 · evaluation 64 · random seed 42'],['Classification loss','Scale 32; label smoothing 0.1; fixed benchmark classifier'],['Angular margin','0 for epochs 1–2; ramp to 0.20 over epochs 3–6'],['Batch normalization','Running statistics stay frozen; selected affine terms may train']], [2.3,6.62],rowh=.48,size=12)
band(s,'Augmentations: flip, down-up blur, Gaussian blur, noise, JPEG compression, brightness / contrast.',y=4.86,h=.28,size=10.5)

s=slide('Inference & Decision Rule','Identical identification pipeline and acceptance threshold for all three configurations.')
# V4 slide 11's visual grammar: six numbered panels, image/diagram on
# the left and the explanation on the right, read left-to-right in two rows.
steps=[('Face Detection','RetinaFace returns a face box and five landmarks.','97ECF8'),
       ('Alignment & Crop','Five landmarks align the face to a 112 × 112 RGB crop.','FFD0AC'),
       ('Feature Extraction','E0 or fold-specific E1 / E2 ArcFace produces 512 features.','E0E0E0'),
       ('L2 Normalization','Normalize the embedding to unit length. No rolling buffer.','FFD0AC'),
       ('Gallery Matching','Compare with all 30 locked pretrained gallery vectors.','F4C5FF'),
       ('Decision Rule','Choose the top identity if cosine ≥ 0.30; else UNKNOWN.','FFF17B')]
for j,(title,body,fill) in enumerate(steps):
    x=.36+(j%3)*3.12; y=1.31+(j//3)*1.69
    sh=rect(s,x,y,3.02,1.55,'FFFFFF'); sh.name='Neutral inference panel: '+str(j+1)
    text(s,x+.11,y+.07,.43,.35,f'{j+1:02}',23,'D32F2F',font='Arimo')
    text(s,x+.62,y+.13,2.3,.3,title,13,'ink',True)
    plainline(s,x+.10,y+.47,x+2.91,y+.47)
    text(s,x+1.23,y+.58,1.65,.87,body,11,'ink')
    if j in [0,1]:
        # Native PowerPoint crop of v4's existing illustration: preserve its
        # original face, red detection box and five landmark markers exactly.
        left,top,right,bottom=(.039,.298,.163,.456) if j==0 else (.772,.313,.834,.425)
        image=s.shapes.add_picture(BytesIO(v4_inference_art),Inches(x+.12),Inches(y+.62),width=Inches(.99 if j==0 else .77),height=Inches(.77))
        image.crop_left=left; image.crop_top=top; image.crop_right=1-right; image.crop_bottom=1-bottom
        if j==1: text(s,x+.12,y+1.37,.80,.16,'112 × 112',9,'navy',align=PP_ALIGN.CENTER)
    elif j==2:
        node(s,x+.10,y+.64,1.04,.71,'ArcFace','→ 512-D',fill='E0E0E0',size=10)
    elif j==3:
        text(s,x+.10,y+.77,1.05,.51,'z / ‖z‖₂',18,'navy',True,align=PP_ALIGN.CENTER)
    elif j==4:
        text(s,x+.09,y+.70,1.08,.71,'512-D\n×\n30 vectors',11,'navy',True,align=PP_ALIGN.CENTER)
    else:
        for dy,label,c in [(.65,'IDENTITY','A4FFA1'),(1.03,'UNKNOWN','FFA0A0')]:
            shape=rect(s,x+.1,y+dy,1.04,.28,'E0E0E0'); shape.name='Diagram node: '+label
            text(s,x+.11,y+dy+.04,1.02,.20,label,9,'ink',True,align=PP_ALIGN.CENTER)
    if j%3<2: text(s,x+3.00,y+.74,.20,.33,'›',18,'navy',True)
band(s,'Correct = right identity AND cosine ≥ 0.30. All queries are enrolled people; UNKNOWN means rejection.',y=4.77,h=.34,size=10.5)

s=slide('Main Results: Pooled Held-Out Queries','5,818 queries across all six test folds; the report’s saved aggregate results.')
table(s,['Model','Accuracy','Macro precision','Macro recall','Macro F1'],[
['E0','84.98%','100.00%','85.12%','91.67%'],['E1','86.18%','99.72%','86.33%','92.15%'],['E2','84.72%','96.43%','84.87%','89.85%']], [1.1,1.65,2.25,2.02,1.9],rowh=.65,size=13)
card(s,.54,4.12,8.92,.88,'Best configuration: E1','+1.20 percentage points over E0, with 4.72M trainable parameters. E2 is 0.26 points below E0.',12.5)

s=slide('Consistency Across Six Test Folds','E1 improves over E0 on every fold; E2 improves on only two test folds.')
def folds(fig,ax):
    x=np.arange(1,7)
    for name,vals,c in [('E0',[84.39,80.71,88.60,80.47,90.83,85.02],'#777777'),('E1',[86.76,81.02,89.02,82.01,92.40,86.06],'#203C80'),('E2',[85.01,81.02,88.60,80.47,90.83,82.52],'#B54436')]: ax.plot(x,vals,'o-',label=name,color=c,linewidth=2)
    ax.set(xlabel='Held-out test fold',ylabel='Accepted-correct accuracy (%)',xticks=x,ylim=(78,95)); ax.grid(alpha=.18); ax.legend(ncol=3)
picfit(s,plot('fold_results',folds),.54,1.24,8.92,3.8)

s=slide('Where E1 Helps Most','The largest benefit is severe compound occlusion—not uniform improvement in every condition.')
def occ(fig,ax):
    cats=['None','Clear','Sun','Cap','Scarf','Mask','Mask + clear','Mask + sun']; x=np.arange(8)
    for j,(name,vals,c) in enumerate([('E0',[100,100,98.5,95.5,97.3,90.2,68.5,35.6],'#9CA3AF'),('E1',[99.4,99,99.2,95.2,94.9,89.9,68.9,47.2],'#203C80'),('E2',[99.6,98.7,97.2,94.3,96,88,66.7,42.7],'#81A8EF')]): ax.bar(x+(j-1)*.24,vals,.24,label=name,color=c)
    ax.set(xticks=x,xticklabels=cats,ylim=(0,110),ylabel='Accepted-correct accuracy (%)'); ax.legend(ncol=3,loc='lower left'); ax.grid(axis='y',alpha=.15)
picfit(s,plot('conditions',occ),.54,1.23,8.92,3.48)
band(s,'Mask + sunglasses: 35.6% → 47.2% (+11.6 points). Scarf, mask, and clean-face results do not all improve.',y=4.77,h=.33,size=10.5)

s=slide('Accuracy Alone Hides the Trade-Off','Lower rejection can come with more accepted wrong identities.')
table(s,['Outcome (count)','E0','E1','E2'],[
['Accepted, correct','4,944','5,014','4,929'],['Rejected, correct argmax','727','493','421'],['Rejected, wrong argmax','147','298','275'],['Accepted, wrong','0','13','193']], [4.0,1.64,1.64,1.64],rowh=.57,size=13)
card(s,.54,4.16,8.92,.90,'Interpretation','E2 accepts 193 wrong identities (3.32%), versus 13 (0.22%) for E1.',12.5)

for e in ['e1','e2']:
    s=slide(f'{e.upper()}: Training & Validation Loss','One common loss: solid = training people, dashed = held-out validation people; one colour per fold.')
    # Both sides use the same objective -- clean crops, no angular margin, no label
    # smoothing, cross-entropy over the 30-benchmark gallery -- from
    # runs/e1e2/learning_curve_matched/ (scripts/27_e1e2_matched_curve.py), which
    # reproduces every selected epoch and adds only the matched training-side loss.
    def loss(fig,ax,e=e):
        for f in range(1,7):
            d=json.loads((ROOT/f'runs/e1e2/learning_curve_matched/fold{f}/{e}_person_model_history.json').read_text())
            h=d['history']
            for key,style in [('train_loss_matched','-'),('val_loss','--')]:
                valid=[r for r in h if r.get(key) is not None]
                ax.plot([r['epoch'] for r in valid],[r[key] for r in valid],style,color=plt.cm.tab10(f-1),label=f'Fold {f}' if key=='train_loss_matched' else None,lw=1.7)
            sel=next(r for r in h if r['epoch']==d['best_epoch'])
            ax.plot(sel['epoch'],sel['val_loss'],'*',color=plt.cm.tab10(f-1),markersize=11,markeredgecolor='black',markeredgewidth=.4,zorder=5)
        ax.set(xlabel='Epoch (0 = pretrained model)',ylabel='Gallery cross-entropy'); ax.grid(alpha=.2); ax.legend(ncol=3,loc='upper right')
    picfit(s,plot(e+'_loss',loss,wh=(9,3.6)),.54,1.23,8.92,3.55)
    text(s,.65,4.85,8.7,.25,'Same objective on both sides (clean crops, no margin, no smoothing, 30-benchmark gallery), so the gap between the lines is a real generalisation gap. Stars mark the selected epoch, chosen by validation accuracy.',10.5,'muted')

s=slide('Why More Trainable Layers Did Not Win','The block-only update is the strongest transfer configuration in this experiment.')
card(s,.54,1.38,4.30,2.93,'Evidence','E1: 86.18%, improvements in all six folds.\nE2: 84.72%, 193 accepted wrong labels.\nValidation retained epoch 0 in three E2 folds.\nThe embedding FC matrix alone has ~12.85M weights.',14)
card(s,5.05,1.38,4.41,2.93,'Interpretation—not proof','With only 20 training people, the larger update may adapt too strongly to them.\nA fixed pretrained gallery may amplify embedding-space drift.\nThese explanations need dedicated ablations; the measured conclusion is narrower.',14)
band(s,'Fine-tuning scope matters: increasing capacity does not guarantee better held-out identity transfer.')

s=slide('Related Work & Positioning','Complementary precedents; their reported accuracies use different datasets and protocols.')
table(s,['Work','Contribution','Relation to our project'],[
['Salari & Rostami, 2016','Pgu-Face partially covered-face dataset','Collection precedent; broaden covering types'],['Chong et al., 2022','CNN features + SVM for masked faces','Different recognition pipeline and benchmark'],['Montero et al., 2022','Multi-task ArcFace for masked faces','Our study varies layer scope, not task losses'],['Deng et al., 2019 / 2020','ArcFace / RetinaFace','Embedding and detection foundations']], [2.1,3.35,3.47],rowh=.71,size=11.5)
band(s,'Our contribution: a documented local collection and a fixed-gallery, person-disjoint adaptation study.',y=4.71,size=11.5)

s=slide('Limitations & Responsible Use','The results describe this collection and protocol, not unrestricted real-world recognition.')
card(s,.54,1.38,4.30,3.36,'Data limitations','Only 30 people; indoor, single-sitting capture.\nScarf coverage and some conditions are incomplete.\nSame-identity duplicates remain.\nPreviously added digital coverings are not flagged individually.\nFaces remain personal data despite pseudonyms.',13)
card(s,5.05,1.38,4.41,3.36,'Evaluation limitations','All test queries are enrolled identities; no unseen-stranger rejection benchmark.\nThreshold 0.30 is fixed, not a deployment guarantee.\nGallery is locked to pretrained embeddings.\nConsent is academic / non-commercial and withdrawable; access is not a public license.',13)

s=slide('Key Findings','A modest, targeted update gives the best measured transfer to held-out people.')
for x,value,label in [(.54,'86.18%','E1 pooled accuracy'),(3.56,'+1.20 pp','E1 improvement over E0'),(6.58,'+11.6 pp','Mask + sunglasses gain')]:
    rect(s,x,1.52,2.88,1.43,'FBFCFF'); text(s,x+.1,1.77,2.68,.55,value,28,'navy',True,align=PP_ALIGN.CENTER); text(s,x+.1,2.43,2.68,.32,label,12,'ink',True,align=PP_ALIGN.CENTER)
text(s,.74,3.42,8.5,1.17,'The new dataset targets richer occlusion variation, not greater identity diversity.\nTrain the final ArcFace block; keep the gallery and early representation stable.\nNext: independent sessions, more participants, and unseen-identity rejection tests.',15,'ink')

s=slide('References & Reproduction','Report-aligned sources and artifacts; original deck retained as the visual template.')
refs=[('Salari & Rostami · Pgu-Face · Data in Brief 9 (2016), 288–291','https://doi.org/10.1016/j.dib.2016.09.002'),('Chong et al. · Masked Face Recognition Using SVM and CNN · ICoICT 2022','https://doi.org/10.1109/ICoICT55009.2022.9914874'),('Montero et al. · Boosting Masked Face Recognition with Multi-Task ArcFace · 2022','https://doi.org/10.1109/SITIS57111.2022.00042'),('Deng et al. · ArcFace · CVPR 2019','https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html'),('Deng et al. · RetinaFace · CVPR 2020','https://openaccess.thecvf.com/content_CVPR_2020/html/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.html')]
for i,(label,url) in enumerate(refs):
    sh=text(s,.66,1.38+i*.47,8.6,.35,label,12,'navy'); sh.text_frame.paragraphs[0].runs[0].hyperlink.address=url
text(s,.66,3.94,8.6,.65,'Project artifacts: final_technical_report.md, final.pdf, metadata/e1e2_person_folds.json, runs/e1e2/. Results reproduced from saved aggregates and histories.',12,'muted')
sh=text(s,.66,4.73,8.6,.3,'Open our dataset folder (Google Drive)',13,'navy',True); sh.text_frame.paragraphs[0].runs[0].hyperlink.address='https://drive.google.com/drive/u/1/folders/1XN55OGpozy5Rhbs5QRFMatlCMTAo8sPu'

# Appendix: large confusion matrices, one model per slide rather than three tiny plots.
for e in ['e0','e1','e2']:
    s=slide(f'Appendix · {e.upper()} Confusion Matrix','30 identity columns + R for rejection; held-out queries pooled across six folds.')
    picfit(s,asset_pdf(e+'_confusion'),1.95,1.18,6.10,3.95)

# Remove stale speaker notes from the reused cover; set useful current notes everywhere.
for i,s in enumerate(P.slides):
    s.notes_slide.notes_text_frame.text=(
        'Current project sources: final_technical_report.md and final.pdf. '
        'E0 frozen; E1 last residual block; E2 last block plus embedding head. '
        'Saved pooled results are authoritative. See References & Reproduction for citations. '
        'Dataset comparison is task-specific; it is not a claim of universal superiority.'
    )
P.core_properties.title='Occlusion-Robust Face Identification — Current Project'
P.core_properties.subject='Person-disjoint six-fold evaluation of partial ArcFace fine-tuning'
P.core_properties.comments='Updated from the supplied v4 deck; no demo or deployment slides.'
deck_path=HERE/'pres.pptx'
try:
    P.save(deck_path)
except PermissionError:
    deck_path=HERE/'pres_updated.pptx'
    try:
        P.save(deck_path)
    except PermissionError:
        deck_path=HERE/'pres_simplified.pptx'
        P.save(deck_path)
print(f'Saved {deck_path.name}: {len(P.slides)} slides')

if __name__=='__main__':
    # PowerPoint is used for actual layout/render validation, not approximation.
    import win32com.client
    app=win32com.client.DispatchEx('PowerPoint.Application')
    deck=None
    try:
        deck=app.Presentations.Open(str(deck_path),True,False,False)
        deck.SaveAs(str(deck_path.with_suffix('.pdf')),32)
        rendered=OUT/'rendered'; rendered.mkdir(exist_ok=True)
        deck.Export(str(rendered),'PNG',1600,900)
        overflow=[]
        for sl in deck.Slides:
            for sh in sl.Shapes:
                if sh.HasTextFrame and sh.TextFrame.HasText:
                    tr=sh.TextFrame2.TextRange
                    if tr.BoundHeight>sh.Height+3 or tr.BoundWidth>sh.Width+3:
                        overflow.append((sl.SlideIndex,sh.Name,round(tr.BoundHeight,1),round(sh.Height,1),sh.TextFrame.TextRange.Text[:65]))
        print('PowerPoint text bounds:',overflow)
    finally:
        if deck is not None: deck.Close()
        app.Quit()
    files=sorted(rendered.glob('*.PNG'),key=lambda p:int(''.join(c for c in p.stem if c.isdigit())))
    if not files: files=sorted(rendered.glob('*.png'),key=lambda p:int(''.join(c for c in p.stem if c.isdigit())))
    for group in range(0,len(files),12):
        canvas=Image.new('RGB',(1200,4*245),'#dddddd'); draw=ImageDraw.Draw(canvas)
        for j,path in enumerate(files[group:group+12]):
            im=Image.open(path); im.thumbnail((396,223)); x=(j%3)*400; y=(j//3)*245
            canvas.paste(im,(x,y)); draw.text((x+8,y+225),f'Slide {group+j+1}',fill='black')
        canvas.save(OUT/f'contact_{group//12+1}.jpg')
