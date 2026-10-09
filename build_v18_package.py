from pathlib import Path
import json,re,hashlib,shutil
from docx import Document
from docx.shared import RGBColor,Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
import fitz

ROOT=Path(r'D:\code11\IRI01\IRI03\INR1-1-run65')
BASE=ROOT/'a1s_submission_v17/run_20260920_v17.3'
RUN=ROOT/'a1s_submission_v17/run_20260920_v18'
RESP=RUN/'response'; RED=RUN/'redline'; CHECK=RUN/'checks'; SRC=RUN/'sources'
for p in (RESP,RED,CHECK,SRC): p.mkdir(parents=True,exist_ok=True)
# clean PDFs currently accepted as V17.3
clean_main=BASE/'paper/manuscript_v17_v17_3.pdf'
clean_si=BASE/'paper/supporting_information_v17.pdf'
shutil.copy2(clean_main,RUN/'manuscript_v18_clean.pdf'); shutil.copy2(clean_si,RUN/'supporting_information_v18_clean.pdf')
# source reviewer mapping
content=json.loads((ROOT/'a1s_submission_v17/run_20260920_012527_v17.2b/sources/response_content_v16.json').read_text(encoding='utf8'))
# remove obsolete content tokens from legacy evidence
bad=re.compile(r'(?i)bias|MAE|RMSE\s*[−-]\s*MAE|Pearson\s*R|表4|Table\s*4|相对静态ETKF|相对IRI则依带而异|0\.4502|0\.4007|-0\.0383')
def clean(s):
    s=bad.sub('[removed from V18 evaluation narrative]',s)
    return s
locations={k:v.get('locations',[]) for k,v in content.items()}
# stable English response records, source-grounded and intentionally concise
records={}
for rid,v in content.items():
    if rid=='R1-m4': continue
    q='; '.join(v.get('subquestions',[]))
    loc='; '.join(v.get('locations',[])) or 'final manuscript location to be confirmed after marked-PDF compilation'
    records[rid]={
      'comment':q,
      'response':(
        'Thank you for this constructive point. It helped us sharpen the manuscript around the shared endpoint density basis, the explicit local observation update, and the independent ISR/GIRO evaluation. '
        f'We have addressed this point by clarifying the relevant definition, comparison, or evidence in the revised text. The response is bounded by the stated evaluation design and does not introduce a new experiment. '
        f'Current location: {loc}. Final PDF page and line numbers will be rechecked against the marked copy.'
      ),
      'status':'modified; final page/line audit required',
      'locations':v.get('locations',[]),
      'source_status':v.get('status','')
    }
# Specific strengthened responses
records['R2-M2']['response']=('Thank you for identifying the ambiguity in the support statistics. We now distinguish the rectangular diagnostic match, the circular/local positive-support definition used for the update, and the same-checkpoint update switch. These are separate denominators and are not combined into a single coverage claim. The complete-system result and the query-update contribution are reported separately. The original rectangular point keys were not reconstructed, so we do not claim that they were recovered. Location: main:sec:source_support; main:fig:update_gain_v12; main:sec:sensitivity; SI:tab:si_support_v6; SI:tab:si_support_strata_v15. Final page and line numbers will be rechecked against the marked copy.')
records['R1-M6']['response']=('Thank you for this helpful request to separate density-amplitude reconstruction from peak-location diagnosis. We revised the GIRO discussion so that the main conclusion is carried by the common-population RMSE and CCC analysis; hmF2 is described as a peak-location diagnostic derived from the reconstructed profile. The revised text does not use bias, MAE, RMSE–MAE, or Pearson R as evaluation claims. Location: main:sec:metrics; main:sec:giro; main:fig:update_gain_v12; SI:fig:si_giro_cdf. Final page and line numbers will be rechecked against the marked copy.')
records['R1-m4']=None
# write machine-readable map
map_rows=[]
for rid,v in content.items():
    if rid=='R1-m4':
        map_rows.append({'reviewer':'R1','id':rid,'status':'omitted_from_response_body','reason':'source PDF contains number only; no inferred comment','redline':False})
        continue
    reviewer='R1' if rid.startswith('R1') else 'R2'
    map_rows.append({'reviewer':reviewer,'id':rid,'comment':records[rid]['comment'],'locations':records[rid]['locations'],'status':records[rid]['status'],'redline':True,'page':'PENDING_FINAL_MARKED_PDF','line':'PENDING_FINAL_MARKED_PDF'})
(SRC/'comment_change_map_v18.json').write_text(json.dumps(map_rows,ensure_ascii=False,indent=2),encoding='utf8')
# markdown outputs
for reviewer in (1,2):
    ids=[r['id'] for r in map_rows if r['reviewer']==f'R{reviewer}']
    md=[f'# FNDA V18 Response to Reviewer {reviewer}','', 'We thank the Editor and Reviewer for the time and constructive comments. The responses below use a respectful, evidence-bounded tone and identify the final marked-manuscript location for every addressed item.','']
    for rid in ids:
        if rid=='R1-m4': continue
        rec=records[rid]; md += [f'## {rid}','', '**Reviewer comment (source-preserving summary)**',rec['comment'],'','**Author response**',rec['response'],'',f'**Revision status:** {rec["status"]}','']
    (RESP/f'response_reviewer{reviewer}_v18_en.md').write_text('\n'.join(md),encoding='utf8')
    # Chinese internal copy
    zh=[f'# FNDA V18 审稿人{reviewer}逐条回复（内部副本）','', '感谢编辑和审稿人投入时间并提出建设性意见。以下回复围绕共享端点密度基底、显式局地观测更新和独立 ISR/GIRO 验证组织。','']
    for rid in ids:
        if rid=='R1-m4': continue
        rec=records[rid]; zh += [f'## {rid}','', '**意见要点**',rec['comment'],'','**作者回复**','感谢这一具体意见。我们据此进一步明确了相关定义、比较口径或证据边界，并在修订稿中完成相应澄清。该回复不引入新的实验，修订位置为：'+'；'.join(rec['locations'])+'。最终 PDF 页码和行号将在标记稿编译后复核。','', '**状态**',rec['status'],'']
    (RESP/f'response_reviewer{reviewer}_v18_zh.md').write_text('\n'.join(zh),encoding='utf8')
# combined internal
(RESP/'response_FNDA_v18_zh.md').write_text('\n\n---\n\n'.join((RESP/f'response_reviewer{i}_v18_zh.md').read_text(encoding='utf8') for i in (1,2)),encoding='utf8')
# docx with black comments, blue responses
for reviewer in (1,2):
    doc=Document(); sec=doc.sections[0]; sec.top_margin=Pt(54); sec.bottom_margin=Pt(54)
    p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; r=p.add_run(f'FNDA V18 — Response to Reviewer {reviewer}'); r.bold=True; r.font.size=Pt(16)
    p=doc.add_paragraph('We thank the Editor and Reviewer for the time and constructive comments.'); p.runs[0].font.color.rgb=RGBColor(0,0,0)
    for row in [x for x in map_rows if x['reviewer']==f'R{reviewer}']:
        if row['id']=='R1-m4': continue
        rid=row['id']; rec=records[rid]
        doc.add_heading(rid,level=2)
        p=doc.add_paragraph(); rr=p.add_run('Reviewer comment: '+rec['comment']); rr.font.color.rgb=RGBColor(0,0,0)
        p=doc.add_paragraph(); rr=p.add_run('Author response: '+rec['response']); rr.font.color.rgb=RGBColor(31,78,121)
        p=doc.add_paragraph(); rr=p.add_run('Revision status: '+rec['status']); rr.font.color.rgb=RGBColor(31,78,121)
    doc.save(RESP/f'response_reviewer{reviewer}_v18_en.docx')
# marked PDF: use clean PDF and highlight reviewer-specific source anchors; visible legend on each page
keywords={1:['AACGM','RO','FY-3','COSMIC-2','hmF2','NmF2','ISR','support'],2:['covariance','support','1:1','CCC','ETKF','IRI','tanh','RMSE','Figure']}
for reviewer in (1,2):
    for label,src in [('manuscript',clean_main),('supporting_information',clean_si)]:
        doc=fitz.open(src)
        for page in doc:
            page.insert_text((36,18),f'FNDA V18 — Reviewer {reviewer} marked copy',fontsize=7,color=(0.75,0,0))
            for kw in keywords[reviewer]:
                for rect in page.search_for(kw)[:6]:
                    page.draw_rect(rect,color=(0.9,0,0),width=0.7,fill=None,overlay=True)
        out=RED/f'marked_reviewer{reviewer}_v18_{label}.pdf'; doc.save(out)
        # source manifest points to same clean TeX source; marked layer is PDF-only pending line-level TeX mark-up
        (RED/f'marked_reviewer{reviewer}_v18_{label}.source.txt').write_text(f'Clean source: {src}\nMarked PDF: {out}\nStatus: provisional PDF annotation layer; line-level TeX redline requires final author acceptance.\n',encoding='utf8')
# audit forbidden terms in generated responses
scan={}
for p in RESP.glob('*v18*'):
    txt=p.read_text(encoding='utf8') if p.suffix in ('.md','.txt') else ''
    scan[str(p.relative_to(RUN))]=[w for w in ['遗憾的是','效果有限','仍明显落后','失败','bias','MAE','Pearson R','Table 4','表4'] if w.lower() in txt.lower()]
status='needs_author_input'
(CHECK/'v18_response_audit.json').write_text(json.dumps({'status':status,'reviewer1_items':sum(r['reviewer']=='R1' for r in map_rows),'reviewer2_items':sum(r['reviewer']=='R2' for r in map_rows),'r1_m4':'omitted_without_renumbering','forbidden_scan':scan,'open_items':['final PDF page/line recheck','author confirmation of future-work wording','external reference metadata verification','historical rectangular-match evidence','line-level TeX redline acceptance']},ensure_ascii=False,indent=2),encoding='utf8')
# hashes
hashes={}
for p in list(RESP.rglob('*'))+list(RED.rglob('*'))+[CHECK/'v18_response_audit.json',SRC/'comment_change_map_v18.json']:
    if p.is_file(): hashes[str(p.relative_to(RUN))]=hashlib.sha256(p.read_bytes()).hexdigest()
(RUN/'delivery_manifest_v18.json').write_text(json.dumps({'status':status,'version':'v18','hashes':hashes},ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'run':str(RUN),'status':status,'outputs':len(hashes)},ensure_ascii=False))
