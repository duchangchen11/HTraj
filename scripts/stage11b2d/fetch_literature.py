"""Download only public author/publisher papers into ignored CPU evidence cache."""
from .common import *
from concurrent.futures import ThreadPoolExecutor
import requests
import time

SOURCES={
 'YNet':['https://arxiv.org/pdf/2012.01526'],
 'PECNet':['https://arxiv.org/pdf/2004.02025'],
 'DLow':['https://arxiv.org/pdf/2003.08386'],
 'AgentFormer':['https://arxiv.org/pdf/2103.14023'],
 'TUTR':['https://openaccess.thecvf.com/content/ICCV2023/papers/Shi_Trajectory_Unified_Transformer_for_Pedestrian_Trajectory_Prediction_ICCV_2023_paper.pdf'],
 'DPP':['https://arxiv.org/pdf/1907.04967','https://www.ri.cmu.edu/app/uploads/2020/04/Yuan_DPP_forecast.pdf'],
 'DenseTNT':['https://arxiv.org/pdf/2108.09640'],
 'NPSN':['https://arxiv.org/pdf/2203.13471'],
 'HYPER':['https://arxiv.org/pdf/2110.02344','https://mers-papers.csail.mit.edu/Conference/2022/ICRA_Huang/Xin_Huang_ICRA22.pdf'],
 'LRVQ2024':['https://arxiv.org/pdf/2405.20743'],
 'GoalFlow2025':['https://arxiv.org/pdf/2503.05689'],
 'GMM2026':['https://arxiv.org/pdf/2606.26424'],
 'SIPTraj2026':['https://arxiv.org/pdf/2608.00779']}

def fetch(item):
    key,urls=item;records=[]
    for url in urls:
        start=time.perf_counter()
        try:
            r=requests.get(url,timeout=35)
            row=dict(paper=key,url=url,status=r.status_code,bytes=len(r.content),seconds=time.perf_counter()-start)
            if r.status_code==200 and r.content.startswith(b'%PDF'):
                p=CACHE/'literature'/(key+'.pdf');p.parent.mkdir(parents=True,exist_ok=True)
                with p.open('xb') as f:f.write(r.content)
                text=p.with_suffix('.txt');subprocess.run(['pdftotext','-layout',str(p),str(text)],check=True)
                row.update(PDF_SHA=sha(p),text_SHA=sha(text),text_chars=len(text.read_text()),evidence='FULLTEXT_AVAILABLE_NOT_YET_REVIEWED')
                records.append(row);break
            row['evidence']='FETCH_FAILED_OR_NONPDF';records.append(row)
        except Exception as ex:
            records.append(dict(paper=key,url=url,evidence='FETCH_FAILED',error=type(ex).__name__+': '+str(ex)))
    return records

def main():
    cpu_guard()
    with ThreadPoolExecutor(max_workers=4) as pool:rows=sum(list(pool.map(fetch,SOURCES.items())),[])
    jnew(OUT/'literature_fetch.json',dict(MCP_status='academic-search MCP tools not mounted; primary web search and public fulltext fallback',
         records=rows,source_version=commit(),CPU_ONLY=True))
    print(json.dumps(rows,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
