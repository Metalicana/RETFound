#!/usr/bin/env python3
"""Run the CFP-viewing glaucoma agent with precomputed external tool evidence."""
from __future__ import annotations
import argparse, base64, csv, io, json, os, re, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))

def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset",required=True);p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--probabilities",type=Path,required=True);p.add_argument("--cdr",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True);p.add_argument("--split",default="test")
    p.add_argument("--max-cases",type=int,default=0);p.add_argument("--deployment",default=os.getenv("AZURE_OPENAI_DEPLOYMENT","gpt-5.1"))
    return p.parse_args()

def read(path):
    with path.open(newline="",encoding="utf-8-sig") as f:return list(csv.DictReader(f))

def resolve(manifest,value):
    p=Path(value).expanduser()
    if p.is_absolute():return p
    candidate=manifest.parent.parent/p
    return candidate if candidate.exists() else manifest.parent/p

def data_url(path):
    from PIL import Image
    image=Image.open(path).convert("RGB");buffer=io.BytesIO();image.save(buffer,format="JPEG",quality=95)
    return "data:image/jpeg;base64,"+base64.b64encode(buffer.getvalue()).decode("ascii")

def parse_label(value):
    match=re.search(r"GLAUCOMA_DETECTED:\s*(-?\d+)",value or "",re.I)
    return int(match.group(1)) if match else -1

def main():
    a=arguments()
    from openai import AzureOpenAI
    from CounterfactualAgent.counterfactual_cfp import CounterfactualCFPAgent
    from Orchestrator.drishti import DrishtiOrchestrator
    client=AzureOpenAI(azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                       api_version=os.getenv("AZURE_OPENAI_API_VERSION","2024-12-01-preview"))
    probabilities={r["case_id"]:r for r in read(a.probabilities) if r.get("split","").lower()==a.split.lower()}
    cdr={r["case_id"]:r for r in read(a.cdr)}
    cases=[r for r in read(a.manifest) if r.get("dataset","").lower()==a.dataset.lower() and r.get("split","").lower()==a.split.lower()]
    if a.max_cases:cases=cases[:a.max_cases]
    missing_probability=[r["case_id"] for r in cases if r["case_id"] not in probabilities]
    missing_cdr=[r["case_id"] for r in cases if r["case_id"] not in cdr]
    if missing_probability or missing_cdr:raise SystemExit(f"Join failure: probability={missing_probability[:5]}, cdr={missing_cdr[:5]}")
    a.out_dir.mkdir(parents=True,exist_ok=True);output=a.out_dir/"predictions.csv"
    rows=read(output) if output.exists() else [];completed={r["case_id"] for r in rows if r.get("Pred_GL") in {"0","1"}}
    counterfactual=CounterfactualCFPAgent(cache_path=a.out_dir/"counterfactual_traces.jsonl")
    orchestrator=DrishtiOrchestrator()
    for index,case in enumerate(cases,start=1):
        case_id=case["case_id"]
        if case_id in completed:
            print(f"skip {index}/{len(cases)} case={case_id}",flush=True);continue
        probability=float(probabilities[case_id]["agent_probability_percent"])
        cdr_text=cdr[case_id].get("vertical_cdr","").strip();vertical_cdr=float(cdr_text) if cdr_text else None
        image_path=resolve(a.manifest,case["cfp_path"])
        try:
            response=client.chat.completions.create(model=a.deployment,temperature=.2,messages=[
                {"role":"system","content":"You are an ophthalmic imaging specialist reviewing one color fundus photograph for glaucoma. Assess gradability, vertical cupping, neuroretinal-rim thinning or notching, superior-inferior asymmetry, vessel displacement or bayoneting, laminar dots, disc hemorrhage, RNFL defects, and peripapillary atrophy. Do not invent a numerical cup-to-disc ratio and do not infer findings from any AI score. End with IMPRESSION: supports glaucoma, supports normal, or indeterminate, and name the features driving it."},
                {"role":"user","content":[{"type":"text","text":"Analyze this CFP for glaucoma-related structural findings."},{"type":"image_url","image_url":{"url":data_url(image_path)}}]}])
            report=response.choices[0].message.content
            audit=counterfactual.analyze(case_id=f"{a.dataset}:{case_id}",retfound_probability=probability,cfp_report=report,cdr=vertical_cdr)
            trace=counterfactual.concise_trace(audit)
            final=orchestrator.analyze(probability,report,vertical_cdr,trace)
            prediction=parse_label(final.get("labels",""))
            row={"dataset":a.dataset,"case_id":case_id,"split":a.split,"Ground_Truth":int(float(case["label"])),
                 "Raw_RETFound_Probability":probabilities[case_id]["raw_probability"],"Agent_RETFound_Probability_Pct":probability,
                 "Vertical_CDR":"" if vertical_cdr is None else vertical_cdr,"CFP_Report":report,
                 "Counterfactual_Trace":json.dumps(trace,sort_keys=True),"Agentic_Decision":final.get("decision",""),
                 "Pred_GL":prediction,"Is_Correct":int(prediction==int(float(case["label"]))) if prediction in (0,1) else -1,"error":""}
        except Exception as exc:
            row={"dataset":a.dataset,"case_id":case_id,"split":a.split,"Ground_Truth":case["label"],
                 "Raw_RETFound_Probability":probabilities[case_id]["raw_probability"],"Agent_RETFound_Probability_Pct":probability,
                 "Vertical_CDR":"" if vertical_cdr is None else vertical_cdr,"CFP_Report":"","Counterfactual_Trace":"",
                 "Agentic_Decision":"","Pred_GL":-1,"Is_Correct":-1,"error":repr(exc)}
        rows.append(row)
        with output.open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        print(f"done {index}/{len(cases)} case={case_id} pred={row['Pred_GL']} correct={row['Is_Correct']}",flush=True)
    valid=[r for r in rows if r.get("split")==a.split and str(r.get("Pred_GL")) in {"0","1"}]
    if valid:
        y=[int(float(r["Ground_Truth"])) for r in valid];p=[int(r["Pred_GL"]) for r in valid]
        tn=sum(x==0 and z==0 for x,z in zip(y,p));fp=sum(x==0 and z==1 for x,z in zip(y,p));fn=sum(x==1 and z==0 for x,z in zip(y,p));tp=sum(x==1 and z==1 for x,z in zip(y,p))
        print(json.dumps({"valid":len(valid),"accuracy":(tn+tp)/len(valid),"f1":2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,"tn":tn,"fp":fp,"fn":fn,"tp":tp,"output":str(output)},indent=2))

if __name__=="__main__":main()
