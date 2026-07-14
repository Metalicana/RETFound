#!/usr/bin/env python3
"""Move a validation-selected probability threshold to 0.5 for agent input."""
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path

def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--threshold",type=float,required=True)
    p.add_argument("--probability-column",default="y_prob")
    return p.parse_args()

def logit(value):
    value=min(max(float(value),1e-7),1-1e-7)
    return math.log(value/(1-value))

def main():
    a=arguments()
    if not 0<a.threshold<1: raise SystemExit("threshold must be between 0 and 1")
    rows=[]
    for path in a.input:
        with path.open(newline="",encoding="utf-8-sig") as handle:
            rows.extend(csv.DictReader(handle))
    if not rows: raise SystemExit("No input rows")
    boundary=logit(a.threshold)
    mismatches=0
    for row in rows:
        raw=float(row[a.probability_column])
        adjusted=1/(1+math.exp(-(logit(raw)-boundary)))
        raw_prediction=int(raw>=a.threshold)
        adjusted_prediction=int(adjusted>=.5)
        mismatches+=raw_prediction!=adjusted_prediction
        row["raw_probability"]=raw
        row["calibration_threshold"]=a.threshold
        row["agent_probability"]=adjusted
        row["agent_probability_percent"]=adjusted*100
        row["agent_prediction_0_5"]=adjusted_prediction
        row["normalization"]="logit_boundary_shift"
    if mismatches: raise SystemExit(f"Decision-preservation check failed for {mismatches} rows")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={"inputs":[str(x) for x in a.input],"output":str(a.output),"rows":len(rows),
             "threshold":a.threshold,"normalization":"sigmoid(logit(raw)-logit(threshold))",
             "decision_mismatches":mismatches}
    a.output.with_suffix(".summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=="__main__": main()
