#!/usr/bin/env python3
"""Run frozen FairVision RETFound checkpoints on an external manifest."""
from __future__ import annotations
import argparse, csv, json, re, sys
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

def args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True);p.add_argument("--dataset",required=True)
    p.add_argument("--modality",choices=("cfp","oct"),required=True);p.add_argument("--weights",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,required=True);p.add_argument("--splits",nargs="+",default=["val","test"])
    p.add_argument("--threshold",type=float,default=.5);p.add_argument("--batch-size",type=int,default=8)
    p.add_argument("--num-workers",type=int,default=2);p.add_argument("--oct-slices",type=int,default=8)
    p.add_argument("--device",default=None);p.add_argument("--max-cases",type=int,default=None)
    return p.parse_args()

def resolve(manifest,value):
    path=Path(value).expanduser()
    if path.is_absolute(): return path
    candidate=manifest.parent.parent/path
    return candidate if candidate.exists() else manifest.parent/path

def numeric_key(path):
    found=re.search(r"\d+",path.name);return int(found.group()) if found else path.name

def mhd_volume(path):
    header={}
    for line in path.read_text(errors="replace").splitlines():
        if "=" in line:
            key,value=line.split("=",1);header[key.strip()]=value.strip()
    dims=[int(x) for x in header["DimSize"].split()]
    types={"MET_UCHAR":np.uint8,"MET_CHAR":np.int8,"MET_USHORT":np.uint16,"MET_SHORT":np.int16,
           "MET_UINT":np.uint32,"MET_INT":np.int32,"MET_FLOAT":np.float32,"MET_DOUBLE":np.float64}
    dtype=np.dtype(types[header["ElementType"]])
    if header.get("ElementByteOrderMSB","False").lower()=="true": dtype=dtype.newbyteorder(">")
    raw=path.parent/header["ElementDataFile"]
    data=np.fromfile(raw,dtype=dtype)
    expected=int(np.prod(dims))
    if data.size!=expected: raise ValueError(f"{path}: expected {expected} voxels, found {data.size}")
    # MetaImage DimSize is X Y Z; NumPy volume is Z Y X.
    return data.reshape(tuple(reversed(dims)))

def oct_arrays(path,count):
    if path.is_dir():
        files=sorted((x for x in path.iterdir() if x.suffix.lower() in {".jpg",".jpeg",".png",".tif",".tiff"}),key=numeric_key)
        from PIL import Image
        volume=np.stack([np.asarray(Image.open(x).convert("L")) for x in files])
    elif path.suffix.lower()==".mhd": volume=mhd_volume(path)
    elif path.suffix.lower()==".npz":
        with np.load(path) as d:
            key=next(k for k in ("oct_bscans","volume","oct","bscans") if k in d);volume=np.asarray(d[key])
    elif path.suffix.lower()==".npy": volume=np.asarray(np.load(path))
    else: raise ValueError(f"Unsupported OCT path: {path}")
    volume=np.squeeze(volume)
    if volume.ndim!=3: raise ValueError(f"Expected [slices,height,width], got {volume.shape}: {path}")
    indices=np.linspace(0,volume.shape[0]-1,count,dtype=int)
    return [volume[i] for i in indices],indices.tolist()

def metrics(y,p,threshold):
    pred=(p>=threshold).astype(int);tn=int(((y==0)&(pred==0)).sum());fp=int(((y==0)&(pred==1)).sum())
    fn=int(((y==1)&(pred==0)).sum());tp=int(((y==1)&(pred==1)).sum());div=lambda a,b:float(a/b) if b else None
    result={"n":len(y),"threshold":threshold,"tn":tn,"fp":fp,"fn":fn,"tp":tp,"accuracy":div(tp+tn,len(y)),
            "precision":div(tp,tp+fp),"sensitivity":div(tp,tp+fn),"specificity":div(tn,tn+fp),"f1":div(2*tp,2*tp+fp+fn)}
    if len(set(y.tolist()))==2:
        from sklearn.metrics import roc_auc_score
        result["auroc"]=float(roc_auc_score(y,p))
    return result

def main():
    a=args();import torch;import torch.nn as nn
    from PIL import Image
    from torch.utils.data import DataLoader,Dataset
    from torchvision import transforms
    from VisionAgent.models_vit import RETFound_mae
    device=torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    with a.manifest.open(newline="",encoding="utf-8-sig") as f: all_rows=list(csv.DictReader(f))
    rows=[dict(r,image_path=str(resolve(a.manifest,r[f"{a.modality}_path"]))) for r in all_rows
          if r.get("dataset","").lower()==a.dataset.lower() and r.get("split","").lower() in a.splits and r.get(f"{a.modality}_path","").strip()]
    if a.max_cases: rows=rows[:a.max_cases]
    if not rows: raise SystemExit("No matching manifest rows")
    tfm=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    class DS(Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self,i):
            r=rows[i];path=Path(r["image_path"])
            if a.modality=="cfp": image=tfm(Image.open(path).convert("RGB"));indices=[]
            else:
                arrays,indices=oct_arrays(path,a.oct_slices);ims=[]
                for x in arrays:
                    x=np.asarray(x);lo,hi=float(np.nanmin(x)),float(np.nanmax(x))
                    x=np.zeros(x.shape,dtype=np.uint8) if hi<=lo else ((x-lo)/(hi-lo)*255).astype(np.uint8)
                    ims.append(tfm(Image.fromarray(x).convert("RGB")))
                image=torch.stack(ims)
            return image,float(r["label"]),r["case_id"],r["split"],json.dumps(indices)
    class Model(nn.Module):
        def __init__(self):
            super().__init__();self.backbone=RETFound_mae(img_size=224,num_classes=0,drop_path_rate=.2,global_pool="")
            self.amd_head=nn.Sequential(nn.Linear(1024,256),nn.ReLU(),nn.Dropout(.25),nn.Linear(256,1))
            self.dr_head=nn.Sequential(nn.Linear(1024,256),nn.ReLU(),nn.Dropout(.25),nn.Linear(256,1))
            self.glaucoma_head=nn.Sequential(nn.Linear(1024,256),nn.ReLU(),nn.Dropout(.25),nn.Linear(256,1))
        def forward(self,x):
            if x.ndim==5:
                b,s,c,h,w=x.shape;features=self.backbone(x.reshape(b*s,c,h,w)).reshape(b,s,-1).mean(1)
            else: features=self.backbone(x)
            return self.glaucoma_head(features).squeeze(1)
    model=Model()
    state=torch.load(a.weights,map_location="cpu",weights_only=False);state=state.get("model",state)
    if a.modality=="cfp":
        # CFP specialist checkpoints contain only the glaucoma head.
        model_state=model.state_dict();model_state.update(state);model.load_state_dict(model_state,strict=True)
    else: model.load_state_dict(state,strict=True)
    model.to(device).eval();loader=DataLoader(DS(),batch_size=a.batch_size,shuffle=False,num_workers=a.num_workers)
    output=[]
    with torch.inference_mode():
        for images,labels,ids,splits,indices in loader:
            probs=torch.sigmoid(model(images.to(device))).cpu().numpy()
            for case,split,label,prob,index in zip(ids,splits,labels.numpy(),probs,indices):
                output.append({"dataset":a.dataset,"case_id":case,"split":split,"y_true":int(label),"y_prob":float(prob),
                    "threshold":a.threshold,"y_pred":int(prob>=a.threshold),"model":a.weights.name,"modality":a.modality,"slice_indices":index})
    a.out_dir.mkdir(parents=True,exist_ok=True)
    with (a.out_dir/"predictions.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    summary={"dataset":a.dataset,"modality":a.modality,"weights":str(a.weights),"device":str(device),"rows":len(output),
             "splits":dict(Counter(x["split"] for x in output)),"metrics":{}}
    for split in sorted(set(x["split"] for x in output)):
        group=[x for x in output if x["split"]==split]
        summary["metrics"][split]=metrics(np.array([x["y_true"] for x in group]),np.array([x["y_prob"] for x in group]),a.threshold)
    (a.out_dir/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=="__main__": main()
