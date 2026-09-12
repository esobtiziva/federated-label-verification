import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=".")
    ap.add_argument("--outdir",default="final_v3_five_seed_combined")
    a=ap.parse_args()
    root=Path(a.root); out=root/a.outdir; out.mkdir(exist_ok=True)
    SS=[]; PP=[]
    for seed in range(1,6):
        f=root/f"final_v3_seed{seed}"
        sp=f/"summary_mean_sd.csv"; pp=f/"paired_effects.csv"
        if not sp.exists() or not pp.exists():
            raise FileNotFoundError(f"Missing summary files in {f}. Run summarize_results.py --root {f.name}")
        s=pd.read_csv(sp); p=pd.read_csv(pp)
        s["seed"]=seed
        if "seed" not in p.columns: p["seed"]=seed
        SS.append(s); PP.append(p)
    s=pd.concat(SS,ignore_index=True); p=pd.concat(PP,ignore_index=True)
    s.to_csv(out/"combined_seed_summaries.csv",index=False)
    p.to_csv(out/"paired_effects_all_five_seeds.csv",index=False)

    rows=[]
    keys=["partition","clients","condition","gamma"]
    metrics=["mse_mean","r2_mean","reviewed_mean","fixed_mean","precision_mean","recall_mean","sigma_actual_mean"]
    for k,g in s.groupby(keys,dropna=False):
        r=dict(zip(keys,k)); r["n_seeds"]=g.seed.nunique()
        for m in metrics:
            if m in g:
                x=pd.to_numeric(g[m],errors="coerce")
                base=m[:-5] if m.endswith("_mean") else m
                r[base+"_mean"]=x.mean(); r[base+"_sd"]=x.std(ddof=1)
        rows.append(r)
    fs=pd.DataFrame(rows); fs.to_csv(out/"five_seed_summary_mean_sd.csv",index=False)

    rows=[]; keys=["partition","clients","gamma"]
    metrics=["baseline_r2","noise_r2","crowd_r2","noise_delta_r2","crowd_gain_r2",
             "baseline_mse","noise_mse","crowd_mse","noise_delta_mse","crowd_gain_mse",
             "reviewed","fixed","selection_precision","correction_recall","noise_sigma_actual"]
    for k,g in p.groupby(keys,dropna=False):
        r=dict(zip(keys,k)); r["n_seeds"]=g.seed.nunique()
        for m in metrics:
            if m in g:
                x=pd.to_numeric(g[m],errors="coerce")
                r[m+"_mean"]=x.mean(); r[m+"_sd"]=x.std(ddof=1)
        rows.append(r)
    pa=pd.DataFrame(rows); pa.to_csv(out/"five_seed_paired_effects_summary.csv",index=False)

    crowd=[c for c in ["partition","clients","gamma","n_seeds","reviewed_mean","reviewed_sd",
      "fixed_mean","fixed_sd","selection_precision_mean","selection_precision_sd",
      "correction_recall_mean","correction_recall_sd"] if c in pa.columns]
    pa[crowd].to_csv(out/"five_seed_crowd_summary.csv",index=False)

    show=[c for c in ["partition","clients","gamma","n_seeds","baseline_r2_mean","noise_r2_mean",
      "crowd_r2_mean","noise_delta_r2_mean","noise_delta_r2_sd","crowd_gain_r2_mean",
      "crowd_gain_r2_sd","selection_precision_mean","correction_recall_mean"] if c in pa.columns]
    print("\n=== FINAL FIVE-SEED PAIRED EFFECTS SUMMARY ===")
    print(pa[show].to_string(index=False))
    print("\nSaved outputs to:",out.resolve())
if __name__=="__main__": main()
