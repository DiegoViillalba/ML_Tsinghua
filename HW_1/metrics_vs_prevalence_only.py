
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def bayes_ppv_npv(sens, spec, prev):
    ppv = (sens*prev) / (sens*prev + (1-spec)*(1-prev) + 1e-12)
    npv = (spec*(1-prev)) / (spec*(1-prev) + (1-sens)*prev + 1e-12)
    return ppv, npv

def sweep_prevalence(sens=0.85, spec=0.90, steps=201):
    prev = np.linspace(1e-3, 0.999, steps)
    ppv = np.empty_like(prev)
    npv = np.empty_like(prev)
    acc = np.empty_like(prev)
    f1  = np.empty_like(prev)
    prec= np.empty_like(prev)
    rec = np.full_like(prev, sens)
    for i,p in enumerate(prev):
        ppv[i], npv[i] = bayes_ppv_npv(sens, spec, p)
        prec[i] = ppv[i]
        tn = spec*(1-p); tp = sens*p
        fp = (1-spec)*(1-p); fn = (1-sens)*p
        acc[i] = tp + tn
        f1[i]  = 2*prec[i]*rec[i]/(prec[i]+rec[i]+1e-12)
    return prev, ppv, npv, acc, prec, rec, f1

def main():
    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)
    sens, spec = 0.85, 0.90
    prev, ppv, npv, acc, prec, rec, f1 = sweep_prevalence(sens, spec)

    # Plot PPV/NPV/Accuracy vs Prevalence
    plt.figure()
    plt.plot(prev, ppv, label="PPV (Precision)")
    plt.plot(prev, npv, label="NPV")
    plt.plot(prev, acc, label="Accuracy")
    plt.xlabel("Prevalence")
    plt.ylabel("Metric value")
    plt.title(f"Metrics vs Prevalence (Sensitivity={sens}, Specificity={spec})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "metrics_vs_prevalence.png", dpi=160)
    plt.close()

    # Plot Precision/Recall/F1 vs Prevalence
    plt.figure()
    plt.plot(prev, prec, label="Precision (PPV)")
    plt.plot(prev, rec, label="Recall (Sensitivity)")
    plt.plot(prev, f1, label="F1-score")
    plt.xlabel("Prevalence")
    plt.ylabel("Metric value")
    plt.title(f"Precision/Recall/F1 vs Prevalence (Sensitivity={sens}, Specificity={spec})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "prf1_vs_prevalence.png", dpi=160)
    plt.close()

    # Print a small table at a few prevalences
    sample_prev = [0.01, 0.05, 0.10, 0.30, 0.50]
    print("Prevalence  PPV   NPV   Accuracy  Recall  F1")
    for p in sample_prev:
        ppv_p, npv_p = bayes_ppv_npv(sens, spec, p)
        tn = spec*(1-p); tp = sens*p
        acc_p = tp + tn
        rec_p = sens
        f1_p = 2*ppv_p*rec_p / (ppv_p + rec_p + 1e-12)
        print(f"{p:9.2f}  {ppv_p:0.3f} {npv_p:0.3f}   {acc_p:0.3f}    {rec_p:0.3f} {f1_p:0.3f}")

if __name__ == "__main__":
    main()
