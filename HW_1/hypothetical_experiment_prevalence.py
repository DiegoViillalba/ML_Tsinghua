
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def bayes_ppv_npv(sens, spec, prev):
    ppv = (sens*prev) / (sens*prev + (1-spec)*(1-prev) + 1e-12)
    npv = (spec*(1-prev)) / (spec*(1-prev) + (1-sens)*prev + 1e-12)
    return ppv, npv

def expected_utility(sens, spec, prev, benefit_tp=1.0, cost_fp=0.05, cost_fn=1.0, benefit_tn=0.0):
    # Outcome probabilities
    p_tp = sens*prev
    p_fn = (1-sens)*prev
    p_tn = spec*(1-prev)
    p_fp = (1-spec)*(1-prev)
    # Utility of "test-and-treat if positive"
    util_test = p_tp*benefit_tp - p_fp*cost_fp - p_fn*cost_fn + p_tn*benefit_tn
    # Utility of "do nothing" (never treat): all positives become FNs
    util_never = - prev*cost_fn
    return util_test, util_never

def run_experiment():
    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)
    sens, spec = 0.90, 0.90  # same test performance across settings

    # Utility parameters (edit to your domain)
    benefit_tp = 1.0
    cost_fp = 0.05
    cost_fn = 1.0
    benefit_tn = 0.0

    prevalences = [0.01, 0.30]  # Low vs High prevalence settings
    policies = ["Test & Treat if +", "Never Treat"]
    table = []

    for prev in prevalences:
        u_test, u_never = expected_utility(sens, spec, prev, benefit_tp, cost_fp, cost_fn, benefit_tn)
        table.append((prev, u_test, u_never))

    # Print table
    print("=== Same Sens/Spec, different prevalence → different optimal decision ===")
    print(f"Sensitivity={sens}, Specificity={spec}, benefits/costs: TP={benefit_tp}, FP={cost_fp}, FN={cost_fn}, TN={benefit_tn}")
    print("Prevalence    U(Test&Treat+)    U(NeverTreat)    Better Policy")
    for prev, u_test, u_never in table:
        better = policies[0] if u_test > u_never else policies[1]
        print(f"{prev:10.2%}    {u_test:14.4f}    {u_never:13.4f}    {better}")

    # Bar chart
    x = np.arange(len(policies))
    width = 0.35
    plt.figure()
    for i, (prev, u_test, u_never) in enumerate(table):
        utils = [u_test, u_never]
        plt.bar(x + (i-0.5)*width, utils, width, label=f"Prevalence={prev:.0%}")
    plt.xticks(x, policies)
    plt.ylabel("Expected utility per person")
    plt.title("Policy choice flips with prevalence (Sens=0.90, Spec=0.90)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "decision_utilities_prevalence.png", dpi=160)
    plt.close()

if __name__ == '__main__':
    run_experiment()
