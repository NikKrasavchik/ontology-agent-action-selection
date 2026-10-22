from __future__ import annotations

import csv
import random
from pathlib import Path


OUT = Path(__file__).resolve().parent
SEED = 20261020
RNG = random.Random(SEED)
ACTIONS = ("query", "adapt", "abstain")
CLASSES = ("facts", "structure", "noise", "task")


def clamp(value, lo=0.02, hi=0.98):
    return max(lo, min(hi, value))


def make_scenario(index, cls, uncertainty, query_cost, adapt_risk):
    u = int(uncertainty == "high")
    qcost = 0.22 if query_cost == "low" else 0.58
    risk = 0.16 if adapt_risk == "low" else 0.55
    if cls == "facts":
        p_q, p_a, p_s = 0.88 - 0.14 * u, 0.28 - 0.06 * u, 0.42 - 0.08 * u
    elif cls == "structure":
        p_q, p_a, p_s = 0.43 - 0.08 * u, 0.88 - 0.14 * u, 0.38 - 0.08 * u
    elif cls == "noise":
        p_q, p_a, p_s = 0.50 - 0.12 * u, 0.22 - 0.08 * u, 0.88 - 0.10 * u
    else:
        p_q, p_a, p_s = 0.34 - 0.08 * u, 0.18 - 0.05 * u, 0.82 - 0.10 * u
    true = {"query": clamp(p_q), "adapt": clamp(p_a), "abstain": clamp(p_s)}
    estimated = {
        f"estimated_p_{action}": clamp(prob + RNG.uniform(-0.18, 0.18))
        for action, prob in true.items()
    }
    return {
        "id": f"S{index:02d}",
        "class": cls,
        "uncertainty": uncertainty,
        "query_cost": qcost,
        "adapt_risk": risk,
        "true_p_query": true["query"],
        "true_p_adapt": true["adapt"],
        "true_p_abstain": true["abstain"],
        **estimated,
    }


def action_cost(scenario, action):
    return scenario["query_cost"] if action == "query" else 0.18 if action == "adapt" else 0.06


def utility(scenario, action, estimated=False, include_cost=True, include_risk=True):
    prefix = "estimated_p_" if estimated else "true_p_"
    probability = scenario[prefix + action]
    cost = action_cost(scenario, action) if include_cost else 0.0
    penalty = scenario["adapt_risk"] * 0.80 if action == "adapt" and include_risk else 0.0
    return probability - cost - penalty


def choose(scenario, method, include_cost=True, include_risk=True, allow_abstain=True):
    if method == "M1":
        pool = ACTIONS if allow_abstain else ("query", "adapt")
        return max(pool, key=lambda action: utility(scenario, action, True, include_cost, include_risk))
    if method == "M2":
        return {"facts": "query", "structure": "adapt", "noise": "abstain", "task": "abstain"}[scenario["class"]]
    if method == "M3":
        return choose(scenario, "M1", include_cost, include_risk, False)
    if method == "M4":
        return "query"
    raise ValueError(method)


scenarios = []
for cls in CLASSES:
    for _ in range(12):
        scenarios.append(
            make_scenario(
                len(scenarios) + 1,
                cls,
                RNG.choice(("low", "high")),
                RNG.choice(("low", "high")),
                RNG.choice(("low", "high")),
            )
        )

methods = ("M1", "M2", "M3", "M4")
rows = []
for scenario in scenarios:
    true_best = max(ACTIONS, key=lambda action: utility(scenario, action))
    for method in methods:
        action = choose(scenario, method)
        rows.append(
            {
                "scenario": scenario["id"],
                "class": scenario["class"],
                "method": method,
                "action": action,
                "true_utility": utility(scenario, action),
                "true_best_action": true_best,
                "cost": action_cost(scenario, action),
                # Evaluation uses the hidden true utilities, not the class label.
                "unjustified_adaptation": int(action == "adapt" and true_best != "adapt"),
            }
        )


def summarise(result_rows):
    summary = {}
    for method in methods:
        selected = [row for row in result_rows if row["method"] == method]
        summary[method] = {
            "mean_utility": sum(row["true_utility"] for row in selected) / len(selected),
            "unjustified_adaptations": sum(row["unjustified_adaptation"] for row in selected) / len(selected),
            "mean_cost": sum(row["cost"] for row in selected) / len(selected),
            "action_accuracy": sum(row["action"] == row["true_best_action"] for row in selected) / len(selected),
        }
    return summary


summary = summarise(rows)

with (OUT / "эксперимент_сценарии.csv").open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=list(scenarios[0]))
    writer.writeheader()
    writer.writerows(scenarios)

with (OUT / "эксперимент_результаты.csv").open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

with (OUT / "эксперимент_сводка.csv").open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow(["method", "mean_utility", "unjustified_adaptations", "mean_cost", "action_accuracy"])
    for method in methods:
        writer.writerow([method] + [round(summary[method][key], 4) for key in ("mean_utility", "unjustified_adaptations", "mean_cost", "action_accuracy")])

grid = []
for query_cost in (0.10, 0.25, 0.40, 0.55, 0.70):
    for adaptation_risk in (0.10, 0.25, 0.40, 0.55, 0.70):
        altered = []
        for scenario in scenarios:
            copy = dict(scenario)
            copy["query_cost"] = query_cost
            copy["adapt_risk"] = adaptation_risk
            altered.append(copy)
        chosen = [choose(scenario, "M1") for scenario in altered]
        n = len(chosen)
        grid.append((query_cost, adaptation_risk, chosen.count("query") / n, chosen.count("adapt") / n, chosen.count("abstain") / n))

with (OUT / "эксперимент_чувствительность.csv").open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow(["query_cost", "adaptation_risk", "query_share", "adapt_share", "abstain_share"])
    writer.writerows(grid)

print("SUMMARY")
for method in methods:
    print(method, {key: round(value, 4) for key, value in summary[method].items()})
