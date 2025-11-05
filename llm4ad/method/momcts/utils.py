import json
import os
from datetime import datetime
from typing import List
from ...base import *

def save_json(iteration, hv_score, long_term_guide, good_reflections, bad_reflections, save_dir="logs"):
   
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "reflections_log.json")

    log_entry = {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "current_hv": hv_score,
        "long_term_guide": long_term_guide,
        "recent_good_reflections": good_reflections[-5:],
        "recent_bad_reflections": bad_reflections[-5:]
    }

    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    data.append(log_entry)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Reflections saved to {save_path}")


def dominates(objective_a: List[float], objective_b: List[float]) -> bool:
   
    is_strictly_better = False
    for i in range(len(objective_a)):
        if objective_a[i] > objective_b[i]:
            return False
        if objective_a[i] < objective_b[i]:
            is_strictly_better = True
    return is_strictly_better

def hypervolume_contribution(objective: List[float], pareto_front: List[Function]) -> float:
    if not pareto_front:
        return 0.0
    distances = [
        sum([(objective[i] - other.score[i])**2 for i in range(len(objective))])**0.5
        for other in pareto_front
    ] # tinh khoang cach cua objective hien tai voi cac diem trong pareto_front, chon thang be nhat
    return min(distances)

