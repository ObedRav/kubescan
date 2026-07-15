#!/usr/bin/env python3
"""Compute usability-study results from plantilla_resultados.csv.

Fill the CSV with real data (SUS1..SUS10 on a 1-5 scale, A1..A4 on 1-5,
T*_success as 1=logrado / 0.5=con ayuda / 0=no logrado), then run:

    python3 compute_results.py

Prints the SUS mean +/- SD (with adjective interpretation), per-task success
rate, and applicability item means — ready to paste into the Evaluacion chapter.
"""
from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

CSV = Path(__file__).parent / "plantilla_resultados.csv"
ODD = [1, 3, 5, 7, 9]   # positively worded SUS items
TASKS = ["T1", "T2", "T3", "T4", "T5"]
APP = ["A1", "A2", "A3", "A4"]


def sus_score(row: dict[str, str]) -> float | None:
    try:
        vals = {i: int(row[f"SUS{i}"]) for i in range(1, 11)}
    except (ValueError, KeyError):
        return None
    total = sum((vals[i] - 1) if i in ODD else (5 - vals[i]) for i in range(1, 11))
    return total * 2.5


def adjective(score: float) -> str:
    if score >= 85: return "excelente (A)"
    if score >= 71.4: return "bueno"
    if score >= 51: return "aceptable/pobre (OK)"
    return "pobre (F)"


def _floats(rows, key):
    out = []
    for r in rows:
        v = r.get(key, "").strip()
        if v:
            try: out.append(float(v))
            except ValueError: pass
    return out


def main() -> None:
    rows = [r for r in csv.DictReader(CSV.open()) if any(v.strip() for v in r.values())]
    suses = [s for r in rows if (s := sus_score(r)) is not None]

    print(f"Participantes con datos: {len(rows)}\n")
    if suses:
        m = st.mean(suses)
        sd = st.stdev(suses) if len(suses) > 1 else 0.0
        print(f"SUS: media {m:.1f} +/- {sd:.1f}  (n={len(suses)}) -> {adjective(m)}")
        print(f"     por participante: {[round(s,1) for s in suses]}")
        print(f"     referencia: media del sector = 68\n")
    else:
        print("SUS: sin datos todavia.\n")

    print("Tasa de exito por tarea:")
    for t in TASKS:
        vals = _floats(rows, f"{t}_success")
        if vals:
            print(f"  {t}: {100*st.mean(vals):.0f}%  (n={len(vals)})")
    all_tasks = [v for t in TASKS for v in _floats(rows, f"{t}_success")]
    if all_tasks:
        print(f"  Global: {100*st.mean(all_tasks):.0f}%\n")

    print("Aplicabilidad (media Likert 1-5):")
    for a in APP:
        vals = _floats(rows, a)
        if vals:
            print(f"  {a}: {st.mean(vals):.1f}")


if __name__ == "__main__":
    main()
