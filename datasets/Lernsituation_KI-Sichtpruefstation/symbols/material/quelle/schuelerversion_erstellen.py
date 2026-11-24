#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Erzeugt aus den Loesungs-Notebooks (loesungen/*.ipynb) die Schuelerversionen.

    python3 material/quelle/schuelerversion_erstellen.py

Regeln in Code-Zellen der Loesungsversion:

    # >>> LOESUNG
    # | x = None   # TODO: ...      <- wird in der Schuelerversion zu "x = None   # TODO: ..."
    x = berechne_loesung()          <- wird in der Schuelerversion entfernt
    # <<< LOESUNG

Zellen mit dem Tag "nur-loesung" werden entfernt, Ausgaben geloescht und
Bildpfade "../material/" zu "material/" angepasst.
"""

import json
import re
from pathlib import Path

ORDNER = Path(__file__).resolve().parents[2]          # .../datasets/symbols
QUELLE = ORDNER / "loesungen"
START = re.compile(r"^(\s*)# >>> LOESUNG\s*$")
ENDE = re.compile(r"^\s*# <<< LOESUNG\s*$")
PLATZHALTER = re.compile(r"^(\s*)# \|( ?)(.*)$")


def schuelercode(zeilen: list[str]) -> list[str]:
    text = "".join(zeilen).split("\n")
    aus, im_block, einzug = [], False, ""
    for zeile in text:
        if (m := START.match(zeile)) and not im_block:
            im_block, einzug = True, m.group(1)
            continue
        if im_block and ENDE.match(zeile):
            im_block = False
            continue
        if im_block:
            if m := PLATZHALTER.match(zeile):
                aus.append(m.group(1) + m.group(3))
            continue
        aus.append(zeile)
    if im_block:
        raise ValueError("'# >>> LOESUNG' ohne passendes '# <<< LOESUNG'")
    text = "\n".join(aus)
    return [z + "\n" for z in text.split("\n")[:-1]] + [text.split("\n")[-1]]


def main():
    for datei in sorted(QUELLE.glob("*.ipynb")):
        nb = json.loads(datei.read_text(encoding="utf-8"))
        zellen = []
        for zelle in nb["cells"]:
            if "nur-loesung" in zelle.get("metadata", {}).get("tags", []):
                continue
            if zelle["cell_type"] == "code":
                zelle["source"] = schuelercode(zelle["source"])
                zelle["outputs"], zelle["execution_count"] = [], None
            else:
                zelle["source"] = [z.replace("](../material/", "](material/") for z in zelle["source"]]
            zellen.append(zelle)
        nb["cells"] = zellen
        ziel = ORDNER / datei.name
        ziel.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        aufgaben = sum("TODO" in "".join(z["source"]) for z in zellen if z["cell_type"] == "code")
        print(f"{ziel.name}: {len(zellen)} Zellen, {aufgaben} Aufgabenzellen")


if __name__ == "__main__":
    main()
