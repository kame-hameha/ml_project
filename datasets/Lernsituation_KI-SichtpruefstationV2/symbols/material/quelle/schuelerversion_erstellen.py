#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Erzeugt aus den Quell-Notebooks (loesungen/*.ipynb) die Versionen fuer die Lernenden.

    python3 material/quelle/schuelerversion_erstellen.py

Notebooks MIT Niveaus (Notebook-Metadaten "niveaus": true) werden dreimal erzeugt:
    <name>_gruen.ipynb   Einsteiger: alles ausfuehrbar, nur Verstaendnisfragen
    <name>_gelb.ipynb    Aufbau: Pseudocode wird in Python umgesetzt
    <name>_rot.ipynb     Profi: Code selbst schreiben
Oben in jeder Version steht eine Auswahlleiste, die auf die anderen Versionen verlinkt.

Notebooks OHNE Niveaus ergeben eine Version <name>.ipynb (Regeln wie "rot").

Regeln in Code-Zellen der Quellversion:

    # >>> LOESUNG
    # GELB| # Bild in Graustufen umwandeln   <- nur in der gelben Version (ohne "# GELB| ")
    # | grau = None   # TODO                 <- nur in der roten Version (ohne "# | ")
    grau = cv2.cvtColor(bild, ...)           <- nur in der gruenen Version
    # <<< LOESUNG

Zell-Tags: "nur-loesung" (wird entfernt), "niveau-gruen", "niveau-gelb", "niveau-rot"
(Zelle nur in diesen Versionen; Zellen ohne Niveau-Tag erscheinen ueberall).
Ausgaben werden geloescht, Bildpfade "../material/" werden zu "material/".
"""

import copy
import json
import re
from pathlib import Path

ORDNER = Path(__file__).resolve().parents[2]          # .../datasets/symbols
QUELLE = ORDNER / "loesungen"
START = re.compile(r"^(\s*)# >>> LOESUNG\s*$")
ENDE = re.compile(r"^\s*# <<< LOESUNG\s*$")
ROT = re.compile(r"^(\s*)# \|( ?)(.*)$")
GELB = re.compile(r"^(\s*)# GELB\|( ?)(.*)$")

NIVEAUS = {
    "gruen": ("GRÜN – Einsteiger", "Code ist fertig, Sie beantworten Fragen", "#2e8b57", "#d8efe1"),
    "gelb": ("GELB – Aufbau", "Sie setzen Pseudocode in Python um", "#c99700", "#fbefc7"),
    "rot": ("ROT – Profi", "Sie schreiben den Code selbst", "#c0392b", "#f6d6d2"),
}


def umwandeln(zeilen: list[str], niveau: str) -> list[str]:
    aus, im_block = [], False
    for zeile in "".join(zeilen).split("\n"):
        if START.match(zeile) and not im_block:
            im_block = True
            continue
        if im_block and ENDE.match(zeile):
            im_block = False
            continue
        if not im_block:
            aus.append(zeile)
            continue
        g, r = GELB.match(zeile), ROT.match(zeile)
        if niveau == "gelb" and g:
            aus.append(g.group(1) + g.group(3))
        elif niveau == "rot" and r:
            aus.append(r.group(1) + r.group(3))
        elif niveau == "gruen" and not g and not r:
            aus.append(zeile)
    if im_block:
        raise ValueError("'# >>> LOESUNG' ohne passendes '# <<< LOESUNG'")
    text = "\n".join(aus)
    teile = text.split("\n")
    return [z + "\n" for z in teile[:-1]] + [teile[-1]]


def auswahlleiste(stamm: str, aktiv: str) -> dict:
    zellen = []
    for key, (titel, text, farbe, hell) in NIVEAUS.items():
        if key == aktiv:
            zellen.append(f'<td style="background-color:{farbe};color:#ffffff;padding:10px;'
                          f'border:3px solid #1d2833;width:33%"><b>● {titel}</b><br>{text}'
                          f'<br><b>(diese Version)</b></td>')
        else:
            zellen.append(f'<td style="background-color:{hell};color:#1d2833;padding:10px;width:33%">'
                          f'<a href="{stamm}_{key}.ipynb"><b>○ {titel}</b></a><br>{text}'
                          f'<br>(zum Öffnen anklicken)</td>')
    md = ("**Niveau wählen:** Klicken Sie auf ein anderes Feld, um diese Version zu öffnen.\n\n"
          '<table style="width:100%;text-align:center;border-collapse:collapse"><tr>'
          + "".join(zellen) + "</tr></table>")
    return {"cell_type": "markdown", "id": f"niveau-{aktiv}", "metadata": {},
            "source": [z + "\n" for z in md.split("\n")[:-1]] + [md.split("\n")[-1]]}


def version(nb: dict, niveau: str | None) -> dict:
    neu = copy.deepcopy(nb)
    zellen = []
    for zelle in neu["cells"]:
        tags = zelle.get("metadata", {}).get("tags", [])
        if "nur-loesung" in tags:
            continue
        niveau_tags = [t[7:] for t in tags if t.startswith("niveau-")]
        if niveau and niveau_tags and niveau not in niveau_tags:
            continue
        if not niveau and niveau_tags and "rot" not in niveau_tags:
            continue
        zelle["metadata"].pop("tags", None)
        if zelle["cell_type"] == "code":
            zelle["source"] = umwandeln(zelle["source"], niveau or "rot")
            zelle["outputs"], zelle["execution_count"] = [], None
        else:
            zelle["source"] = [z.replace("](../material/", "](material/") for z in zelle["source"]]
        zellen.append(zelle)
    neu["cells"] = zellen
    neu["metadata"].pop("niveaus", None)
    return neu


def speichere(nb: dict, ziel: Path):
    ziel.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{ziel.name}: {len(nb['cells'])} Zellen")


def main():
    for datei in sorted(QUELLE.glob("*.ipynb")):
        nb = json.loads(datei.read_text(encoding="utf-8"))
        if nb.get("metadata", {}).get("niveaus"):
            for niveau in NIVEAUS:
                v = version(nb, niveau)
                v["cells"].insert(0, auswahlleiste(datei.stem, niveau))
                speichere(v, ORDNER / f"{datei.stem}_{niveau}.ipynb")
        else:
            speichere(version(nb, None), ORDNER / datei.name)


if __name__ == "__main__":
    main()
