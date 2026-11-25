#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Erstellt von:   Kai Metzger (Franz-Oberthuer-Schule Wuerzburg)
# Ueberarbeitet:  November 2026 - Lernsituation "KI-Sichtpruefstation"
# Version:        2.0.0
# =============================================================================
"""Hilfsfunktionen fuer die Lernsituation "KI-Sichtpruefstation".

Dieses Modul enthaelt alles, was NICHT im Mittelpunkt des Unterrichts steht,
aber in mehreren Notebooks gebraucht wird:

- Klassen eines Datensatzes lesen/schreiben   (klassen.txt)
- Bilder einheitlich vorbereiten              (Training UND Inferenz!)
- Datensatz laden, zwischenspeichern, aufteilen
- Diagramme: Beispielbilder, Lernkurven, Konfusionsmatrix
- Interaktive HTML-Seiten im Notebook anzeigen
- Kamera (USB-Kamera ueber OpenCV oder Raspberry-Pi-Kamera ueber Picamera2)
- Klassifikator fuer die Inferenz (.tflite oder .keras)

TensorFlow wird nur dort importiert, wo es gebraucht wird. Dadurch laesst sich
das Modul auch ohne TensorFlow (z. B. im Aufnahmeskript 00) verwenden.

Datenstruktur eines Datensatzes:

    dataset/            <- Datensatz-Ordner
      klassen.txt       <- "0;Kreuz", "1;Kreis", ... (eine Zeile je Klasse)
      0/                <- Bilder der Klasse 0
      1/
      ...
"""

from __future__ import annotations

import hashlib
import html
import json
import time
from pathlib import Path

import cv2
import numpy as np

# =============================================================================
# Grundeinstellungen
# =============================================================================
ORDNER = Path(__file__).resolve().parent           # .../datasets/symbols
BILDENDUNGEN = {".png", ".jpg", ".jpeg", ".bmp"}
KLASSEN_DATEI = "klassen.txt"
CACHE_ORDNER = ORDNER / ".cache"
VORVERARBEITUNG = "linear-v3"     # aendert sich die Vorverarbeitung, wird neu geladen


def _pfad(pfad) -> Path:
    """Relative Datensatz-Namen wie "dataset1" auch aus Unterordnern finden."""
    pfad = Path(pfad)
    if not pfad.is_absolute() and not pfad.exists() and (ORDNER / pfad).exists():
        return ORDNER / pfad
    return pfad


# =============================================================================
# Reproduzierbarkeit
# =============================================================================
def setze_seed(seed: int = 42, tensorflow: bool = True) -> None:
    """Setzt alle Zufallsgeneratoren, damit Ergebnisse wiederholbar sind."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    if tensorflow:
        import keras
        keras.utils.set_random_seed(seed)


# =============================================================================
# Klassen eines Datensatzes
# =============================================================================
def schreibe_klassen(datensatz: str | Path, klassen: list[str]) -> Path:
    """Legt Klassenordner 0..n-1 an und schreibt klassen.txt."""
    datensatz = Path(datensatz)
    datensatz.mkdir(parents=True, exist_ok=True)
    for i in range(len(klassen)):
        (datensatz / str(i)).mkdir(exist_ok=True)
    datei = datensatz / KLASSEN_DATEI
    datei.write_text(
        "".join(f"{i};{name}\n" for i, name in enumerate(klassen)),
        encoding="utf-8",
    )
    return datei


def lese_klassen(datensatz: str | Path) -> list[str]:
    """Liest die Klassennamen.

    Reihenfolge = Klassennummer = Ordnername. Fehlt klassen.txt, werden die
    numerischen Unterordner (0, 1, 2, ...) als Namen verwendet.
    """
    datensatz = _pfad(datensatz)
    datei = datensatz / KLASSEN_DATEI
    if datei.exists():
        klassen = {}
        for zeile in datei.read_text(encoding="utf-8").splitlines():
            zeile = zeile.strip()
            if not zeile or zeile.startswith("#"):
                continue
            nr, _, name = zeile.partition(";")
            klassen[int(nr)] = name.strip() or nr.strip()
        return [klassen[i] for i in sorted(klassen)]

    ordner = [p.name for p in datensatz.iterdir() if p.is_dir() and p.name.isdigit()]
    if not ordner:
        raise FileNotFoundError(f"Keine Klassenordner 0, 1, ... in {datensatz}")
    return [str(i) for i in sorted(int(n) for n in ordner)]


def bilddateien(ordner: str | Path) -> list[Path]:
    """Alle Bilddateien eines Ordners, sortiert."""
    ordner = _pfad(ordner)
    if not ordner.exists():
        return []
    return sorted(p for p in ordner.iterdir()
                  if p.is_file() and p.suffix.lower() in BILDENDUNGEN)


def zaehle_bilder(datensatz: str | Path) -> dict[str, int]:
    """Anzahl Bilder je Klasse, z. B. {'Kreuz': 432, 'Kreis': 430, ...}."""
    datensatz = _pfad(datensatz)
    return {name: len(bilddateien(datensatz / str(i)))
            for i, name in enumerate(lese_klassen(datensatz))}


def alle_bilder(datensatz: str | Path) -> list[tuple[Path, int]]:
    """Liste aller Bilder als Paare (Datei, Klassennummer).

    Beispiel:
        for datei, klasse in kh.alle_bilder("dataset1"):
            print(datei.name, klasse)
    """
    datensatz = _pfad(datensatz)
    return [(datei, nummer) for nummer in range(len(lese_klassen(datensatz)))
            for datei in bilddateien(datensatz / str(nummer))]


def zeige_anzahl(datensatz: str | Path):
    """Balkendiagramm: Wie viele Bilder hat jede Klasse?"""
    import matplotlib.pyplot as plt

    anzahl = zaehle_bilder(datensatz)
    plt.figure(figsize=(max(5, len(anzahl) * 1.3), 3))
    balken = plt.bar(list(anzahl.keys()), list(anzahl.values()), color="#3a6ea5")
    plt.bar_label(balken)
    plt.title(f"{_pfad(datensatz).name}: {sum(anzahl.values())} Bilder")
    plt.ylabel("Anzahl Bilder")
    plt.tight_layout()
    plt.show()


# =============================================================================
# Bildvorbereitung - IDENTISCH fuer Training und Inferenz
# =============================================================================
def bild_vorbereiten(bild_bgr: np.ndarray, groesse: int | tuple[int, int],
                     farbe: str = "rgb") -> np.ndarray:
    """Skaliert ein OpenCV-Bild (BGR) auf groesse x groesse.

    Rueckgabe: uint8-Array mit Form (H, W, 3) bei farbe="rgb"
               bzw. (H, W, 1) bei farbe="grau".
    Werte bleiben 0..255 - die Normierung passiert IM Modell (Rescaling).

    Verkleinert wird bilinear (cv2.INTER_LINEAR). Das entspricht der
    Voreinstellung von tf.keras.utils.image_dataset_from_directory in den
    Original-Notebooks 02a/02b/02c, mit denen die Netze trainiert wurden.
    """
    if isinstance(groesse, int):
        groesse = (groesse, groesse)
    hoehe, breite = groesse
    klein = cv2.resize(bild_bgr, (breite, hoehe), interpolation=cv2.INTER_LINEAR)
    if farbe == "grau":
        return cv2.cvtColor(klein, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
    if farbe == "rgb":
        return cv2.cvtColor(klein, cv2.COLOR_BGR2RGB)
    raise ValueError("farbe muss 'rgb' oder 'grau' sein")


def zeige_bild(bild, titel: str = "", bgr: bool = False):
    """Zeigt ein Bild an. bgr=True fuer Bilder direkt aus cv2.imread / Kamera."""
    import matplotlib.pyplot as plt

    if bild is None:
        _meldung(False, "Es ist noch kein Bild vorhanden.")
        return
    bild = np.asarray(bild)
    if bild.ndim == 3 and bild.shape[-1] == 1:
        bild = bild[..., 0]
    plt.figure(figsize=(4, 4) if bild.shape[0] < 100 else (6, 4.5))
    if bild.ndim == 2:
        plt.imshow(bild, cmap="gray", vmin=0, vmax=255 if bild.max() > 1 else 1)
    else:
        plt.imshow(bild[..., ::-1] if bgr else bild)
    plt.title(titel)
    plt.axis("off")
    plt.show()


def zeige_bgr_rgb(bild_bgr):
    """Zeigt dasselbe Kamerabild einmal falsch (BGR) und einmal richtig (RGB)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].imshow(bild_bgr)
    ax[0].set_title("falsch: Farbkanäle in OpenCV-Reihenfolge B, G, R")
    ax[1].imshow(cv2.cvtColor(bild_bgr, cv2.COLOR_BGR2RGB))
    ax[1].set_title("richtig: umgewandelt in R, G, B")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    plt.show()


def farbtest() -> bool:
    """Prueft die Farbreihenfolge der ganzen Pipeline mit einem roten Testbild.

    1. OpenCV speichert ein rotes Bild (B, G, R) = (0, 0, 255) als PNG.
    2. Ein anderes Programm (Matplotlib) liest die Datei: sie muss rot sein.
    3. cv2.imread + bild_vorbereiten muessen fuer das Netz (R, G, B) = (255, 0, 0) liefern.
    """
    import tempfile
    import matplotlib.image as mpimg

    rot_bgr = np.zeros((40, 40, 3), np.uint8)
    rot_bgr[..., 2] = 255                                    # Kanal 2 = Rot bei OpenCV
    ergebnisse = []
    with tempfile.TemporaryDirectory() as tmp:
        datei = Path(tmp) / "rot.png"
        cv2.imwrite(str(datei), rot_bgr)
        datei_rgb = (mpimg.imread(str(datei))[0, 0, :3] * 255).round()
        ergebnisse.append(("PNG-Datei ist wirklich rot", list(datei_rgb) == [255, 0, 0]))
        geladen = cv2.imread(str(datei))
        netz = bild_vorbereiten(geladen, 8, "rgb")[0, 0]
        ergebnisse.append(("Netz bekommt (R, G, B) = (255, 0, 0)", list(netz) == [255, 0, 0]))
        grau = bild_vorbereiten(geladen, 8, "grau")[0, 0, 0]
        ergebnisse.append(("Graustufe von Rot = 76 (0,299 · 255)", int(grau) == 76))
    for text, ok in ergebnisse:
        _meldung(ok, text if ok else text + " - FEHLER in der Farbreihenfolge!")
    return all(ok for _, ok in ergebnisse)


# =============================================================================
# Datensatz laden (mit Zwischenspeicher)
# =============================================================================
def _cache_schluessel(dateien: list[Path], groesse, farbe) -> str:
    h = hashlib.sha1()
    for p in dateien:
        h.update(str(p).encode())
        h.update(str(p.stat().st_size).encode())
    h.update(f"{groesse}-{farbe}-{VORVERARBEITUNG}".encode())
    return h.hexdigest()[:12]


def lade_datensatz(datensatz: str | Path, groesse: int = 32, farbe: str = "rgb",
                   zwischenspeichern: bool = True):
    """Laedt alle Bilder eines Datensatzes in den Arbeitsspeicher.

    Rueckgabe: X (uint8, Form N x H x W x C), y (int, Klassennummer), klassen
    Beim ersten Mal werden die Bilder einzeln gelesen (auf dem Pi bei
    1920x1080-Bildern ca. 1-2 Minuten). Danach liegt eine .npz-Datei im Ordner
    .cache und das Laden dauert nur noch Sekunden.
    """
    datensatz = _pfad(datensatz)
    klassen = lese_klassen(datensatz)
    paare = [(p, i) for i in range(len(klassen))
             for p in bilddateien(datensatz / str(i))]
    if not paare:
        raise FileNotFoundError(f"Keine Bilder in {datensatz.resolve()}")

    cache = None
    if zwischenspeichern:
        schluessel = _cache_schluessel([p for p, _ in paare], groesse, farbe)
        CACHE_ORDNER.mkdir(exist_ok=True)
        cache = CACHE_ORDNER / f"{datensatz.name}_{groesse}_{farbe}_{schluessel}.npz"
        if cache.exists():
            daten = np.load(cache)
            print(f"Aus Zwischenspeicher geladen: {len(daten['y'])} Bilder")
            return daten["X"], daten["y"], klassen

    X, y = [], []
    start = time.time()
    for n, (pfad, klasse) in enumerate(paare, 1):
        bild = cv2.imread(str(pfad))
        if bild is None:
            print(f"\nWarnung: {pfad} ist nicht lesbar und wird uebersprungen.")
            continue
        X.append(bild_vorbereiten(bild, groesse, farbe))
        y.append(klasse)
        if n % 100 == 0 or n == len(paare):
            print(f"\rBilder gelesen: {n}/{len(paare)} "
                  f"({time.time() - start:.0f} s)", end="")
    print()
    X = np.stack(X).astype(np.uint8)
    y = np.array(y, dtype=np.int64)
    if cache is not None:
        np.savez_compressed(cache, X=X, y=y)
    return X, y, klassen


def teile_datensatz(X, y, anteile=(0.70, 0.15, 0.15), seed: int = 42):
    """Teilt geschichtet (stratifiziert) in Training, Validierung und Test."""
    from sklearn.model_selection import train_test_split

    a_train, a_val, a_test = anteile
    if abs(a_train + a_val + a_test - 1.0) > 1e-6:
        raise ValueError("Die Anteile muessen zusammen 1.0 ergeben.")
    X_train, X_rest, y_train, y_rest = train_test_split(
        X, y, test_size=a_val + a_test, random_state=seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_rest, y_rest, test_size=a_test / (a_val + a_test),
        random_state=seed, stratify=y_rest)
    return X_train, X_val, X_test, y_train, y_val, y_test


def als_tf_dataset(X, y, batch_size=32, mischen=False, seed: int = 42):
    """Macht aus NumPy-Arrays ein tf.data.Dataset (Stapel = Batches).

    Wie in den Original-Notebooks steckt die Augmentierung IM Modell
    (aktiv nur beim Training) - hier werden die Bilder nur gemischt und
    in Stapel aufgeteilt.
    """
    import tensorflow as tf

    # Bilder bleiben uint8 (spart 3/4 des Arbeitsspeichers) und werden erst
    # stapelweise in float32 umgewandelt.
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if mischen:
        ds = ds.shuffle(len(X), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).map(lambda b, l: (tf.cast(b, tf.float32), l))
    return ds.prefetch(1)    # 1 Stapel Vorlauf schont den RAM des Pi


# =============================================================================
# Diagramme
# =============================================================================
def zeige_beispiele(X, y, klassen, anzahl: int = 12, vorhersage=None, seed=0):
    """Zeigt zufaellige Bilder mit Klassenname (und ggf. Vorhersage)."""
    import matplotlib.pyplot as plt

    if X is None or len(X) == 0:
        _meldung(False, "Es sind noch keine Bilder vorhanden.")
        return
    rng = np.random.default_rng(seed)
    auswahl = rng.choice(len(X), size=min(anzahl, len(X)), replace=False)
    spalten = 6
    zeilen = int(np.ceil(len(auswahl) / spalten))
    plt.figure(figsize=(2.2 * spalten, 2.4 * zeilen))
    for k, i in enumerate(auswahl):
        plt.subplot(zeilen, spalten, k + 1)
        bild = np.asarray(X[i])
        if bild.max() <= 1.0:                       # normierte Bilder (0 ... 1)
            bild = bild * 255
        if bild.ndim == 3 and bild.shape[-1] == 1:
            bild = bild[..., 0]
        if bild.ndim == 2:
            plt.imshow(bild, cmap="gray", vmin=0, vmax=255)
        else:
            plt.imshow(bild.astype("uint8"))
        titel = klassen[y[i]]
        farbe = "black"
        if vorhersage is not None:
            titel += f"\n→ {klassen[vorhersage[i]]}"
            farbe = "green" if vorhersage[i] == y[i] else "red"
        plt.title(titel, fontsize=10, color=farbe)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


def zeige_augmentierung(bild, augmentierung, anzahl: int = 9):
    """Zeigt ein Trainingsbild mehrmals zufaellig veraendert (augmentiert)."""
    import matplotlib.pyplot as plt

    x = np.asarray(bild, dtype="float32")[np.newaxis]
    spalten = 3
    plt.figure(figsize=(6, 6))
    for i in range(anzahl):
        aug = np.clip(np.asarray(augmentierung(x, training=True))[0], 0, 255).astype("uint8")
        plt.subplot(int(np.ceil(anzahl / spalten)), spalten, i + 1)
        if aug.ndim == 3 and aug.shape[-1] == 1:
            aug = aug[..., 0]
        plt.imshow(aug, cmap="gray" if aug.ndim == 2 else None, vmin=0, vmax=255)
        plt.axis("off")
    plt.suptitle("Ein Trainingsbild – neunmal zufällig verändert")
    plt.tight_layout()
    plt.show()


def tabelle(zeilen: list[dict]):
    """Liste von Ergebnissen als uebersichtliche Tabelle."""
    import pandas as pd
    return pd.DataFrame(zeilen)


def zeige_verteilung(y, klassen, titel="Bilder je Klasse"):
    import matplotlib.pyplot as plt

    anzahl = np.bincount(y, minlength=len(klassen))
    plt.figure(figsize=(max(4, len(klassen) * 1.2), 3))
    balken = plt.bar(klassen, anzahl, color="#3a6ea5")
    plt.bar_label(balken)
    plt.title(titel)
    plt.ylabel("Anzahl")
    plt.tight_layout()
    plt.show()


def zeige_lernkurven(*verlaeufe, markierung: int | None = None):
    """Genauigkeit und Verlust ueber die Epochen (Training vs. Validierung).

    Mehrere History-Objekte (z. B. Training + Feinabstimmung) werden
    hintereinander gehaengt. markierung = Epoche fuer eine senkrechte Linie.
    """
    import matplotlib.pyplot as plt

    werte = {}
    for v in verlaeufe:
        h = v.history if hasattr(v, "history") else v
        for k, liste in h.items():
            werte.setdefault(k, []).extend(liste)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for i, (metrik, titel) in enumerate([("accuracy", "Genauigkeit"),
                                         ("loss", "Verlust (Loss)")]):
        ax[i].plot(werte.get(metrik, []), label="Training")
        ax[i].plot(werte.get("val_" + metrik, []), label="Validierung")
        if markierung is not None:
            ax[i].axvline(markierung, color="gray", ls="--", label="Feinabstimmung")
        ax[i].set_title(titel)
        ax[i].set_xlabel("Epoche")
        ax[i].grid(alpha=0.3)
        ax[i].legend()
    plt.tight_layout()
    plt.show()


def zeige_konfusionsmatrix(y_wahr, y_vorhersage, klassen, titel="Konfusionsmatrix"):
    """Konfusionsmatrix + Klassifikationsbericht (Precision, Recall, F1)."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix

    cm = confusion_matrix(y_wahr, y_vorhersage, labels=range(len(klassen)))
    groesse = max(4, len(klassen) * 1.1)
    fig, ax = plt.subplots(figsize=(groesse + 1, groesse))
    bild = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(klassen)), klassen, rotation=45, ha="right")
    ax.set_yticks(range(len(klassen)), klassen)
    ax.set_xlabel("Vorhersage des Modells")
    ax.set_ylabel("Tatsaechliche Klasse")
    ax.set_title(titel)
    grenze = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > grenze else "black")
    fig.colorbar(bild, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.show()
    print(classification_report(y_wahr, y_vorhersage, labels=range(len(klassen)),
                                target_names=klassen, digits=3, zero_division=0))
    return cm


def zeige_schwellenwirkung(wahrsch, y_wahr):
    """Wie veraendern sich Rueckweisung und Genauigkeit mit der Konfidenzschwelle?"""
    import matplotlib.pyplot as plt

    wahrsch = np.asarray(wahrsch)
    y_pred = wahrsch.argmax(axis=1)
    konf = wahrsch.max(axis=1)
    richtig = y_pred == np.asarray(y_wahr)
    schwellen = np.linspace(0.3, 0.99, 40)
    mensch = [(konf < s).mean() * 100 for s in schwellen]
    genau = [richtig[konf >= s].mean() * 100 if (konf >= s).any() else np.nan for s in schwellen]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(schwellen, genau, label="Genauigkeit der automatischen Entscheidungen")
    ax[0].plot(schwellen, mensch, label="Anteil, den ein Mensch prüft")
    ax[0].set_xlabel("Schwelle")
    ax[0].set_ylabel("%")
    ax[0].grid(alpha=0.3)
    ax[0].legend()
    ax[1].hist(konf[richtig], bins=20, alpha=0.7, label="richtig erkannt", color="#2e8b57")
    ax[1].hist(konf[~richtig], bins=20, alpha=0.7, label="falsch erkannt", color="#c0392b")
    ax[1].set_xlabel("Konfidenz (höchste Wahrscheinlichkeit)")
    ax[1].set_ylabel("Anzahl Testbilder")
    ax[1].legend()
    plt.tight_layout()
    plt.show()


# =============================================================================
# Interaktive HTML-Seiten im Notebook
# =============================================================================
def zeige_html(datei: str | Path, hoehe: int = 600):
    """Bettet eine interaktive HTML-Seite aus material/ ins Notebook ein.

    Die Seite wird als <iframe srcdoc=...> eingebunden und passt ihre Hoehe
    automatisch an den Inhalt an (keine Scrollleiste im Notebook).
    Funktioniert in JupyterLab, Jupyter Notebook und VS Code - ohne Internet.
    Jede Datei laesst sich auch direkt im Browser oeffnen.
    """
    from IPython.display import HTML, display

    pfad = Path(datei)
    if not pfad.is_absolute() and not pfad.exists():
        pfad = ORDNER / "material" / pfad
    inhalt = pfad.read_text(encoding="utf-8")
    anpassen = (
        "<script>(function(){function f(){try{var e=window.frameElement;if(e){"
        "e.style.height=(document.documentElement.scrollHeight+2)+'px';}}catch(x){}}"
        "window.addEventListener('load',f);if(window.ResizeObserver){"
        "new ResizeObserver(f).observe(document.documentElement);}setTimeout(f,300);})();</script>")
    inhalt = inhalt.replace("</body>", anpassen + "</body>") if "</body>" in inhalt else inhalt + anpassen
    display(HTML(
        f'<iframe srcdoc="{html.escape(inhalt, quote=True)}" '
        f'style="width:100%;height:{hoehe}px;border:1px solid #ccc;'
        f'border-radius:8px;background:white;display:block" '
        f'sandbox="allow-scripts allow-same-origin allow-popups allow-modals">'
        f'</iframe>'))


# =============================================================================
# Selbstkontrolle in den Notebooks
# =============================================================================
def _meldung(ok: bool, text: str):
    """Farbige Rueckmeldung (gruen = richtig, rot = noch nicht richtig)."""
    try:
        from IPython.display import HTML, display
        farbe, rand, wort = (("#e3f4ea", "#2e8b57", "Richtig")
                             if ok else ("#fbe4e1", "#c0392b", "Noch nicht richtig"))
        display(HTML(f'<div style="background:{farbe};border-left:5px solid {rand};'
                     f'padding:5px 10px;margin:3px 0;font-family:sans-serif">'
                     f'<b>{wort}:</b> {html.escape(text)}</div>'))
    except ImportError:
        print(("[OK]   " if ok else "[FEHLT] ") + text)


def pruefe(bedingung, ok: str, fehler: str) -> bool:
    """Prueft eine Bedingung und gibt eine farbige Rueckmeldung aus."""
    try:
        erfuellt = bool(bedingung() if callable(bedingung) else bedingung)
    except Exception as e:                       # noqa: BLE001
        erfuellt = False
        fehler = f"{fehler}  ({type(e).__name__}: {e})"
    _meldung(erfuellt, ok if erfuellt else fehler)
    return erfuellt


def pruefe_form(wert, form: tuple, name: str) -> bool:
    """Prueft, ob ein Array die erwartete Form hat, z. B. (32, 32)."""
    ist = getattr(wert, "shape", None)
    if ist is None:
        _meldung(False, f"{name} ist noch leer (None) oder kein Bild/Array.")
        return False
    ok = tuple(ist) == tuple(form)
    _meldung(ok, f"{name} hat die Form {tuple(ist)}" + ("" if ok else f", erwartet wird {tuple(form)}."))
    return ok


def pruefe_gleich(wert, erwartet, name: str) -> bool:
    """Prueft, ob ein Wert dem erwarteten Wert entspricht."""
    try:
        ok = bool(np.all(np.asarray(wert) == np.asarray(erwartet)))
    except Exception:                            # noqa: BLE001
        ok = False
    _meldung(ok, f"{name} = {wert}" + ("" if ok else f", erwartet wird {erwartet}."))
    return ok


def pruefe_bereich(wert, minimum, maximum, name: str) -> bool:
    """Prueft, ob alle Werte eines Arrays zwischen minimum und maximum liegen."""
    if wert is None or not hasattr(wert, "min"):
        _meldung(False, f"{name} ist noch leer (None).")
        return False
    ok = minimum <= float(wert.min()) and float(wert.max()) <= maximum
    _meldung(ok, f"{name}: Werte von {float(wert.min()):g} bis {float(wert.max()):g}"
             + ("" if ok else f", erwartet wird {minimum} bis {maximum}."))
    return ok


def zeige_vorhersage(bild, wahrscheinlichkeiten, klassen, richtig=None):
    """Bild links, Wahrscheinlichkeit jeder Klasse als Balken rechts."""
    import matplotlib.pyplot as plt

    if wahrscheinlichkeiten is None:
        _meldung(False, "Es gibt noch keine Wahrscheinlichkeiten.")
        return
    p = np.asarray(wahrscheinlichkeiten, dtype=float)
    nummer = int(np.argmax(p))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1, 2]})
    b = np.asarray(bild)
    if b.ndim == 3 and b.shape[-1] == 1:
        b = b[..., 0]
    ax[0].imshow(b, cmap="gray" if b.ndim == 2 else None)
    ax[0].axis("off")
    titel = f"Erkannt: {klassen[nummer]}"
    if richtig is not None:
        titel += f"\n(richtig: {klassen[int(richtig)]})"
    ax[0].set_title(titel, color="green" if richtig is None or nummer == int(richtig) else "red")
    farben = ["#d9731a" if i == nummer else "#3a6ea5" for i in range(len(klassen))]
    balken = ax[1].barh(klassen, p * 100, color=farben)
    ax[1].bar_label(balken, labels=[f"{v * 100:.1f} %" for v in p], padding=3)
    ax[1].set_xlim(0, 115)
    ax[1].invert_yaxis()
    ax[1].set_xlabel("Wahrscheinlichkeit in %")
    plt.tight_layout()
    plt.show()


# =============================================================================
# Kamera: USB-Kamera (OpenCV) oder Raspberry-Pi-Kameramodul (Picamera2)
# =============================================================================
class Kamera:
    """Einheitlicher Zugriff auf die Kamera. lies() liefert ein BGR-Bild.

    quelle: 0, 1, ...    -> USB-Kamera (OpenCV)
            "picamera"   -> Kameramodul am CSI-Anschluss (Picamera2)
            "datei.mp4"  -> Videodatei (zum Testen ohne Kamera)
    """

    def __init__(self, quelle=0, breite: int = 640, hoehe: int = 480,
                 mjpg: bool = True):
        self.picam = None
        self.cap = None
        if quelle == "picamera":
            from picamera2 import Picamera2   # nur auf dem Raspberry Pi
            self.picam = Picamera2()
            # "RGB888" liefert bei Picamera2 Pixel in BGR-Reihenfolge -> passt zu OpenCV
            cfg = self.picam.create_preview_configuration(
                main={"size": (breite, hoehe), "format": "RGB888"})
            self.picam.configure(cfg)
            self.picam.start()
            time.sleep(0.5)                    # Belichtung einschwingen lassen
        else:
            if isinstance(quelle, str) and quelle.isdigit():
                quelle = int(quelle)
            self.cap = cv2.VideoCapture(quelle)
            if not self.cap.isOpened():
                raise RuntimeError(
                    f"Kamera {quelle!r} laesst sich nicht oeffnen. "
                    "Angeschlossen? Pruefen mit: v4l2-ctl --list-devices")
            if isinstance(quelle, int):
                if mjpg:   # MJPG erlaubt bei USB-Kameras hoehere Aufloesungen
                    self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, breite)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hoehe)

    def lies(self):
        if self.picam is not None:
            return self.picam.capture_array()
        ok, bild = self.cap.read()
        return bild if ok else None

    def aufloesung(self):
        bild = self.lies()
        return None if bild is None else (bild.shape[1], bild.shape[0])

    def schliessen(self):
        if self.picam is not None:
            self.picam.stop()
        if self.cap is not None:
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.schliessen()


# =============================================================================
# Merkmalskarten eines trainierten CNN
# =============================================================================
def zeige_merkmalskarten(modell, bild, faltung_nr: int = 1, anzahl: int = 8):
    """Zeigt fuer ein Bild die Merkmalskarten einer Faltungsschicht.

    faltung_nr: 1 = erste Conv2D-Schicht, 2 = zweite, ...
    Gezeigt werden die Karten mit der staerksten Reaktion. Ueber jeder Karte
    steht der gelernte 3 x 3-Filter (Mittelwert ueber die Farbkanaele).
    hell = an dieser Stelle passt das Muster des Filters.
    """
    import keras
    import matplotlib.pyplot as plt

    faltungen = [s for s in modell.layers if type(s).__name__ == "Conv2D"]
    schicht = faltungen[faltung_nr - 1]
    teilmodell = keras.Model(modell.inputs, schicht.output)
    karten = teilmodell.predict(np.asarray(bild, dtype="float32")[np.newaxis], verbose=0)[0]
    staerke = karten.mean(axis=(0, 1))
    reihenfolge = np.argsort(staerke)[::-1][:anzahl]
    filter_ = schicht.get_weights()[0]                         # Form (3, 3, Kanaele, Filter)

    spalten = anzahl + 1
    fig, ax = plt.subplots(2, spalten, figsize=(1.9 * spalten, 4.2),
                           gridspec_kw={"height_ratios": [1, 2.2]})
    b = np.asarray(bild)
    ax[1, 0].imshow(b[..., 0] if b.shape[-1] == 1 else b.astype("uint8"), cmap="gray")
    ax[1, 0].set_title("Eingabe", fontsize=10)
    ax[0, 0].text(0.5, 0.5, "gelernter\nFilter 3 × 3", ha="center", va="center", fontsize=9)
    for k, nr in enumerate(reihenfolge, start=1):
        f = filter_[..., nr].mean(axis=2)
        grenze = np.abs(f).max() + 1e-9
        ax[0, k].imshow(f, cmap="bwr", vmin=-grenze, vmax=grenze)
        ax[0, k].set_title(f"Filter {nr}", fontsize=9)
        ax[1, k].imshow(karten[..., nr], cmap="magma")
        ax[1, k].set_title(f"Stärke {staerke[nr]:.2f}", fontsize=9)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    ax[0, 0].axis("off")
    fig.suptitle(f"Faltungsschicht {faltung_nr} ({schicht.name}): die {anzahl} stärksten von "
                 f"{karten.shape[-1]} Merkmalskarten, Größe {karten.shape[0]} × {karten.shape[1]}",
                 fontsize=11)
    plt.tight_layout()
    plt.show()
    print("Oben: der gelernte Filter (rot = positives Gewicht, blau = negatives Gewicht).")
    print("Unten: seine Merkmalskarte – hell = hier hat der Filter sein Muster gefunden.")


def zeige_gradcam(modell, basis, kopf, X, y, klassen, schicht: str = "block_6_expand"):
    """Grad-CAM fuer je ein Testbild pro Klasse (Transfer-Learning-Modell aus Notebook 13).

    basis: das MobileNetV2-Basisnetz, kopf: Liste der Kopf-Schichten
    (z. B. [pooling, dropout, ausgabe]), schicht: Zwischenschicht der Basis.
    """
    import keras
    import matplotlib.pyplot as plt
    import tensorflow as tf

    grad_modell = keras.Model(basis.inputs, [basis.get_layer(schicht).output, basis.output])

    def berechne(bild):
        x = tf.convert_to_tensor(np.asarray(bild, dtype="float32")[np.newaxis] / 127.5 - 1)
        with tf.GradientTape() as tape:
            tape.watch(x)                       # sonst zeichnet das eingefrorene Netz nichts auf
            karten, z = grad_modell(x, training=False)
            for s in kopf:
                z = s(z, training=False)
            klasse = int(tf.argmax(z[0]))
            wert = z[0, klasse]
        grad = tape.gradient(wert, karten)
        gewichte = tf.reduce_mean(grad, axis=(0, 1, 2))
        heat = tf.nn.relu(tf.reduce_sum(karten[0] * gewichte, axis=-1)).numpy()
        heat = heat / (heat.max() + 1e-8)
        return cv2.resize(heat, (bild.shape[1], bild.shape[0])), klasse, float(z[0, klasse])

    n = len(klassen)
    plt.figure(figsize=(3.2 * n, 6.4))
    for k in range(n):
        treffer = np.where(np.asarray(y) == k)[0]
        if len(treffer) == 0:
            continue
        bild = np.asarray(X[treffer[0]])
        heat, klasse, p = berechne(bild)
        farbig = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
        overlay = (0.55 * bild + 0.45 * farbig).astype("uint8")
        plt.subplot(2, n, k + 1)
        plt.imshow(bild.astype("uint8"))
        plt.axis("off")
        plt.title(f"richtig: {klassen[k]}")
        plt.subplot(2, n, n + k + 1)
        plt.imshow(overlay)
        plt.axis("off")
        plt.title(f"erkannt: {klassen[klasse]} ({p:.0%})", color="green" if klasse == k else "red")
    plt.tight_layout()
    plt.show()


def modellpfad(dateiname: str) -> Path:
    """Pfad im Ordner modelle/ (wird bei Bedarf angelegt)."""
    ordner = ORDNER / "modelle"
    ordner.mkdir(exist_ok=True)
    return ordner / dateiname


# =============================================================================
# Modell speichern / exportieren / laden
# =============================================================================
def speichere_modell(modell, name: str, klassen: list[str], groesse: int,
                     farbe: str, tflite: bool = True, ordner: str | Path = None,
                     **zusatz) -> dict:
    """Speichert Modell (.keras), optional TFLite-Datei und modell_info.json.

    In modell_info.json steht alles, was die Inferenz (Notebook 20 / Skript 20)
    wissen muss: Klassen, Bildgroesse, Farbmodus.
    """
    ordner = Path(ordner or ORDNER / "modelle")
    ordner.mkdir(parents=True, exist_ok=True)
    keras_pfad = ordner / f"{name}.keras"
    modell.save(keras_pfad)
    info = {"name": name, "klassen": list(klassen), "groesse": int(groesse),
            "farbe": farbe, "keras": keras_pfad.name,
            "erstellt": time.strftime("%Y-%m-%d %H:%M"), **zusatz}
    if tflite:
        try:
            tfl_pfad = exportiere_tflite(modell, ordner / f"{name}.tflite")
            info["tflite"] = tfl_pfad.name
        except Exception as e:                    # noqa: BLE001
            print(f"Hinweis: TFLite-Export nicht moeglich ({e}). "
                  "Die .keras-Datei funktioniert trotzdem.")
    (ordner / f"{name}_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    groesse_mb = keras_pfad.stat().st_size / 1e6
    print(f"Gespeichert: {keras_pfad.name} ({groesse_mb:.1f} MB)"
          + (f", {info['tflite']}" if "tflite" in info else "")
          + f", {name}_info.json")
    return info


def exportiere_tflite(modell, pfad: str | Path, quantisieren: bool = False) -> Path:
    """Wandelt ein Keras-Modell in eine TFLite-Datei (fuer schnelle Inferenz).

    quantisieren=True: Gewichte als 8-Bit-Ganzzahlen -> ca. 4x kleiner.
    """
    import tempfile
    import tensorflow as tf

    pfad = Path(pfad)
    try:
        konverter = tf.lite.TFLiteConverter.from_keras_model(modell)
        if quantisieren:
            konverter.optimizations = [tf.lite.Optimize.DEFAULT]
        daten = konverter.convert()
    except Exception:
        # Weg ueber SavedModel (robuster bei Keras 3)
        with tempfile.TemporaryDirectory() as tmp:
            modell.export(tmp)
            konverter = tf.lite.TFLiteConverter.from_saved_model(tmp)
            if quantisieren:
                konverter.optimizations = [tf.lite.Optimize.DEFAULT]
            daten = konverter.convert()
    pfad.write_bytes(daten)
    return pfad


def _interpreter_klasse():
    """Sucht die schlankeste verfuegbare TFLite-Laufzeit (LiteRT).

    Ab TensorFlow 2.20 gibt es tf.lite.Interpreter nicht mehr - dann wird
    das Paket ai-edge-litert gebraucht (pip install ai-edge-litert).
    """
    try:
        from ai_edge_litert.interpreter import Interpreter   # LiteRT (empfohlen)
        return Interpreter
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter   # aeltere Laufzeit
        return Interpreter
    except ImportError:
        pass
    try:
        import tensorflow as tf
        return tf.lite.Interpreter                           # nur TensorFlow < 2.20
    except (ImportError, AttributeError):
        raise ImportError("Keine TFLite-Laufzeit gefunden: pip install ai-edge-litert") from None


def tflite_verfuegbar() -> bool:
    """True, wenn eine TFLite-Laufzeit installiert ist."""
    try:
        _interpreter_klasse()
        return True
    except ImportError:
        return False


def _lade_tflite_interpreter(pfad: str):
    Interpreter = _interpreter_klasse()
    try:
        return Interpreter(model_path=pfad, num_threads=4)
    except TypeError:
        return Interpreter(model_path=pfad)


class Klassifikator:
    """Laedt ein gespeichertes Modell und klassifiziert Kamerabilder.

    Beispiel:
        k = Klassifikator("modelle/cnn_info.json")
        wahrsch = k.vorhersage(bild_bgr)       # Array mit einer Zahl je Klasse
    """

    def __init__(self, info_datei: str | Path, bevorzuge_tflite: bool = True):
        info_datei = Path(info_datei)
        if not info_datei.is_absolute() and not info_datei.exists():
            info_datei = ORDNER / info_datei
        self.info = json.loads(info_datei.read_text(encoding="utf-8"))
        self.klassen = self.info["klassen"]
        self.groesse = self.info["groesse"]
        self.farbe = self.info["farbe"]
        ordner = info_datei.parent
        self.interpreter = None
        self.modell = None
        tflite_datei = ordner / self.info.get("tflite", "-")
        if bevorzuge_tflite and tflite_datei.is_file() and not tflite_verfuegbar():
            print("Hinweis: keine TFLite-Laufzeit installiert (pip install ai-edge-litert) "
                  "- es wird das Keras-Modell verwendet.")
        if bevorzuge_tflite and tflite_datei.is_file() and tflite_verfuegbar():
            self.interpreter = _lade_tflite_interpreter(str(tflite_datei))
            self.interpreter.allocate_tensors()
            self._ein = self.interpreter.get_input_details()[0]
            self._aus = self.interpreter.get_output_details()[0]
            self.art = "TFLite"
        else:
            import keras
            self.modell = keras.models.load_model(ordner / self.info["keras"])
            self.art = "Keras"

    def vorhersage(self, bild_bgr: np.ndarray) -> np.ndarray:
        x = bild_vorbereiten(bild_bgr, self.groesse, self.farbe)
        x = x[np.newaxis].astype(np.float32)          # Form (1, H, W, C)
        if self.interpreter is None:
            return self.modell.predict(x, verbose=0)[0]
        ein, aus = self._ein, self._aus
        if ein["dtype"] in (np.uint8, np.int8):       # vollquantisiertes Modell
            skala, null = ein["quantization"]
            x = np.clip(np.round(x / skala + null), np.iinfo(ein["dtype"]).min,
                        np.iinfo(ein["dtype"]).max)
        self.interpreter.set_tensor(ein["index"], x.astype(ein["dtype"]))
        self.interpreter.invoke()
        y = self.interpreter.get_tensor(aus["index"])[0]
        if aus["dtype"] in (np.uint8, np.int8):
            skala, null = aus["quantization"]
            y = (y.astype(np.float32) - null) * skala
        return y


def messe_inferenzzeit(funktion, eingabe, wiederholungen: int = 20) -> float:
    """Mittlere Rechenzeit einer Vorhersage in Millisekunden."""
    funktion(eingabe)                                    # Aufwaermen
    start = time.perf_counter()
    for _ in range(wiederholungen):
        funktion(eingabe)
    return (time.perf_counter() - start) / wiederholungen * 1000


def ram_aufraeumen():
    """Gibt Speicher frei - wichtig auf dem Raspberry Pi zwischen Versuchen."""
    import gc
    try:
        import keras
        keras.backend.clear_session()
    except ImportError:
        pass
    gc.collect()
    try:
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass
