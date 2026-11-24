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
    datensatz = Path(datensatz)
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
    ordner = Path(ordner)
    if not ordner.exists():
        return []
    return sorted(p for p in ordner.iterdir()
                  if p.is_file() and p.suffix.lower() in BILDENDUNGEN)


def zaehle_bilder(datensatz: str | Path) -> dict[str, int]:
    """Anzahl Bilder je Klasse, z. B. {'Kreuz': 432, 'Kreis': 430, ...}."""
    datensatz = Path(datensatz)
    return {name: len(bilddateien(datensatz / str(i)))
            for i, name in enumerate(lese_klassen(datensatz))}


# =============================================================================
# Bildvorbereitung - IDENTISCH fuer Training und Inferenz
# =============================================================================
def bild_vorbereiten(bild_bgr: np.ndarray, groesse: int | tuple[int, int],
                     farbe: str = "rgb") -> np.ndarray:
    """Skaliert ein OpenCV-Bild (BGR) auf groesse x groesse.

    Rueckgabe: uint8-Array mit Form (H, W, 3) bei farbe="rgb"
               bzw. (H, W, 1) bei farbe="grau".
    Werte bleiben 0..255 - die Normierung passiert IM Modell (Rescaling).
    """
    if isinstance(groesse, int):
        groesse = (groesse, groesse)
    hoehe, breite = groesse
    # INTER_AREA ist beim Verkleinern am saubersten (kein Treppeneffekt)
    klein = cv2.resize(bild_bgr, (breite, hoehe), interpolation=cv2.INTER_AREA)
    if farbe == "grau":
        return cv2.cvtColor(klein, cv2.COLOR_BGR2GRAY)[..., np.newaxis]
    if farbe == "rgb":
        return cv2.cvtColor(klein, cv2.COLOR_BGR2RGB)
    raise ValueError("farbe muss 'rgb' oder 'grau' sein")


# =============================================================================
# Datensatz laden (mit Zwischenspeicher)
# =============================================================================
def _cache_schluessel(dateien: list[Path], groesse, farbe) -> str:
    h = hashlib.sha1()
    for p in dateien:
        h.update(str(p).encode())
        h.update(str(p.stat().st_size).encode())
    h.update(f"{groesse}-{farbe}".encode())
    return h.hexdigest()[:12]


def lade_datensatz(datensatz: str | Path, groesse: int = 32, farbe: str = "rgb",
                   zwischenspeichern: bool = True):
    """Laedt alle Bilder eines Datensatzes in den Arbeitsspeicher.

    Rueckgabe: X (uint8, Form N x H x W x C), y (int, Klassennummer), klassen
    Beim ersten Mal werden die Bilder einzeln gelesen (auf dem Pi bei
    1920x1080-Bildern ca. 1-2 Minuten). Danach liegt eine .npz-Datei im Ordner
    .cache und das Laden dauert nur noch Sekunden.
    """
    datensatz = Path(datensatz)
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


def als_tf_dataset(X, y, batch_size=32, mischen=False, augmentierung=None,
                   seed: int = 42):
    """Macht aus NumPy-Arrays ein tf.data.Dataset (Stapel = Batches).

    augmentierung: optionales keras.Sequential, das NUR auf die Trainingsdaten
    angewendet wird. Es ist bewusst nicht Teil des Modells - so bleibt das
    exportierte Modell schlank und laeuft problemlos als TFLite auf dem Pi.
    """
    import tensorflow as tf

    # Bilder bleiben uint8 (spart 3/4 des Arbeitsspeichers) und werden erst
    # stapelweise in float32 umgewandelt.
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if mischen:
        ds = ds.shuffle(len(X), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size).map(lambda b, l: (tf.cast(b, tf.float32), l))
    if augmentierung is not None:
        ds = ds.map(lambda b, l: (augmentierung(b, training=True), l),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(1)    # 1 Stapel Vorlauf schont den RAM des Pi


# =============================================================================
# Diagramme
# =============================================================================
def zeige_beispiele(X, y, klassen, anzahl: int = 12, vorhersage=None, seed=0):
    """Zeigt zufaellige Bilder mit Klassenname (und ggf. Vorhersage)."""
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(seed)
    auswahl = rng.choice(len(X), size=min(anzahl, len(X)), replace=False)
    spalten = 6
    zeilen = int(np.ceil(len(auswahl) / spalten))
    plt.figure(figsize=(2.2 * spalten, 2.4 * zeilen))
    for k, i in enumerate(auswahl):
        plt.subplot(zeilen, spalten, k + 1)
        bild = X[i]
        if bild.ndim == 3 and bild.shape[-1] == 1:
            plt.imshow(bild[..., 0], cmap="gray", vmin=0, vmax=255)
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


# =============================================================================
# Interaktive HTML-Seiten im Notebook
# =============================================================================
def zeige_html(datei: str | Path, hoehe: int = 650):
    """Bettet eine (eigenstaendige) HTML-Seite aus material/ ins Notebook ein.

    Die Seite wird als <iframe srcdoc=...> eingebunden. Das funktioniert in
    JupyterLab, Jupyter Notebook und VS Code - auch ohne Internet.
    Alternativ kann jede Datei direkt im Browser geoeffnet werden.
    """
    from IPython.display import HTML, display

    pfad = Path(datei)
    if not pfad.is_absolute() and not pfad.exists():
        pfad = ORDNER / "material" / pfad
    inhalt = pfad.read_text(encoding="utf-8")
    display(HTML(
        f'<iframe srcdoc="{html.escape(inhalt, quote=True)}" '
        f'style="width:100%;height:{hoehe}px;border:1px solid #ccc;'
        f'border-radius:8px;background:white" '
        f'sandbox="allow-scripts allow-same-origin allow-popups allow-modals">'
        f'</iframe>'
        f'<div style="font-size:12px;color:#666">Datei: material/{pfad.name} '
        f'- laesst sich auch direkt im Browser oeffnen.</div>'))


# =============================================================================
# Selbstkontrolle in den Notebooks
# =============================================================================
def pruefe(bedingung, ok: str, fehler: str) -> bool:
    """Gibt eine gut sichtbare Rueckmeldung zu einer Teilaufgabe aus."""
    try:
        erfuellt = bool(bedingung() if callable(bedingung) else bedingung)
    except Exception as e:                       # noqa: BLE001
        erfuellt = False
        fehler = f"{fehler}  (Fehler: {type(e).__name__}: {e})"
    print(("✅ " + ok) if erfuellt else ("❌ " + fehler))
    return erfuellt


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


def _lade_tflite_interpreter(pfad: str):
    """Sucht die schlankeste verfuegbare TFLite-Laufzeit."""
    try:
        from ai_edge_litert.interpreter import Interpreter   # neu (LiteRT)
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
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
        if bevorzuge_tflite and "tflite" in self.info and (ordner / self.info["tflite"]).exists():
            self.interpreter = _lade_tflite_interpreter(str(ordner / self.info["tflite"]))
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
