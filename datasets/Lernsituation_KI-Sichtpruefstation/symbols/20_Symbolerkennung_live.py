#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Erstellt von:   Kai Metzger (Franz-Oberthuer-Schule Wuerzburg)
# Erstellt am:    07.04.2025
# Ueberarbeitet:  November 2026 - Lernsituation "KI-Sichtpruefstation"
# Version:        2.0.0
# =============================================================================
"""20 - KI-Sichtpruefstation im Dauerbetrieb (Raspberry Pi 4/5).

Laedt ein in Notebook 11, 12 oder 13 gespeichertes Modell (bevorzugt TFLite),
klassifiziert jedes Kamerabild, zeigt das Ergebnis an, steuert optional eine
LED-Ampel ueber GPIO und schreibt ein Pruefprotokoll.

Beispiele:
    python3 20_Symbolerkennung_live.py --modell cnn
    python3 20_Symbolerkennung_live.py --modell transfer --schwelle 0.9 --gpio --ausschuss Kreuz
    python3 20_Symbolerkennung_live.py --modell cnn --kamera picamera --intervall 0.5

Tasten im Fenster:
    ESC / q   Beenden
    p         Pause an/aus
    + / -     Konfidenzschwelle um 5 Prozentpunkte aendern
    s         aktuelles Kamerabild speichern (protokoll/aufnahmen/)
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from ki_hilfen import Kamera, Klassifikator

ORDNER = Path(__file__).resolve().parent


# =============================================================================
# Ampel (GPIO) - wird ohne Hardware automatisch simuliert
# =============================================================================
class Ampel:
    def __init__(self, aktiv: bool, rot=17, gelb=27, gruen=22):
        self.leds = {}
        if not aktiv:
            return
        try:
            from gpiozero import LED
            self.leds = {"rot": LED(rot), "gelb": LED(gelb), "gruen": LED(gruen)}
            print("GPIO-Ampel aktiv (BCM 17 rot, 27 gelb, 22 gruen)")
        except Exception as e:                                # noqa: BLE001
            print("GPIO nicht verfuegbar, Ampel nur auf dem Bildschirm:", e)

    def zeige(self, farbe: str | None):
        for name, led in self.leds.items():
            led.on() if name == farbe else led.off()

    def aus(self):
        self.zeige(None)


FARBEN_BGR = {"gruen": (60, 170, 60), "gelb": (0, 190, 255), "rot": (40, 40, 220)}


# =============================================================================
# Anzeige
# =============================================================================
def zeichne(bild, klassen, wahrsch, ergebnis, farbe, schwelle, ms, fps, pause):
    h, w = bild.shape[:2]
    leiste = 260
    anzeige = np.full((h, w + leiste, 3), 245, dtype=np.uint8)
    anzeige[:, :w] = bild

    # Ergebnisbalken oben
    cv2.rectangle(anzeige, (0, 0), (w, 56), FARBEN_BGR[farbe], -1)
    text = ergebnis if ergebnis != "unsicher" else "UNSICHER - manuell pruefen"
    cv2.putText(anzeige, ascii_text(text), (14, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (255, 255, 255), 2, cv2.LINE_AA)

    # Wahrscheinlichkeiten je Klasse
    x0 = w + 14
    cv2.putText(anzeige, "Wahrscheinlichkeit", (x0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 60), 1)
    for i, (name, p) in enumerate(zip(klassen, wahrsch)):
        y = 60 + i * 42
        cv2.putText(anzeige, ascii_text(name)[:16], (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1)
        cv2.rectangle(anzeige, (x0, y + 6), (x0 + 200, y + 22), (220, 220, 220), -1)
        cv2.rectangle(anzeige, (x0, y + 6), (x0 + int(200 * p), y + 22),
                      (160, 95, 31) if p < schwelle else (60, 170, 60), -1)
        cv2.putText(anzeige, f"{p:5.1%}", (x0 + 150, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1)
    # Schwelle als Linie
    y_ende = 60 + len(klassen) * 42
    xs = x0 + int(200 * schwelle)
    cv2.line(anzeige, (xs, 50), (xs, y_ende), (0, 0, 0), 1)

    info = [f"Schwelle: {schwelle:.0%}  (+/-)", f"Inferenz: {ms:.0f} ms", f"Bilder/s: {fps:.1f}"]
    if pause:
        info.append("PAUSE (p)")
    for k, t in enumerate(info):
        cv2.putText(anzeige, t, (x0, y_ende + 30 + 24 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1)
    return anzeige


def ascii_text(text: str) -> str:
    for alt, neu in {"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue", "ß": "ss"}.items():
        text = text.replace(alt, neu)
    return text.encode("ascii", "replace").decode()


# =============================================================================
# Hauptprogramm
# =============================================================================
def main():
    p = argparse.ArgumentParser(description="KI-Sichtpruefstation (Live-Erkennung)")
    p.add_argument("--modell", default="cnn", help="Name (fcnn, cnn, transfer) oder Pfad zur *_info.json")
    p.add_argument("--kamera", default="0", help="0, 1, ... (USB), picamera oder Videodatei")
    p.add_argument("--aufloesung", default="640x480")
    p.add_argument("--schwelle", type=float, default=0.80, help="Mindest-Konfidenz (0 ... 1)")
    p.add_argument("--ausschuss", nargs="*", default=[], help="Klassen, die 'rot' bedeuten")
    p.add_argument("--gpio", action="store_true", help="LED-Ampel ueber GPIO ansteuern")
    p.add_argument("--intervall", type=float, default=0.0,
                   help="Sekunden zwischen zwei Pruefungen (0 = jedes Bild)")
    p.add_argument("--keras", action="store_true", help="Keras statt TFLite verwenden")
    args = p.parse_args()

    info = Path(args.modell)
    if info.suffix != ".json":
        info = ORDNER / "modelle" / f"{args.modell}_info.json"
    if not info.exists():
        raise SystemExit(f"Modell nicht gefunden: {info}\nBitte zuerst Notebook 11, 12 oder 13 ausfuehren.")
    klassifikator = Klassifikator(info, bevorzuge_tflite=not args.keras)
    klassen = klassifikator.klassen
    print(f"Modell: {info.name} ({klassifikator.art}) | Klassen: {klassen} | "
          f"Bild: {klassifikator.groesse} {klassifikator.farbe}")
    unbekannt = set(args.ausschuss) - set(klassen)
    if unbekannt:
        raise SystemExit(f"Unbekannte Ausschuss-Klassen: {unbekannt}. Vorhanden: {klassen}")

    protokoll = ORDNER / "protokoll"
    (protokoll / "unsicher").mkdir(parents=True, exist_ok=True)
    (protokoll / "aufnahmen").mkdir(exist_ok=True)
    csv = protokoll / f"pruefprotokoll_{datetime.now():%Y%m%d}.csv"
    if not csv.exists():
        csv.write_text("zeit;modell;ergebnis;klasse;konfidenz\n", encoding="utf-8")

    ampel = Ampel(args.gpio)
    schwelle, pause = args.schwelle, False
    breite, hoehe = (int(v) for v in args.aufloesung.lower().split("x"))
    letzte_pruefung, fps, t_vorher = 0.0, 0.0, time.time()
    wahrsch = np.zeros(len(klassen))
    ergebnis, farbe, ms = "-", "gelb", 0.0
    anzahl = {"gesamt": 0, "unsicher": 0}

    print(__doc__.split("Tasten im Fenster:")[1])
    with Kamera(args.kamera, breite, hoehe) as kamera, open(csv, "a", encoding="utf-8") as log:
        cv2.namedWindow("KI-Sichtpruefstation", cv2.WINDOW_NORMAL)
        try:
            while True:
                bild = kamera.lies()
                if bild is None:
                    print("Kein Kamerabild - Ende.")
                    break
                jetzt = time.time()
                if not pause and jetzt - letzte_pruefung >= args.intervall:
                    t0 = time.perf_counter()
                    wahrsch = klassifikator.vorhersage(bild)
                    ms = (time.perf_counter() - t0) * 1000
                    nummer = int(np.argmax(wahrsch))
                    konfidenz = float(wahrsch[nummer])
                    if konfidenz < schwelle:
                        ergebnis, farbe = "unsicher", "gelb"
                    else:
                        ergebnis = klassen[nummer]
                        farbe = "rot" if ergebnis in args.ausschuss else "gruen"
                    ampel.zeige(farbe)
                    zeit = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    log.write(f"{zeit};{info.stem.replace('_info', '')};{ergebnis};"
                              f"{klassen[nummer]};{konfidenz:.3f}\n")
                    anzahl["gesamt"] += 1
                    if ergebnis == "unsicher":
                        anzahl["unsicher"] += 1
                        if args.intervall > 0:   # im Dauerbetrieb nicht jedes Bild speichern
                            cv2.imwrite(str(protokoll / "unsicher" /
                                            f"{zeit.replace(':', '-').replace(' ', '_')}.png"), bild)
                    letzte_pruefung = jetzt

                dt = jetzt - t_vorher
                t_vorher = jetzt
                fps = 0.9 * fps + 0.1 * (1 / dt if dt > 0 else 0)
                cv2.imshow("KI-Sichtpruefstation",
                           zeichne(bild, klassen, wahrsch, ergebnis, farbe, schwelle, ms, fps, pause))

                taste = cv2.waitKey(1) & 0xFF
                if taste in (27, ord("q")):
                    break
                if taste == ord("p"):
                    pause = not pause
                elif taste in (ord("+"), ord("=")):
                    schwelle = min(0.99, schwelle + 0.05)
                elif taste == ord("-"):
                    schwelle = max(0.05, schwelle - 0.05)
                elif taste == ord("s"):
                    datei = protokoll / "aufnahmen" / f"{datetime.now():%Y%m%d_%H%M%S}.png"
                    cv2.imwrite(str(datei), bild)
                    print("gespeichert:", datei)
        except KeyboardInterrupt:
            pass
        finally:
            ampel.aus()
            cv2.destroyAllWindows()

    print(f"\n{anzahl['gesamt']} Pruefungen, davon {anzahl['unsicher']} unsicher. Protokoll: {csv}")


if __name__ == "__main__":
    main()
