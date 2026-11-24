#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Erstellt von:   Kai Metzger (Franz-Oberthuer-Schule Wuerzburg)
# Erstellt am:    23.02.2025
# Ueberarbeitet:  November 2026 - Lernsituation "KI-Sichtpruefstation"
# Version:        2.0.0
# =============================================================================
"""00 - Datensatz mit der Kamera aufnehmen (Raspberry Pi 4/5 + USB-Kamera).

Jedes Bild wird direkt im Ordner seiner Klasse gespeichert:

    dataset_neu/
      klassen.txt         <- 0;Kreuz  1;Kreis  2;Dreieck  3;Quadrat
      0/  1/  2/  3/      <- Bilder der jeweiligen Klasse

Die Klassen sind frei waehlbar - entweder unten unter KONFIGURATION oder beim
Start im Terminal, z. B. fuer eine Teilesortierung:

    python3 00_Datensatz_aufnehmen.py --datensatz teile --klassen Schraube Mutter Scheibe Leer

Gibt es den Datensatz schon, werden die Klassen aus klassen.txt gelesen und
die Aufnahme einfach fortgesetzt.

Tasten (das Kamerafenster muss aktiv sein):
    0 ... 9        Bild sofort in Klasse 0 ... 9 speichern
    n / p          naechste / vorherige Klasse waehlen (fuer mehr als 10 Klassen)
    LEERTASTE      Bild in der gewaehlten Klasse speichern
    s              Serienaufnahme an/aus (alle --intervall Sekunden ein Bild)
    z / RUECKTASTE letzte Aufnahme rueckgaengig machen (Datei wird geloescht)
    h              Hilfe einblenden / ausblenden
    ESC / q        Beenden
"""

# =============================================================================
# Importe
# =============================================================================
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2

from ki_hilfen import Kamera, bilddateien, lese_klassen, schreibe_klassen

# =============================================================================
# KONFIGURATION - Standardwerte (koennen auch beim Start uebergeben werden)
# =============================================================================
DATENSATZ = "dataset_neu"                       # Ordnername (neben diesem Skript)
KLASSEN = ["Kreuz", "Kreis", "Dreieck", "Quadrat"]
KAMERA = "0"                                    # 0 = erste USB-Kamera, "picamera"
AUFLOESUNG = "640x480"                          # z. B. 1920x1080 bei HD-Kameras
SERIEN_INTERVALL = 0.5                          # Sekunden zwischen Serienbildern

ORDNER = Path(__file__).resolve().parent


# =============================================================================
# Hilfsfunktionen
# =============================================================================
def ascii_text(text: str) -> str:
    """OpenCV kann keine Umlaute zeichnen -> ae, oe, ue, ss."""
    for alt, neu in {"ä": "ae", "ö": "oe", "ü": "ue", "Ä": "Ae", "Ö": "Oe",
                     "Ü": "Ue", "ß": "ss"}.items():
        text = text.replace(alt, neu)
    return text.encode("ascii", "replace").decode()


def datensatz_vorbereiten(pfad: Path, klassen_wunsch: list[str] | None) -> list[str]:
    """Legt den Datensatz an oder prueft einen vorhandenen."""
    if (pfad / "klassen.txt").exists():
        vorhanden = lese_klassen(pfad)
        if klassen_wunsch and klassen_wunsch != vorhanden:
            sys.exit(f"Der Datensatz {pfad.name} hat bereits die Klassen {vorhanden}.\n"
                     f"Gewuenscht waren {klassen_wunsch}. Bitte anderen Namen "
                     "waehlen oder --klassen weglassen.")
        schreibe_klassen(pfad, vorhanden)        # fehlende Ordner ergaenzen
        print(f"Vorhandener Datensatz wird fortgesetzt: {pfad}")
        return vorhanden
    klassen = klassen_wunsch or KLASSEN
    if len(set(klassen)) != len(klassen):
        sys.exit("Klassennamen muessen eindeutig sein.")
    schreibe_klassen(pfad, klassen)
    print(f"Neuer Datensatz angelegt: {pfad}")
    return klassen


def zaehle(pfad: Path, klassen: list[str]) -> list[int]:
    return [len(bilddateien(pfad / str(i))) for i in range(len(klassen))]


def speichere(bild, pfad: Path, klasse: int) -> Path:
    zeit = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    datei = pfad / str(klasse) / f"{klasse}_{zeit}.png"
    if not cv2.imwrite(str(datei), bild):
        raise OSError(f"Konnte {datei} nicht schreiben")
    return datei


def taste_zu_ziffer(taste: int) -> int | None:
    """Ziffern der Haupttastatur UND des Ziffernblocks erkennen."""
    if ord("0") <= taste <= ord("9"):
        return taste - ord("0")
    if 176 <= taste <= 185:              # Ziffernblock unter Linux/GTK (0xB0...)
        return taste - 176
    return None


def zeichne_overlay(bild, klassen, anzahl, aktiv, serie, hilfe, blitz, meldung):
    """Zeichnet Statusinformationen auf eine KOPIE des Kamerabildes."""
    anzeige = bild.copy()
    h, w = anzeige.shape[:2]
    skala = max(0.5, w / 1280)
    dicke = max(1, int(2 * skala))

    zeilen = [f"Klasse [{aktiv}]: {ascii_text(klassen[aktiv])}"
              + ("   SERIE AKTIV" if serie else "")]
    zeilen += [f"{'>' if i == aktiv else ' '} {i}: {ascii_text(n)} = {anzahl[i]}"
               for i, n in enumerate(klassen)]
    if hilfe:
        zeilen += ["", "0-9: speichern in Klasse  |  n/p: Klasse waehlen",
                   "Leertaste: speichern  |  s: Serie  |  z: rueckgaengig",
                   "h: Hilfe  |  ESC/q: Ende"]
    if meldung:
        zeilen += ["", meldung]

    zeilenhoehe = int(32 * skala)
    hoehe_box = zeilenhoehe * (len(zeilen) + 1)
    breite_box = int(min(w - 20, 760 * skala))
    overlay = anzeige.copy()
    cv2.rectangle(overlay, (10, 10), (10 + breite_box, 10 + hoehe_box), (0, 0, 0), -1)
    anzeige = cv2.addWeighted(overlay, 0.55, anzeige, 0.45, 0)
    for k, text in enumerate(zeilen):
        farbe = (0, 255, 255) if k == 0 else (255, 255, 255)
        cv2.putText(anzeige, text, (20, 10 + zeilenhoehe * (k + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7 * skala, farbe, dicke, cv2.LINE_AA)
    if blitz:                                     # sichtbare Rueckmeldung
        cv2.rectangle(anzeige, (0, 0), (w - 1, h - 1), (0, 255, 0), int(12 * skala))
    return anzeige


# =============================================================================
# Hauptprogramm
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Bilddatensatz mit Klassenordnern aufnehmen")
    parser.add_argument("--datensatz", default=DATENSATZ,
                        help="Ordnername oder Pfad des Datensatzes")
    parser.add_argument("--klassen", nargs="+", default=None,
                        help="Klassennamen in Reihenfolge 0, 1, 2, ...")
    parser.add_argument("--kamera", default=KAMERA,
                        help="0, 1, ... (USB), picamera oder Videodatei")
    parser.add_argument("--aufloesung", default=AUFLOESUNG, help="z. B. 640x480")
    parser.add_argument("--intervall", type=float, default=SERIEN_INTERVALL,
                        help="Sekunden zwischen Serienbildern")
    args = parser.parse_args()

    pfad = Path(args.datensatz)
    if not pfad.is_absolute():
        pfad = ORDNER / pfad
    klassen = datensatz_vorbereiten(pfad, args.klassen)
    breite, hoehe = (int(v) for v in args.aufloesung.lower().split("x"))

    print("Tasten" + __doc__.split("Tasten", 1)[1])
    anzahl = zaehle(pfad, klassen)
    aktiv, serie, hilfe = 0, False, True
    letzte = []                       # Liste gespeicherter Dateien (fuer "z")
    letzte_serie = 0.0
    blitz_bis = 0.0
    meldung, meldung_bis = "", 0.0

    with Kamera(args.kamera, breite, hoehe) as kamera:
        print("Kameraaufloesung:", kamera.aufloesung())
        cv2.namedWindow("Datensatz aufnehmen", cv2.WINDOW_NORMAL)
        try:
            while True:
                bild = kamera.lies()
                if bild is None:
                    print("Kein Bild von der Kamera - Ende.")
                    break
                jetzt = time.time()
                speichern_in = None

                taste = cv2.waitKey(1) & 0xFF
                ziffer = taste_zu_ziffer(taste)
                if taste in (27, ord("q")):
                    break
                elif ziffer is not None:
                    if ziffer < len(klassen):
                        aktiv = ziffer
                        speichern_in = ziffer
                    else:
                        meldung, meldung_bis = f"Klasse {ziffer} gibt es nicht", jetzt + 2
                elif taste == ord("n"):
                    aktiv = (aktiv + 1) % len(klassen)
                elif taste == ord("p"):
                    aktiv = (aktiv - 1) % len(klassen)
                elif taste == 32:                              # Leertaste
                    speichern_in = aktiv
                elif taste == ord("s"):
                    serie = not serie
                    print("Serienaufnahme", "AN" if serie else "AUS")
                elif taste in (ord("z"), 8):                   # z / Ruecktaste
                    if letzte:
                        datei, k = letzte.pop()
                        datei.unlink(missing_ok=True)
                        anzahl[k] -= 1
                        meldung, meldung_bis = f"Geloescht: {datei.name}", jetzt + 2
                        print("Rueckgaengig:", datei)
                elif taste == ord("h"):
                    hilfe = not hilfe

                if serie and jetzt - letzte_serie >= args.intervall:
                    speichern_in = aktiv
                    letzte_serie = jetzt

                if speichern_in is not None:
                    datei = speichere(bild, pfad, speichern_in)
                    letzte.append((datei, speichern_in))
                    anzahl[speichern_in] += 1
                    blitz_bis = jetzt + 0.15
                    print(f"[{klassen[speichern_in]}] {datei.name}  "
                          f"(jetzt {anzahl[speichern_in]} Bilder)")

                anzeige = zeichne_overlay(bild, klassen, anzahl, aktiv, serie, hilfe,
                                          jetzt < blitz_bis,
                                          meldung if jetzt < meldung_bis else "")
                cv2.imshow("Datensatz aufnehmen", anzeige)
        except KeyboardInterrupt:
            print("Abgebrochen (Strg+C).")
        finally:
            cv2.destroyAllWindows()

    # Zusammenfassung
    anzahl = zaehle(pfad, klassen)
    print("\nDatensatz:", pfad)
    for i, name in enumerate(klassen):
        print(f"  Klasse {i} ({name}): {anzahl[i]} Bilder")
    if max(anzahl) > 0 and min(anzahl) < 0.8 * max(anzahl):
        print("Hinweis: Die Klassen sind ungleich gross. Fuer ein faires Training "
              "moeglichst gleich viele Bilder je Klasse aufnehmen.")


if __name__ == "__main__":
    main()
