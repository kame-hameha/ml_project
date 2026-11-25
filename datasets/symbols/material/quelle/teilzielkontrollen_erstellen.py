#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Erzeugt die interaktiven Teilzielkontrollen (tzk*.html) im Ordner material/.

Fragen aendern oder ergaenzen: unten in TEILZIELKONTROLLEN bearbeiten und dann

    python3 material/quelle/teilzielkontrollen_erstellen.py

ausfuehren. Die erzeugten HTML-Dateien sind eigenstaendig (kein Internet noetig).

Fragetypen:
  einfach      genau eine richtige Antwort       richtig = Index
  mehrfach     mehrere richtige Antworten        richtig = [Indizes]
  zahl         Zahlenwert mit Toleranz           richtig, toleranz, einheit
  zuordnung    Paare zuordnen                    paare = [[links, rechts], ...]
  reihenfolge  Schritte sortieren                schritte = [in richtiger Reihenfolge]
"""

import json
from pathlib import Path

ZIEL = Path(__file__).resolve().parents[1]

TEILZIELKONTROLLEN = {
    "tzk1_daten_und_bilder": {
        "titel": "Teilzielkontrolle 1 – KI-Grundlagen, Bilddaten und Datensatz",
        "ds": "nach Doppelstunde 2",
        "fragen": [
            {"typ": "einfach", "frage": "Was beschreibt <b>überwachtes Lernen</b> am besten?",
             "optionen": ["Das Modell lernt aus Beispielen, zu denen die richtige Antwort (Label) bekannt ist.",
                          "Das Modell sucht ohne Vorgaben selbstständig Gruppen in den Daten.",
                          "Ein Programmierer schreibt Regeln wie „wenn 4 Ecken, dann Quadrat“.",
                          "Das Modell lernt durch Belohnung und Bestrafung in einer Simulation."],
             "richtig": 0,
             "erklaerung": "Bei uns ist zu jedem Bild bekannt, welche Klasse es zeigt – der Ordnername ist das Label."},
            {"typ": "zuordnung", "frage": "Ordnen Sie die Begriffe der Prüfstation zu.",
             "paare": [["Eingabe (Merkmale)", "die Pixelwerte des Kamerabildes"],
                       ["Label", "die Klassennummer, z. B. 2 = Dreieck"],
                       ["Modell", "das neuronale Netz mit seinen Gewichten"],
                       ["Inferenz", "das trainierte Netz auf ein neues Bild anwenden"]],
             "erklaerung": "Training = aus Eingaben und Labels lernen. Inferenz = später im Betrieb vorhersagen."},
            {"typ": "zahl", "frage": "Wie viele Zahlen enthält ein RGB-Bild mit 96 × 96 Pixeln?",
             "richtig": 27648, "toleranz": 0, "einheit": "Zahlen",
             "erklaerung": "96 · 96 · 3 Kanäle = 27 648."},
            {"typ": "einfach", "frage": "Ein Pixel im Graustufenbild (uint8) hat den Wert <span class='mono'>0</span>. Was bedeutet das?",
             "optionen": ["schwarz", "weiß", "mittelgrau", "Pixel fehlt"], "richtig": 0,
             "erklaerung": "0 = schwarz, 255 = weiß."},
            {"typ": "mehrfach", "frage": "Welche Maßnahmen verbessern den Datensatz für die Prüfstation? (mehrere richtig)",
             "optionen": ["Bilder unter derselben Beleuchtung aufnehmen, die später im Betrieb herrscht",
                          "Symbole in verschiedenen Lagen, Drehungen und Größen aufnehmen",
                          "ungefähr gleich viele Bilder je Klasse aufnehmen",
                          "alle Bilder einer Klasse mit exakt derselben Position aufnehmen",
                          "die Testbilder zusätzlich zum Training verwenden"],
             "richtig": [0, 1, 2],
             "erklaerung": "Vielfalt und Ausgewogenheit helfen. Identische Bilder bringen keine neue Information, und Testbilder im Training verfälschen die Bewertung."},
            {"typ": "einfach", "frage": "Wozu dient der <b>Testdatensatz</b>?",
             "optionen": ["für eine einmalige, unabhängige Endbewertung nach der Modellwahl",
                          "um nach jeder Epoche die Lernrate anzupassen",
                          "als zusätzliche Trainingsdaten, wenn die Genauigkeit zu niedrig ist",
                          "um EarlyStopping zu steuern"],
             "richtig": 0,
             "erklaerung": "Validierungsdaten steuern das Training (EarlyStopping, Hyperparameter). Testdaten bleiben bis zum Schluss unangetastet."},
            {"typ": "zahl", "frage": "Der Datensatz hat 1735 Bilder und wird 70 / 15 / 15 aufgeteilt. Wie viele Bilder etwa sind im <b>Trainings</b>datensatz?",
             "richtig": 1214, "toleranz": 3, "einheit": "Bilder",
             "erklaerung": "0,70 · 1735 ≈ 1214."},
            {"typ": "einfach", "frage": "Wo speichert <span class='mono'>00_Datensatz_aufnehmen.py</span> ein Bild, wenn Sie die Taste <b>2</b> drücken?",
             "optionen": ["<span class='mono'>dataset_neu/2/2_&lt;Zeitstempel&gt;.png</span>",
                          "<span class='mono'>dataset_neu/data/2.png</span> und das Label in <span class='mono'>gt/2.txt</span>",
                          "<span class='mono'>dataset_neu/Dreieck.png</span>",
                          "in <span class='mono'>klassen.txt</span>"],
             "richtig": 0,
             "erklaerung": "Jede Klasse hat einen eigenen Ordner. Der Ordnername ist das Label, klassen.txt enthält nur die Namen."},
            {"typ": "einfach", "frage": "Warum werden die Pixelwerte vor dem Training durch 255 geteilt?",
             "optionen": ["Kleine Eingabewerte (0 … 1) machen das Training stabiler und schneller.",
                          "Damit das Bild weniger Speicher braucht.",
                          "Damit aus dem Farbbild ein Graustufenbild wird.",
                          "Weil Keras keine Zahlen über 1 verarbeiten kann."],
             "richtig": 0,
             "erklaerung": "Normierung bringt alle Eingaben in denselben kleinen Bereich. Bei uns übernimmt das die Schicht Rescaling(1/255) im Modell."},
            {"typ": "reihenfolge", "frage": "Bringen Sie die Schritte der KI-Pipeline in die richtige Reihenfolge.",
             "schritte": ["Bilder mit der Kamera aufnehmen und beschriften",
                          "Datensatz aufteilen und vorbereiten",
                          "Modell trainieren",
                          "Modell mit Testdaten bewerten",
                          "Modell exportieren (TFLite)",
                          "Inferenz an der Prüfstation"],
             "erklaerung": "Diese Reihenfolge entspricht den Notebooks 00 → 10–13 → 20."},
        ],
    },
    "tzk2_neuronale_netze": {
        "titel": "Teilzielkontrolle 2 – Neuron, FCNN und Training",
        "ds": "nach Doppelstunde 5",
        "fragen": [
            {"typ": "einfach", "frage": "Was berechnet ein künstliches Neuron?",
             "optionen": ["gewichtete Summe der Eingänge plus Bias, danach die Aktivierungsfunktion",
                          "den Mittelwert aller Eingänge",
                          "das Produkt aller Eingänge",
                          "die größte Eingabe (Maximum)"],
             "richtig": 0, "erklaerung": "y = f(w₁x₁ + w₂x₂ + … + b)."},
            {"typ": "zahl", "frage": "Wie viele Parameter hat <span class='mono'>Dense(64)</span> nach einem Flatten eines 32 × 32-Graubildes?",
             "richtig": 65600, "toleranz": 0, "einheit": "Parameter",
             "erklaerung": "1024 Eingänge · 64 Neuronen + 64 Biases = 65 600."},
            {"typ": "einfach", "frage": "Welche Ausgabeschicht passt für unsere 4 Symbolklassen?",
             "optionen": ["<span class='mono'>Dense(4, activation=\"softmax\")</span>",
                          "<span class='mono'>Dense(1, activation=\"relu\")</span>",
                          "<span class='mono'>Dense(4, activation=\"relu\")</span>",
                          "<span class='mono'>Dense(255, activation=\"softmax\")</span>"],
             "richtig": 0, "erklaerung": "Ein Neuron pro Klasse; Softmax liefert Wahrscheinlichkeiten, die sich zu 1 addieren."},
            {"typ": "zahl", "frage": "Softmax liefert [0,70 · 0,10 · 0,15 · ?]. Welcher Wert fehlt?",
             "richtig": 0.05, "toleranz": 0.001, "einheit": "",
             "erklaerung": "Alle Wahrscheinlichkeiten ergeben zusammen 1: 1 − 0,70 − 0,10 − 0,15 = 0,05."},
            {"typ": "einfach", "frage": "Wie lautet der One-Hot-Vektor für Klasse <b>2</b> (Dreieck) bei 4 Klassen?",
             "optionen": ["[0, 0, 1, 0]", "[0, 1, 0, 0]", "[2]", "[0, 0, 2, 0]"], "richtig": 0,
             "erklaerung": "Gezählt wird ab 0: Position 0, 1, 2, 3."},
            {"typ": "einfach", "frage": "In der ersten Epoche zeigt ein untrainiertes 4-Klassen-Netz einen Verlust von etwa …",
             "optionen": ["1,39", "0,00", "0,25", "4,00"], "richtig": 0,
             "erklaerung": "Zufälliges Raten: p = 0,25 für jede Klasse, −ln(0,25) ≈ 1,39."},
            {"typ": "zahl", "frage": "1214 Trainingsbilder, <span class='mono'>batch_size=32</span>: Wie viele Gradientenschritte gibt es pro Epoche?",
             "richtig": 38, "toleranz": 0, "einheit": "Schritte",
             "erklaerung": "1214 / 32 = 37,9 → aufgerundet 38 (der letzte Stapel ist kleiner)."},
            {"typ": "zuordnung", "frage": "Ordnen Sie jedem Kurvenverlauf die Diagnose zu.",
             "paare": [["Trainingsverlust sinkt weiter, Validierungsverlust steigt wieder", "Überanpassung"],
                       ["beide Verluste bleiben hoch und flach", "Unteranpassung"],
                       ["Verlust springt stark hin und her", "Lernrate zu hoch"],
                       ["beide Kurven fallen am Ende noch deutlich", "zu wenige Epochen"]],
             "erklaerung": "Diese vier Muster sollten Sie in jedem Training erkennen können."},
            {"typ": "mehrfach", "frage": "Was hilft gegen Überanpassung? (mehrere richtig)",
             "optionen": ["Dropout", "Datenaugmentierung", "mehr Trainingsbilder", "EarlyStopping",
                          "mehr Neuronen pro Schicht", "Lernrate stark erhöhen"],
             "richtig": [0, 1, 2, 3],
             "erklaerung": "Mehr Neuronen erhöhen die Kapazität und damit eher das Auswendiglernen."},
            {"typ": "einfach", "frage": "Was macht <span class='mono'>EarlyStopping(monitor=\"val_loss\", patience=20, restore_best_weights=True)</span>?",
             "optionen": ["Es beendet das Training, wenn sich val_loss 20 Epochen lang nicht verbessert, und stellt die besten Gewichte wieder her.",
                          "Es stoppt nach genau 20 Epochen.",
                          "Es halbiert alle 20 Epochen die Lernrate.",
                          "Es speichert alle 20 Epochen eine Datei."],
             "richtig": 0, "erklaerung": "So erhält man automatisch das Modell aus der besten Epoche."},
        ],
    },
    "tzk3_cnn_und_bewertung": {
        "titel": "Teilzielkontrolle 3 – Faltungsnetze und Modellbewertung",
        "ds": "nach Doppelstunde 7",
        "fragen": [
            {"typ": "einfach", "frage": "Warum eignet sich ein CNN für Bilder besser als ein FCNN?",
             "optionen": ["Es nutzt die Nachbarschaft der Pixel und wendet dieselben Filter überall an – weniger Parameter, robust gegen Verschiebung.",
                          "Es rechnet mit höherer Genauigkeit (float64).",
                          "Es braucht keine Trainingsdaten.",
                          "Es wandelt das Bild zuerst in einen Vektor um."],
             "richtig": 0, "erklaerung": "Gewichtsteilung (shared weights) und lokale Verbindungen sind die Kernidee."},
            {"typ": "zahl", "frage": "Bildausschnitt [[0,0,1],[0,0,1],[0,0,1]] wird mit dem Kernel [[−1,0,1],[−1,0,1],[−1,0,1]] gefaltet. Ergebnis?",
             "richtig": 3, "toleranz": 0, "einheit": "",
             "erklaerung": "Nur die rechte Spalte ist 1 und wird mit +1 multipliziert: 1 + 1 + 1 = 3 → senkrechte Kante erkannt."},
            {"typ": "zahl", "frage": "Wie viele Parameter hat <span class='mono'>Conv2D(32, 3)</span>, wenn die Eingabe 16 Kanäle hat?",
             "richtig": 4640, "toleranz": 0, "einheit": "Parameter",
             "erklaerung": "32 Filter · (3 · 3 · 16 Gewichte + 1 Bias) = 32 · 145 = 4640."},
            {"typ": "zahl", "frage": "Ein 224 × 224-Bild durchläuft fünfmal <span class='mono'>MaxPooling2D()</span> (2 × 2). Wie breit ist die Merkmalskarte danach?",
             "richtig": 7, "toleranz": 0, "einheit": "Pixel",
             "erklaerung": "224 → 112 → 56 → 28 → 14 → 7."},
            {"typ": "einfach", "frage": "Was bewirkt <span class='mono'>padding=\"same\"</span>?",
             "optionen": ["Die Merkmalskarte bleibt so groß wie die Eingabe (Rand wird mit Nullen aufgefüllt).",
                          "Alle Filter erhalten dieselben Gewichte.",
                          "Das Bild wird auf die gleiche Größe wie der Kernel verkleinert.",
                          "Es wird keine Aktivierungsfunktion verwendet."],
             "richtig": 0, "erklaerung": "Ohne Padding („valid“) schrumpft die Karte bei 3 × 3 um je 2 Pixel."},
            {"typ": "zuordnung", "frage": "Ordnen Sie jeder Schicht ihre Aufgabe zu.",
             "paare": [["Rescaling(1/255)", "normiert die Pixelwerte auf 0 … 1"],
                       ["Conv2D", "findet lokale Muster wie Kanten und Ecken"],
                       ["MaxPooling2D", "verkleinert die Merkmalskarten"],
                       ["Flatten", "macht aus den Merkmalskarten einen Vektor"],
                       ["Dense(…, softmax)", "entscheidet über die Klasse"]],
             "erklaerung": "So ist das CNN in Notebook 12 aufgebaut."},
            {"typ": "einfach", "frage": "In der Konfusionsmatrix steht in Zeile <b>Kreis</b>, Spalte <b>Quadrat</b> die Zahl 7. Was heißt das?",
             "optionen": ["7 Kreise wurden fälschlich als Quadrat erkannt.",
                          "7 Quadrate wurden fälschlich als Kreis erkannt.",
                          "7 Kreise wurden richtig erkannt.",
                          "Das Modell ist zu 7 % unsicher."],
             "richtig": 0, "erklaerung": "Zeile = tatsächliche Klasse, Spalte = Vorhersage."},
            {"typ": "zahl", "frage": "Von 66 echten Dreiecken erkennt das Modell 60 richtig. Wie groß ist der <b>Recall</b> für Dreieck (in %)?",
             "richtig": 90.9, "toleranz": 0.6, "einheit": "%",
             "erklaerung": "Recall = richtig gefunden / alle echten = 60 / 66 ≈ 90,9 %."},
            {"typ": "einfach", "frage": "Ein durchgerutschtes fehlerhaftes Teil (Schlupf) kostet 250 €, ein unnötig aussortiertes 8 €. Wie stellen Sie die Prüfstation ein?",
             "optionen": ["Schwelle für „n. i. O.“ senken, damit der Recall für fehlerhafte Teile steigt",
                          "Schwelle erhöhen, damit weniger Teile aussortiert werden",
                          "nur auf die Accuracy achten",
                          "die Klasse „n. i. O.“ aus dem Training entfernen"],
             "richtig": 0, "erklaerung": "Lieber etwas mehr Pseudoausschuss als teurer Schlupf."},
            {"typ": "mehrfach", "frage": "Kreis und Quadrat werden oft verwechselt. Was sind sinnvolle Gegenmaßnahmen? (mehrere richtig)",
             "optionen": ["mehr und abwechslungsreichere Bilder dieser beiden Klassen aufnehmen",
                          "größere Eingabebilder verwenden, damit Ecken besser sichtbar sind",
                          "Augmentierung mit Drehung und Zoom nutzen",
                          "die Testbilder mit ins Training nehmen",
                          "die Klasse Quadrat löschen"],
             "richtig": [0, 1, 2],
             "erklaerung": "Testbilder im Training schönen nur die Zahl, verbessern aber nicht das Modell."},
        ],
    },
    "tzk4_transfer_und_einsatz": {
        "titel": "Teilzielkontrolle 4 – Transfer Learning und Einsatz in der Fertigung",
        "ds": "nach Doppelstunde 9",
        "fragen": [
            {"typ": "einfach", "frage": "Was ist die Grundidee von <b>Transfer Learning</b>?",
             "optionen": ["Ein auf vielen Bildern (ImageNet) vortrainiertes Netz liefert Merkmale; nur ein neuer, kleiner Klassifikationskopf wird trainiert.",
                          "Das Modell wird von einem PC auf den Raspberry Pi übertragen.",
                          "Zwei Modelle werden gleichzeitig trainiert.",
                          "Die Trainingsdaten werden aus dem Internet geladen."],
             "richtig": 0, "erklaerung": "Kanten, Ecken und Formen hat MobileNetV2 schon gelernt – das sparen wir uns."},
            {"typ": "einfach", "frage": "Was bewirkt <span class='mono'>base_model.trainable = False</span>?",
             "optionen": ["Die Gewichte des vortrainierten Netzes bleiben beim Training unverändert.",
                          "Das Basismodell wird gelöscht.",
                          "Das gesamte Modell kann nicht mehr trainiert werden.",
                          "Das Basismodell wird zufällig neu initialisiert."],
             "richtig": 0, "erklaerung": "Nur der neue Kopf lernt (Merkmalsextraktion)."},
            {"typ": "einfach", "frage": "Warum nutzt man bei der Feinabstimmung (Fine-Tuning) eine <b>sehr kleine</b> Lernrate?",
             "optionen": ["damit die gut vortrainierten Merkmale nicht zerstört werden",
                          "damit das Training schneller fertig ist",
                          "weil sonst der RAM nicht reicht",
                          "weil Softmax nur kleine Lernraten erlaubt"],
             "richtig": 0, "erklaerung": "Große Schritte würden die mühsam gelernten ImageNet-Gewichte schnell „verbiegen“."},
            {"typ": "einfach", "frage": "Warum genügen beim Transfer Learning etwa 10 Epochen, während das eigene CNN bis zu 300 Epochen bekommt?",
             "optionen": ["Nur der kleine Kopf (gut 5 000 Gewichte) lernt – die Merkmale des Basisnetzes sind schon gelernt.",
                          "Weil MobileNetV2 schneller rechnet.",
                          "Weil der Datensatz beim Transfer Learning größer ist.",
                          "Weil EarlyStopping bei MobileNetV2 nicht funktioniert."],
             "richtig": 0, "erklaerung": "Das eingefrorene Basisnetz bringt sein Wissen aus 1,4 Millionen Bildern mit."},
            {"typ": "einfach", "frage": "Was zeigt eine <b>Grad-CAM</b>-Darstellung?",
             "optionen": ["welche Bildbereiche die Entscheidung des Netzes am stärksten beeinflusst haben",
                          "die Lernkurve des Trainings",
                          "alle Filter der ersten Schicht",
                          "die Konfusionsmatrix als Wärmebild"],
             "richtig": 0, "erklaerung": "So prüft man, ob das Netz wirklich auf das Symbol schaut – und nicht z. B. auf den Tisch daneben."},
            {"typ": "zahl", "frage": "Bildaufnahme 30 ms + Inferenz 45 ms pro Teil. Wie viele Teile pro Minute schafft die Station höchstens?",
             "richtig": 800, "toleranz": 1, "einheit": "Teile/min",
             "erklaerung": "60 000 ms / 75 ms = 800 Teile pro Minute."},
            {"typ": "mehrfach", "frage": "Was muss bei der Inferenz <b>genau wie beim Training</b> sein? (mehrere richtig)",
             "optionen": ["Bildgröße (z. B. 224 × 224)", "Farbreihenfolge RGB (OpenCV liefert BGR!)",
                          "Normierung der Pixelwerte", "Reihenfolge der Klassen",
                          "Helligkeit des Bildschirms"],
             "richtig": [0, 1, 2, 3],
             "erklaerung": "Darum nutzen Training und Inferenz dieselbe Funktion bild_vorbereiten() und dieselbe modell_info.json."},
            {"typ": "einfach", "frage": "Das Modell ist sich nur zu 55 % sicher. Was soll die Prüfstation tun?",
             "optionen": ["das Teil zur manuellen Prüfung ausschleusen (Rückweisung)",
                          "trotzdem die wahrscheinlichste Klasse verwenden",
                          "das Bild löschen",
                          "das Modell sofort neu trainieren"],
             "richtig": 0, "erklaerung": "Unsichere Entscheidungen gehören zum Menschen – mit einer Konfidenzschwelle (z. B. 80 %)."},
            {"typ": "zuordnung", "frage": "Ordnen Sie jeder industriellen Aufgabe passende Klassen zu.",
             "paare": [["Teilesortierung", "Schraube · Mutter · Scheibe · leer"],
                       ["Oberflächenprüfung", "i. O. · Kratzer · Delle"],
                       ["Vollständigkeitsprüfung", "Bauteil bestückt · Bauteil fehlt"],
                       ["Lageerkennung", "richtig · verdreht · auf dem Kopf"]],
             "erklaerung": "Mit 00_Datensatz_aufnehmen.py --klassen … lässt sich jede dieser Aufgaben aufnehmen."},
            {"typ": "einfach", "frage": "Warum braucht eine Prüfstation fast immer eine Klasse <b>„leer / kein Teil“</b>?",
             "optionen": ["Sonst ordnet das Modell auch ein leeres Band zwingend einer Teileklasse zu.",
                          "Damit der Datensatz größer wird.",
                          "Weil Softmax mindestens 5 Klassen braucht.",
                          "Damit die Kamera scharf stellt."],
             "richtig": 0, "erklaerung": "Softmax verteilt immer 100 % auf die bekannten Klassen – auch wenn gar nichts zu sehen ist."},
            {"typ": "einfach", "frage": "Nach einem Lieferantenwechsel sind die Teile etwas glänzender. Die Fehlerquote der KI steigt. Wie nennt man das?",
             "optionen": ["Datendrift – die Einsatzdaten weichen von den Trainingsdaten ab",
                          "Überanpassung durch zu viele Epochen",
                          "Unteranpassung",
                          "ein Fehler im Raspberry Pi"],
             "richtig": 0, "erklaerung": "Abhilfe: regelmäßig neue Bilder aufnehmen, Leistung überwachen und nachtrainieren."},
        ],
    },
}

VORLAGE = r"""<!DOCTYPE html>
<html lang="de"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITEL__</title>
<style>
:root{--blau:#1f5f99;--blau-hell:#e8f0f8;--orange:#d9731a;--gruen:#2e8b57;--rot:#c0392b;--grau:#5f6b76;--linie:#d5dde5;--hg:#f7f9fb}
*{box-sizing:border-box}body{margin:0;font-family:system-ui,-apple-system,"Segoe UI",Roboto,Ubuntu,sans-serif;color:#1d2833;line-height:1.45;font-size:15px}
header{background:var(--blau);color:#fff;padding:12px 18px}
header .ls{font-size:12px;opacity:.85;text-transform:uppercase;letter-spacing:.03em}header h1{margin:2px 0 4px;font-size:20px}
#fortschritt{height:6px;background:rgba(255,255,255,.3);border-radius:3px;overflow:hidden}#fortschritt div{height:100%;background:#ffd18c;width:0;transition:width .3s}
main{max-width:900px;margin:0 auto;padding:12px 18px 30px}
.frage{border:1px solid var(--linie);border-radius:10px;padding:12px 14px;margin:12px 0;background:#fff}
.frage.ok{border-color:var(--gruen);background:#f2fbf5}.frage.teil{border-color:var(--orange);background:#fff8ef}.frage.falsch{border-color:var(--rot);background:#fdf3f2}
.kopfzeile{font-size:12px;color:var(--grau);margin-bottom:4px}.text{font-weight:600;margin-bottom:8px}
.opt{display:flex;gap:8px;align-items:flex-start;padding:6px 8px;border-radius:6px;cursor:pointer;margin:2px 0}.opt:hover{background:var(--hg)}
.opt.r{background:#dff2e6}.opt.f{background:#f8dcd8}
button{font:inherit;background:var(--blau);color:#fff;border:0;border-radius:6px;padding:6px 14px;cursor:pointer;margin:6px 4px 0 0}
button.zweit{background:#fff;color:var(--blau);border:1px solid var(--blau)}button:disabled{opacity:.45;cursor:default}
select,input{font:inherit;padding:4px 6px;border:1px solid var(--linie);border-radius:5px}
.paar{display:grid;grid-template-columns:1fr 1.3fr;gap:8px;align-items:center;margin:4px 0}
.paar select{width:100%}.paar.r select{background:#dff2e6}.paar.f select{background:#f8dcd8}
.schritt{display:flex;align-items:center;gap:6px;border:1px solid var(--linie);border-radius:6px;padding:5px 8px;margin:4px 0;background:#fff}
.schritt span{flex:1}.schritt button{margin:0;padding:2px 8px}.schritt.r{background:#dff2e6}.schritt.f{background:#f8dcd8}
.erkl{margin-top:8px;padding:8px 10px;border-radius:6px;background:var(--blau-hell);font-size:14px;display:none}
.mono{font-family:ui-monospace,"DejaVu Sans Mono",Consolas,monospace}
#ergebnis{border:2px solid var(--blau);border-radius:10px;padding:14px;margin-top:18px;background:var(--blau-hell)}
.gross{font-size:26px;font-weight:700}
</style></head><body>
<header><div class="ls">Lernsituation KI-Sichtprüfstation · __DS__</div><h1>__TITEL__</h1><div id="fortschritt"><div></div></div></header>
<main>
<p>Beantworten Sie jede Frage und klicken Sie auf <b>Prüfen</b>. Sie sehen sofort, ob Ihre Antwort stimmt, und erhalten eine kurze Erklärung.</p>
<div id="fragen"></div>
<div id="ergebnis">
 <div>Punkte: <span class="gross" id="punkte">0</span> von <span id="max"></span> <span id="urteil"></span></div>
 <button class="zweit" id="neu">Neu starten</button>
</div>
</main>
<script>
const DATEN=__DATEN__;
const $=id=>document.getElementById(id);
const mische=a=>{a=[...a];for(let i=a.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[a[i],a[j]]=[a[j],a[i]]}return a};
let stand=[];
function baue(){
  stand=DATEN.fragen.map(()=>({punkte:null}));$('max').textContent=DATEN.fragen.length;
  $('fragen').innerHTML='';
  DATEN.fragen.forEach((q,n)=>{
    const d=document.createElement('div');d.className='frage';d.id='q'+n;
    let h=`<div class="kopfzeile">Frage ${n+1} von ${DATEN.fragen.length}${q.typ==='mehrfach'?' · mehrere Antworten möglich':''}</div><div class="text">${q.frage}</div>`;
    if(q.typ==='einfach'||q.typ==='mehrfach'){
      q._ord=mische([...q.optionen.keys()]);
      h+=q._ord.map(i=>`<label class="opt" data-i="${i}"><input type="${q.typ==='einfach'?'radio':'checkbox'}" name="a${n}" value="${i}"><span>${q.optionen[i]}</span></label>`).join('');
    }else if(q.typ==='zahl'){
      h+=`<input type="text" inputmode="decimal" id="z${n}" style="width:140px"> ${q.einheit||''}<br>`;
    }else if(q.typ==='zuordnung'){
      const rechts=mische(q.paare.map(p=>p[1]));
      h+=q.paare.map((p,i)=>`<div class="paar" id="p${n}_${i}"><span>${p[0]}</span><select><option value="">– bitte wählen –</option>${rechts.map(r=>`<option>${r}</option>`).join('')}</select></div>`).join('');
    }else if(q.typ==='reihenfolge'){
      q._liste=mische([...q.schritte.keys()]);
      h+=`<div id="r${n}"></div>`;
    }
    h+=`<button id="b${n}">Prüfen</button><div class="erkl" id="e${n}"></div>`;
    d.innerHTML=h;$('fragen').appendChild(d);
    if(q.typ==='reihenfolge')zeigeListe(n);
    $('b'+n).onclick=()=>pruefe(n);
  });
  aktualisiere();
}
function zeigeListe(n,markiere){
  const q=DATEN.fragen[n];
  $('r'+n).innerHTML=q._liste.map((s,k)=>`<div class="schritt ${markiere?(s===k?'r':'f'):''}"><b>${k+1}.</b><span>${q.schritte[s]}</span>${markiere?'':`<button class="zweit" data-k="${k}" data-d="-1">▲</button><button class="zweit" data-k="${k}" data-d="1">▼</button>`}</div>`).join('');
  $('r'+n).querySelectorAll('button').forEach(b=>b.onclick=()=>{const k=+b.dataset.k,d=+b.dataset.d,j=k+d;if(j<0||j>=q._liste.length)return;[q._liste[k],q._liste[j]]=[q._liste[j],q._liste[k]];zeigeListe(n)});
}
function zahl(s){return parseFloat(String(s).replace(/\s/g,'').replace(/\./g,'').replace(',','.'))}
function pruefe(n){
  const q=DATEN.fragen[n],box=$('q'+n);let p=0;
  if(q.typ==='einfach'){
    const w=box.querySelector('input:checked');if(!w){alert('Bitte wählen Sie eine Antwort.');return}
    p=+w.value===q.richtig?1:0;
    box.querySelectorAll('.opt').forEach(o=>{const i=+o.dataset.i;if(i===q.richtig)o.classList.add('r');else if(o.querySelector('input').checked)o.classList.add('f')});
  }else if(q.typ==='mehrfach'){
    const gew=[...box.querySelectorAll('input:checked')].map(e=>+e.value);if(!gew.length){alert('Bitte wählen Sie mindestens eine Antwort.');return}
    const soll=new Set(q.richtig);p=(gew.length===soll.size&&gew.every(g=>soll.has(g)))?1:0;
    box.querySelectorAll('.opt').forEach(o=>{const i=+o.dataset.i,c=o.querySelector('input').checked;if(soll.has(i))o.classList.add('r');else if(c)o.classList.add('f')});
  }else if(q.typ==='zahl'){
    let v=$('z'+n).value.trim();if(!v){alert('Bitte geben Sie eine Zahl ein.');return}
    let x=parseFloat(v.replace(',','.'));if(/^\d{1,3}(\.\d{3})+$/.test(v))x=zahl(v);
    p=Math.abs(x-q.richtig)<=q.toleranz+1e-9?1:0;
  }else if(q.typ==='zuordnung'){
    let r=0;q.paare.forEach((pp,i)=>{const z=$(`p${n}_${i}`),ok=z.querySelector('select').value===pp[1];if(ok)r++;z.classList.add(ok?'r':'f')});
    p=r/q.paare.length;
  }else if(q.typ==='reihenfolge'){
    const r=q._liste.filter((s,k)=>s===k).length;p=r===q._liste.length?1:Math.round(r/q._liste.length*2)/4;zeigeListe(n,true);
  }
  stand[n].punkte=p;
  box.classList.add(p>=1?'ok':p>0?'teil':'falsch');
  box.querySelectorAll('input,select').forEach(e=>e.disabled=true);$('b'+n).disabled=true;
  const e=$('e'+n);e.style.display='block';
  let loesung='';
  if(q.typ==='zahl'&&p<1)loesung=`Richtig: <b>${String(q.richtig).replace('.',',')} ${q.einheit||''}</b>. `;
  if(q.typ==='zuordnung'&&p<1)loesung='Richtig: '+q.paare.map(pp=>`${pp[0]} → <b>${pp[1]}</b>`).join('; ')+'. ';
  if(q.typ==='reihenfolge'&&p<1)loesung='Richtig: '+q.schritte.map((s,i)=>`${i+1}. ${s}`).join(' → ')+'. ';
  e.innerHTML=(p>=1?'<b style="color:var(--gruen)">Richtig!</b> ':p>0?`<b style="color:var(--orange)">Teilweise richtig (${String(p).replace('.',',')} Punkte).</b> `:'<b style="color:var(--rot)">Leider falsch.</b> ')+loesung+(q.erklaerung||'');
  aktualisiere();
}
function aktualisiere(){
  const fertig=stand.filter(s=>s.punkte!==null).length,summe=stand.reduce((a,s)=>a+(s.punkte||0),0),max=DATEN.fragen.length;
  document.querySelector('#fortschritt div').style.width=(fertig/max*100)+'%';
  $('punkte').textContent=String(Math.round(summe*100)/100).replace('.',',');
  const quote=summe/max;
  $('urteil').innerHTML=fertig<max?`<span style="color:var(--grau)">(${fertig} von ${max} beantwortet)</span>`:quote>=0.7?' – <b style="color:var(--gruen)">Teilziel erreicht.</b>':' – <b style="color:var(--rot)">Teilziel noch nicht erreicht – wiederholen Sie die rot markierten Themen.</b>';
}
$('neu').onclick=()=>{if(confirm('Alle Antworten löschen und neu beginnen?')){baue();window.scrollTo(0,0)}};
baue();
</script></body></html>
"""


def main():
    for name, tzk in TEILZIELKONTROLLEN.items():
        daten = dict(tzk)
        html = (VORLAGE.replace("__TITEL__", tzk["titel"]).replace("__DS__", tzk["ds"])
                .replace("__DATEN__", json.dumps(daten, ensure_ascii=False)))
        (ZIEL / f"{name}.html").write_text(html, encoding="utf-8")
        print(f"erstellt: material/{name}.html ({len(tzk['fragen'])} Fragen)")


if __name__ == "__main__":
    main()
