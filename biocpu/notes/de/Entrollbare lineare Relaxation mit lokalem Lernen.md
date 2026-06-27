# Entrollbare lineare Relaxation mit lokalem Lernen

> Technische Spezifikation. Status: Konzept und Kernmathematik definiert;
> noch keine Implementierung.

---

## 0. Zusammenfassung

Eine Lernmethode, bei der eine lineare Relaxationsdynamik mit multiplikativer
Dämpfung (Decay) zwei äquivalente Formen besitzt: eine rekurrente Form und eine in
einen einzigen parallelen Ganzzahl-Durchlauf entrollte Form. Das Lernen ist lokal
und wird aus der Differenz zweier Gleichgewichte berechnet, ohne Backpropagation
und ohne Forward-Pass im üblichen Sinne.

Allgemeines Prinzip: Der Zielkonflikt zwischen Rekurrenz, Parallelität und Leistung
wird nicht durch Heuristiken aufgelöst, sondern durch Beschränkung der Operation auf
eine Unterklasse (Linearität mit Decay), die zwei äquivalente Rechenformen besitzt.

---

## 1. Zielvorgaben

Ziel ist lokales Lernen, das auf einer CPU effizient ist: Ganzzahlarithmetik
(int8/int16), Vektorbefehle (z. B. AVX-512) und Skalierung über Mehrkernsysteme und
Cluster.

Bewusst vermiedene Eigenschaften, mit Begründung:

| Ansatz | Grund für den Ausschluss |
|---|---|
| Backpropagation | biologisch unplausibel (Weight-Transport-Problem) |
| Spike-Timing-Dependent Plasticity (STDP) | ineffizient auf der CPU: zeitliche Spuren pro Synapse, speichergebunden; effizient nur auf neuromorpher Hardware |
| Spiking-Neuronen (Integrate-and-Fire) | benötigen T Zeitschritte pro Beispiel; die Zeitachse parallelisiert nicht |
| Forward-Forward | enthält einen Forward-Pass |
| Equilibrium Propagation (vollständig) | rechnerisch teuer (nichtlineare Relaxation) und instabil |

Allgemeine Beobachtung: Ansätze mit zeitlicher Dynamik sind auf der CPU teuer;
Einzeldurchlauf-Ansätze enthalten einen Forward-Pass. Die Lösung ist die Äquivalenz
zweier Formen einer Operation.

---

## 2. Entwurfsentscheidungen

**Harte Invarianten:**
1. Berechnung und Gewichte sind ganzzahlig (Gewichte int8, Zustand int16);
   Multiplikation wird durch Addition/Shifts ersetzt (ternäre Gewichte {−1, 0, +1}
   oder Zweierpotenzen; Ganzzahl-MAC über Vektorbefehle).
2. Lernen ist lokal: eine Synapsenaktualisierung hängt nur von ihren beiden Enden ab.
3. Keine Zeitachse — eine feste, kleine Zahl von Ganzzahlschritten pro Beispiel.
4. Sparsität ist aktivierungsbasiert (k-WTA zwischen Schichten) — Quelle von
   Geschwindigkeit, Nichtlinearität und rascher Reihenabbruch.
5. Kleine Aktualisierungen werden in Gleitkomma akkumuliert (punktuell); die
   Gewichte bleiben ganzzahlig. Hohe Präzision wird nur dort aufgewendet, wo
   Rundung sonst Information zerstören würde.
6. Skalierbarkeit: Batch-Parallelität ohne zeitliche Abhängigkeiten.

**Bewusster Kompromiss:** Zielgenauigkeit ~95–96 % (Machbarkeitsnachweis), nicht
99 %. Dies erlaubt einen linearen Kern mit geschlossener Gleichgewichtsform statt
teurer nichtlinearer Relaxation.

**Die Nichtlinearität liegt zwischen den Schichten (k-WTA), nicht innerhalb der
Relaxation.** Innerhalb einer Schicht ist die Dynamik linear (entrollbar, stabil);
zwischen den Schichten liefert k-WTA Nichtlinearität und Sparsität. Dadurch bleibt
die Relaxation linear, und eine geschlossene Form ist gewährleistet.

---

## 3. Kernmathematik (einzelne Schicht)

Notation: Zustand `x`, Eingang aus der vorherigen Schicht `u`, Eingangsgewichte `W`,
latente Kopplung zwischen den Neuronen der Schicht `M`, Dämpfung `γ ∈ (0, 1)`.

### 3.1 Rekurrente Form
```
x[t+1] = γ · M · x[t] + W · u
```
Der Zustand entwickelt sich zum Gleichgewicht; γ ist die Dämpfung (Vergessen);
`W·u` ist die Eingangsstörung; `M` ist die latente Wechselwirkung. Bei `M = 0` ist
dies eine reine Kette.

### 3.2 Gleichgewicht
```
x* = γ M x* + W u   ⟹   x* = (I − γM)⁻¹ W u
```

### 3.3 Entrollte Form (Neumann-Reihe)
```
(I − γM)⁻¹ = Σ_{j≥0} (γM)^j = I + γM + γ²M² + …
```
Nach k Gliedern abgebrochen:
```
x* ≈ Σ_{j=0}^{k} γ^j M^j (W u)
```
Berechnung nach dem Horner-Schema (ohne Speicherung der Potenzen von M):
```
y ← W u
acc ← y
for j in 1..k:
    y ← γ · M · y          # ein Ganzzahl-Matvec (oder Additionen/Shifts)
    acc ← acc + y
x* ← acc
```
k Glieder = k wiederholte Ganzzahl-Matvecs, parallel über den Batch. Dämpfung
γ < 1 und k-WTA-Sparsität brechen die Reihe schnell ab: typischerweise k ≈ 3–5.

### 3.4 Äquivalenz der Formen
Die rekurrente Form (3.1) und die abgebrochene Reihe (3.3) konvergieren zum selben
`x*`. Dies ist eine Operation in zwei Formen: konzeptionell setzt sich das Netz in
der Zeit; rechnerisch wird das Gleichgewicht in k Ganzzahlschritten erreicht.

---

## 4. Wahl von M und Garantien

Die Universalitätsanforderung teilt sich in zwei unabhängige Eigenschaften:

- **Universelle Approximation** (Abdeckung einer beliebigen Aufgabe/Modalität) —
  durch Tiefe und die k-WTA-Nichtlinearität zwischen den Schichten; gilt für jedes M.
- **Konvergenz** (garantierte Divergenzfreiheit) — durch Symmetrie von M.

| M | Gleichgewicht | Garantie |
|---|---|---|
| M = 0 | x* = Wu | trivial, immer |
| M = Mᵀ (symmetrisch) | (I−γM)⁻¹Wu | bewiesenes eindeutiges Gleichgewicht und Konvergenz |
| M beliebig | existiert evtl. nicht | keine (Oszillationen möglich) |

**Symmetrischer Fall.** Für `M = Mᵀ` existiert eine Energiefunktion:
```
E(x) = ½ xᵀ (I − γM) x − xᵀ W u
```
Falls `γ‖M‖₂ < 1`, ist die Matrix `(I − γM)` positiv definit, E ist konvex, und es
gibt ein eindeutiges globales Minimum, zu dem die Relaxation stets konvergiert.

**Entscheidung:** Der Kern verwendet ein symmetrisches M mit der Bedingung
`γ‖M‖₂ < 1`. Der Fall M = 0 ist sein Spezialfall (möglicher Ausgangspunkt).
Universalität ergibt sich aus Tiefe und k-WTA; Stabilität aus der Symmetrie.

**Modalitätsspezifische Struktur liegt vor dem Kern, nicht in M.** Eingaben jeder
Modalität treten als Vektor `u` in die Schicht ein. Die Struktur von `W·u` bestimmt
die Modalität: Faltungsstruktur (lokale rezeptive Felder) für Bilder; positionelle
Kodierung für Sequenzen. Ein universeller Kern, verschiedene Eingangs-Frontends.

---

## 5. Lokales Lernen

Ein Zwei-Phasen-Prinzip (im Geist der Equilibrium Propagation), verbilligt dadurch,
dass beide Gleichgewichte unter linearer Relaxation geschlossene Formen besitzen.

**Zwei Gleichgewichte:**
- frei: `x*_free` — Gleichung (3.3) nur mit Eingang;
- geklemmt: `x*_clamped` — Ausgang mit Stärke β zum Ziel gezogen.

**Lokale Aktualisierungsregel:**
```
ΔW ∝ (x*_clamped ⊗ u) − (x*_free ⊗ u)
```
Jede Synapse verwendet nur ihre beiden Enden (postsynaptischer Zustand und
präsynaptischer Eingang). Es gibt keinen Rückwärtsdurchlauf durch das Netz, keine
Kettenregel und keine explizite Verlustfunktion.

ΔW wird in Gleitkomma akkumuliert; Gewichte werden ganzzahlig quantisiert, sobald
ein Quantisierungsschritt überschritten wird.

Im Vergleich zur vollständigen Equilibrium Propagation: Beide Phasen werden in
geschlossener Form berechnet, sodass die zweite Phase keine teure nichtlineare
Relaxation benötigt — wodurch sowohl Kosten als auch Instabilität entfallen.

---

## 6. k-WTA (k-Winners-Take-All)

Ein Mechanismus der Aktivierungssparsität. Für eine Schicht aus N Neuronen bleiben
nur die k Neuronen mit den größten Aktivierungen aktiv; die übrigen werden auf null
gesetzt.

```
Eingang:  N Aktivierungen
Ausgang:  k größte behalten, (N − k) auf null
```

Das biologische Analogon ist die laterale Hemmung: die aktivsten Neuronen
unterdrücken über hemmende Verbindungen ihre Nachbarn, sodass nur ein kleiner
Anteil aktiv bleibt.

In dieser Methode erfüllt k-WTA drei Rollen:
1. **Sparsität** — nur k von N aktiv; inaktive Einheiten werden in der Berechnung
   übersprungen.
2. **Nichtlinearität zwischen den Schichten** — die Top-k-Auswahl ist nichtlinear
   und liegt zwischen den Schichten, sodass die schichtinterne Dynamik linear (und
   entrollbar) bleibt.
3. **Beschleunigung des Reihenabbruchs** — ein dünnbesetzter Vektor nach jedem
   Reihenglied (3.3) beschleunigt das Abklingen der folgenden Glieder, sodass
   k ≈ 3–5 genügt.

Die Nichtdifferenzierbarkeit der Top-k-Auswahl ist kein Problem, da das Lernen
lokal ist (Abschnitt 5) und keinen Gradienten verwendet.

Der Parameter k (typischerweise 2–10 % der Neuronen einer Schicht) ist ein
Hyperparameter: zu klein verringert die Kapazität, zu groß verringert die Sparsität.

---

## 7. Offene Fragen

1. Tatsächliche Zahl der Glieder k für die Zielgenauigkeit (empirisch).
2. k-WTA: festes k oder adaptive Schwelle.
3. Quantisierung des Aktualisierungs-Akkumulators auf Ganzzahl: Skala pro Schicht oder pro Kanal.
4. Erhaltung der Symmetrie von M: explizit (M ← ½(M+Mᵀ)) oder Parametrisierung M = LLᵀ.
5. Klemmstärke β: fest oder geplant.
6. Eingangs-Frontends: Ganzzahlfaltung für Bilder; positionelle Kodierung für Sequenzen.
7. Möglicher Vorteil der rekurrenten Form bei der Inferenz (konstanter Speicher).

---

## 8. Nächster Schritt

Ein minimaler Machbarkeits-Prototyp: Kern (3.3) mit M = 0, dann symmetrisches M,
k-WTA zwischen den Schichten, die Zwei-Phasen-Regel (Abschnitt 5). Erstes Ziel —
das Überschreiten einer Basisgenauigkeit und die Bestimmung des tatsächlichen k.
