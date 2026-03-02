# Mathematische Grundlagen – Neural ODE Optimal Control

Dieses Dokument erklärt alle Gleichungen des Projekts, ihre Herkunft und den Zusammenhang zwischen den einzelnen Modulen.

---

## 1. Das physikalische System (`dynamics.py`)

### Modell: Block mit viskoser Reibung

Wir betrachten einen Massepunkt auf einer Linie, der durch eine Kraft $u(t)$ gesteuert wird und einer viskosen (geschwindigkeitsproportionalen) Reibung unterliegt. Nach dem zweiten Newtonschen Gesetz ($F = ma$, mit $m=1$):

$$m\ddot{x} = u(t) - \mu \dot{x}$$

Mit Einheitsmaße $m=1$ und dem Zustandsvektor $z = (x, v)^\top$ mit $v = \dot{x}$:

$$\boxed{\frac{dz}{dt} = \begin{pmatrix} \dot{x} \\ \dot{v} \end{pmatrix} = \begin{pmatrix} v \\ u(t) - \mu v \end{pmatrix}}$$

Dies ist ein **lineares Kontrollsystem** der Form $\dot{z} = Az + Bu$, mit:

$$A = \begin{pmatrix} 0 & 1 \\ 0 & -\mu \end{pmatrix}, \quad B = \begin{pmatrix} 0 \\ 1 \end{pmatrix}$$

Der Parameter $\mu = 0.5$ ist der Reibungskoeffizient. $A$ beschreibt die ungesteuerte Dynamik (Eigenvektor-Analyse zeigt: ohne Kontrolle konvergiert die Geschwindigkeit exponentiell zu null), $B$ sagt, dass die Kraft direkt auf die Beschleunigung wirkt.

---

## 2. Das Optimierungsproblem (Kostenstruktur)

### Kostenfunktional

Das Ziel ist, den Block von $z_0 = (0, 1)^\top$ zum Zielzustand $z^* = (1, 1)^\top$ zu steuern und dabei möglichst wenig Energie zu verbrauchen. Formal minimieren wir:

$$\boxed{J[u] = w_T \|z(T) - z^*\|^2 + w_e \int_0^T u(t)^2 \, dt}$$

**Herkunft der Terme:**

| Term | Bedeutung | Gewicht |
|------|-----------|---------|
| $\|z(T) - z^*\|^2$ | Euklidischer quadratischer Abstand vom Zielzustand am Endzeitpunkt | $w_T = 100$ |
| $\int_0^T u(t)^2 \, dt$ | Kontrollenergie (L²-Norm der Steuergröße) | $w_e = 0.01$ |

Dies ist ein klassisches **LQR-ähnliches (Linear Quadratic Regulator) Soft-Terminal-Constraint**-Problem. Das große Gewicht $w_T = 100$ erzwingt fast sicher die Erreichung des Ziels; $w_e = 0.01$ bestraft übermäßigen Energieaufwand leicht.

### Diskretisierung der Energiekosten (Trapezregel, `trainer.py`)

Die Integral-Approximation erfolgt via Trapezregel über ein Gitter $\{t_0, t_1, \ldots, t_N\}$:

$$\int_0^T u(t)^2 \, dt \approx \sum_{k=0}^{N-1} \frac{u(t_k)^2 + u(t_{k+1})^2}{2} \cdot (t_{k+1} - t_k)$$

Im Code (`trainer.py:66`):
```python
energy_cost = torch.sum(0.5 * (u_squared[:-1] + u_squared[1:]) * dt)
```

Die Trapezregel hat Genauigkeit $O(h^2)$ und ist für glatte Steuertrajektorien gut geeignet.

---

## 3. Neuronaler Regler (`controller.py`)

### Architektur

Das neuronale Netz parametrisiert die Steuerfunktion $u(t; \theta)$ als offene Steuerschleife (**open-loop control**):

$$u(t; \theta) = \text{MLP}_\theta\!\left(\frac{t}{T}\right)$$

Die Zeit $t \in [0, T]$ wird auf $[0, 1]$ normiert, um numerische Stabilität zu gewährleisten und den Gradienten ausgeglichen zu halten. Die Architektur:

$$\tau \xrightarrow{\text{Linear}(1 \to 32)} \xrightarrow{\text{ELU}} \xrightarrow{(\text{Linear}+\text{ELU}) \times 3} \xrightarrow{\text{Linear}(32 \to 1)} u$$

**ELU-Aktivierungsfunktion:**

$$\text{ELU}(x) = \begin{cases} x & \text{falls } x > 0 \\ e^x - 1 & \text{sonst} \end{cases}$$

ELU wird gegenüber ReLU bevorzugt, weil sie glatt und differenzierbar ist (wichtig für den ODE-Löser) und negative Aktivierungen unterstützt.

**Xavier-Initialisierung** (mit Gain 0.5):

$$W_{ij} \sim \mathcal{U}\!\left(-\frac{\sqrt{6}}{(n_\text{in}+n_\text{out})^{1/2}}, \frac{\sqrt{6}}{(n_\text{in}+n_\text{out})^{1/2}}\right) \cdot 0.5$$

Der reduzierte Gain (0.5 statt 1.0) sorgt für ein ruhiges Startverhalten – der Regler beginnt nahe null, damit das System nicht sofort instabil wird.

---

## 4. Neural ODE: Backpropagation durch den ODE-Löser (`trainer.py`)

### Die Kernidee

Statt analytische Gradienten herzuleiten, wird der ODE-Löser als **differenzierbarer Layer** behandelt:

$$z(T) = z_0 + \int_0^T f(t, z(t); \theta) \, dt$$

Der Gradient $\frac{\partial J}{\partial \theta}$ wird durch Backpropagation durch alle internen Schritte des Solvers berechnet (**direct differentiation**, nicht Adjoint-Methode). Dies erfordert, dass alle Operationen in PyTorch-Tensoren mit aktiviertem Gradientenfluss durchgeführt werden.

### Gradientenfluss

```
theta → u(t; theta) → dv/dt = u - mu*v → z(T) → L → dL/dtheta
```

Der ODE-Solver (DOPRI5, ein adaptiver Runge-Kutta-Solver 4./5. Ordnung) löst:

$$\dot{z} = f(t, z, \theta), \quad z(0) = z_0$$

Der Gradient $\nabla_\theta J$ wird durch automatische Differentiation durch alle Solver-Zwischenschritte propagiert. Dies ist möglich, weil `torchdiffeq` jeden Schritt als Rechenoperationen auf PyTorch-Tensoren ausführt.

### Adam-Optimierer

Update-Regel für Parameter $\theta$:

$$m_{k+1} = \beta_1 m_k + (1-\beta_1) g_k$$
$$v_{k+1} = \beta_2 v_k + (1-\beta_2) g_k^2$$
$$\theta_{k+1} = \theta_k - \frac{\alpha}{\sqrt{\hat{v}_{k+1}} + \epsilon} \hat{m}_{k+1}$$

mit Standardwerten $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$, Lernrate $\alpha=10^{-3}$.

### Lernraten-Scheduler

`ReduceLROnPlateau` halbiert die Lernrate ($\times 0.5$), wenn der Loss nach 200 Epochen nicht gesunken ist (Mindest-LR: $10^{-6}$).

### Gradientenclipping

$$g \leftarrow g \cdot \min\!\left(1, \frac{10}{\|g\|}\right)$$

Verhindert explodierende Gradienten, die bei Backprop durch den ODE-Solver auftreten können.

### Early Stopping

Das Training stoppt, wenn der **euklidische Endfehler** kleiner als $\delta = 10^{-3}$ wird:

$$\|z(T) - z^*\| = \sqrt{(x(T)-x^*)^2 + (v(T)-v^*)^2} < 10^{-3}$$

**Wichtig:** Der Stopp erfolgt *vor* dem Gradientenschritt, sodass der gespeicherte Regler exakt dem Zustand entspricht, der das Kriterium erfüllt hat.

---

## 5. Analytische Referenzlösung via Pontryagin (`analytical.py`)

### Pontryaginsches Minimumprinzip

Für das lineare quadratische Problem gibt das **Pontryaginsche Minimumprinzip** die Existenz einer Ko-Zustandsvariablen $p(t) \in \mathbb{R}^2$ (auch: **adjunkte Variable** oder **Lagrange-Multiplikator der ODE**), sodass:

**Hamilton-Funktion:**

$$H(z, u, p) = w_e u^2 + p^\top (Az + Bu)$$

Die optimale Steuergröße minimiert $H$ bezüglich $u$:

$$\frac{\partial H}{\partial u} = 2 w_e u + B^\top p = 0 \implies \boxed{u^*(t) = -\frac{1}{2 w_e} B^\top p(t)}$$

Da $B = (0,1)^\top$, gilt $B^\top p = p_1$ (zweite Komponente des Ko-Zustands), also:

$$u^*(t) = -\frac{p_1(t)}{2 w_e}$$

**Ko-Zustandsgleichung (rückwärts in der Zeit):**

$$\dot{p} = -\frac{\partial H}{\partial z} = -A^\top p$$

**Transversalitätsbedingung (Randwert am Endzeit):**

$$p(T) = \frac{\partial}{\partial z(T)}\left[w_T \|z(T) - z^*\|^2\right] = 2 w_T (z(T) - z^*)$$

Da $z(T)$ von $u$ abhängt und somit von $p$, ist dies ein **Randwertproblem** – keine einfache Anfangswertaufgabe.

---

### Lösung via Matrix-Riccati-Gleichung

Für **lineare** Systeme mit **quadratischer** Kostenfunktion lässt sich der Ko-Zustand als lineare Funktion des Zustands schreiben:

$$p(t) = 2 P(t) z(t) - 2 r(t)$$

Einsetzen in die Ko-Zustandsgleichung und Koeffizientenvergleich ergibt die **Matrix-Riccati-ODE** für $P(t)$ und die **affine Ko-Zustands-ODE** für $r(t)$:

$$\boxed{-\dot{P} = A^\top P + P A - \frac{1}{w_e} P B B^\top P, \quad P(T) = w_T I}$$

$$\boxed{-\dot{r} = \left(A^\top - \frac{1}{w_e} P B B^\top\right) r, \quad r(T) = w_T z^*}$$

Die optimale Steuerung lautet dann:

$$\boxed{u^*(t) = -\frac{1}{w_e} B^\top P(t) z(t) + \frac{1}{w_e} B^\top r(t)}$$

### Ausgeschriebene Skalare für $B=(0,1)^\top$

Da $B^\top P = \begin{pmatrix} P_{01} & P_{11} \end{pmatrix}$ (zweite Zeile von $P$, da $B$ nur die zweite Komponente selektiert):

$$u^*(t) = \frac{1}{w_e}\bigl(-P_{01}(t)\, x(t) - P_{11}(t)\, v(t) + r_1(t)\bigr)$$

Die Riccati-ODE in Komponenten (mit der Symmetrie $P = P^\top$, also $P_{10} = P_{01}$):

| Gleichung | Herkunft |
|-----------|----------|
| $\dot{P}_{00} = \frac{1}{w_e} P_{01}^2$ | Zeile 0, Spalte 0 der Riccati-ODE |
| $\dot{P}_{01} = -P_{00} + \mu P_{01} + \frac{1}{w_e} P_{01} P_{11}$ | Zeile 0, Spalte 1 |
| $\dot{P}_{11} = -2 P_{01} + 2\mu P_{11} + \frac{1}{w_e} P_{11}^2$ | Zeile 1, Spalte 1 |
| $\dot{r}_0 = \frac{1}{w_e} P_{01} r_1$ | Ko-Zustand, Komponente 0 |
| $\dot{r}_1 = -r_0 + \left(\mu + \frac{1}{w_e} P_{11}\right) r_1$ | Ko-Zustand, Komponente 1 |

*(Vorzeichen: rückwärts in $t$, im Code via Zeitumkehr $\tau = T - t$ als Vorwärts-Integration gelöst.)*

### Zeitumkehr-Trick (`analytical.py`)

Die Riccati-ODE läuft rückwärts ($t = T \to 0$). Im Code wird $\tau = T - t$ eingeführt, sodass $\tau$ vorwärts läuft ($\tau = 0 \to T$). Dann gelten die obigen Gleichungen mit positivem Vorzeichen (die Zeitableitung wechselt Vorzeichen durch Kettenregel $d/d\tau = -d/dt$).

---

## 6. Gesamtüberblick: Informationsfluss

```
              Physikalisches System
              ┌─────────────────────────────┐
              │  ẋ = v                       │
              │  v̇ = u(t) - μv              │  ← dynamics.py
              │  z = (x, v)ᵀ                │
              └────────────┬────────────────┘
                           │  ODE lösen
                           ▼
┌──────────────────────────────────────────────┐
│           Lernbasiert (Neural ODE)           │
│                                              │
│  u(t;θ) = MLP_θ(t/T)    ← controller.py     │
│                                              │
│  J = w_T‖z(T)-z*‖² + w_e∫u²dt  ← trainer.py│
│                                              │
│  ∂J/∂θ via Backprop durch Solver             │
│  Update: θ ← θ - α∇_θJ  (Adam)              │
└──────────────────────────────────────────────┘
                           │
                           │  Vergleich
                           ▼
┌──────────────────────────────────────────────┐
│        Analytisch optimal (LQR)              │
│                                              │
│  Riccati-ODE rückwärts → P(t), r(t)         │
│                                              │  ← analytical.py
│  u*(t) = (1/w_e)(-P₀₁x - P₁₁v + r₁)        │
│                                              │
│  Untere Schranke für J (global optimal)      │
└──────────────────────────────────────────────┘
```

---

## 7. Warum ist die LQR-Lösung optimal?

Die LQR-Lösung löst das **Randwertproblem der Optimalitätsbedingungen** exakt. Für **lineare** Systeme und **konvexe** (quadratische) Kostenfunktionale ist das Pontryaginsche Minimumprinzip sowohl notwendig als auch **hinreichend** für globale Optimalität. Die neuronale Lösung approximiert dieses Optimum, bleibt aber im Allgemeinen suboptimal – der Unterschied hängt von der Netzarchitektur, dem Training und der Kostenlandschaft ab.

---

## 8. Parameterübersicht

| Symbol | Wert | Bedeutung |
|--------|------|-----------|
| $z_0$ | $(0, 1)$ | Anfangszustand $(x_0, v_0)$ |
| $z^*$ | $(1, 1)$ | Zielzustand $(x^*, v^*)$ |
| $T$ | $2.0$ | Zeithorizont |
| $\mu$ | $0.5$ | Reibungskoeffizient |
| $w_T$ | $100$ | Gewicht Terminalkosten |
| $w_e$ | $0.01$ | Gewicht Energiekosten |
| $\delta$ | $10^{-3}$ | Early-Stopping-Schwelle für $\|z(T)-z^*\|$ |
| $\alpha$ | $10^{-3}$ | Lernrate (Adam) |
| $N$ | $200$ | Anzahl Auswertepunkte im ODE-Solver |
