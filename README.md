# Modeling-3D

Plain‑English, dimensioned spec for regenerating the current mansion model.

**Units:** meters (m)  
**Axes:** X = left→right (façade length), Y = front→back, Z = up.  
**Origin:** (0,0,0) is the front‑left corner of the main body at ground level.

---

## Overall footprint & base
- **Main body footprint:** 24.0 (X) × 8.0 (Y), from **X=0..24**, **Y=0..8**.
- **Base/floor slab:** extend **≥1.0 m** around the full building footprint:  
  **X = -1.0..25.0**, **Y = -1.0..9.0**, **Z = 0..0.30** (floor thickness 0.30).
- **Wall thickness:** 0.20.

---

## Main body (2 floors)
- **Ground floor shell:**  
  Outer box **X=0..24**, **Y=0..8**, **Z=0.30..3.50** (height 3.2).  
  Hollow walls (0.20 thick).
- **Second floor shell (setback):**  
  Inset 0.35 on all sides → **X=0.35..23.65**, **Y=0.35..7.65**, **Z=3.50..6.50** (height 3.0).

---

## Front & back openings (main body)
- **Main front door (centered):** width 2.0, height 2.8.  
  Center at **X=12.0**, **Y=0**, **Z=0.45..3.10**.
- **Rear double door (centered):** width 1.9, height 2.5.  
  Center at **X=12.0**, **Y=8.0**, **Z=0.45..2.95**.
- **Ground‑floor windows (front + back):**  
  Centers along X at **4.5, 6.0, 18.0, 19.5**.  
  Each window width 1.2, height 1.1, centered on those Xs.  
  Vertical: **Z=1.4..2.5**.
- **Second‑floor windows (front + back):**  
  Centers along X at **5.2, 7.0, 17.0, 18.8**.  
  Width 0.9, height 1.0.  
  Vertical: **Z=4.4..5.4**.
- **Central second‑floor “arched” opening (front + back):**  
  Approx rectangle width 2.0, height 1.2.  
  Center at **X=12.0**, vertical **Z=4.4..5.6**.

---

## Side wings (ground floor only)
- **Left wing:**  
  **X = -1.5..3.8**, **Y = 0.6..7.4**, **Z=0.30..3.50** (hollow walls).
- **Right wing removed** (replaced by annex).

---

## Right annex (outside, rectangular)
- **Size:** 2.5 (width in X) × 5.0 (depth in Y), height 3.2.  
- **Placement:** outside to the right of main body:  
  **X=24.0..26.5**, **Y=0.6..5.6**, **Z=0.30..3.50**.
- **Annex windows (open cut‑throughs):**  
  Window size **0.7 (W) × 1.05 (H)**, vertical **Z=1.25..2.30**.  
  - **Front (Y=0.6):** one window on front wall, center X=24.6.  
  - **Back (Y=5.6):** one window on back wall, same X.  
  - **Right side (X=26.5):** two windows evenly spaced along Y:  
    centers at Y≈2.27 and Y≈3.93.
- **Annex roof:** simple gable over annex:  
  pitch 30°, overhang 0.12, thickness 0.18.  
  Ridge runs **front‑to‑back (Y)**.

---

## Entrance portico (center front)
- **Central projection:**  
  **X=8..16**, **Y=-2.0..0.5**, **Z=0.30..4.50** with a clear opening in the middle.
- **Side columns at opening (octagonal):**  
  2 columns, radius 0.34, height 4.2.  
  Placed left/right of door opening at X≈(12±2.15), Y≈-0.75.
- **Front columns (octagonal):**  
  2 columns, radius ≈0.23, height 3.5 (from Z=0.35..3.85).  
  Positioned near front edge at X≈(8.8 and 15.2), Y≈-2.1.
- **Rear columns (rectangular):**  
  2 columns, 0.42×0.42, same height, Y≈-1.15.

---

## Balcony & rails
- **Balcony slab:**  
  **X=8.55..15.45**, **Y=-2.18..-0.58**, **Z≈3.95..4.15**.
- **Balcony rail:**  
  Front + sides + back, with rear gap at balcony door (centered).  
  Rail height ≈0.52, balusters spaced evenly.

---

## Front stairs
- **Terrace:** **X=8.75..15.25**, **Y=-2.10..-0.78**, **Z=0.30..0.78**.
- **Steps:** 8 steps, each 0.19 deep, rise evenly from Z=0.30 to Z=0.78.  
  Stair width 5.8, centered at X=12.0.

---

## Roofs
- **Main roof:** gable, pitch 35°, overhang 0.35, thickness 0.22.  
  Base at **Z=6.20** (main roof slightly lowered).
- **Left wing roof:** gable, pitch 30°, overhang 0.25, thickness 0.20.
- **Roof‑wall sealing band** around main roof to eliminate gaps.

---

## Towers
- **Two octagonal towers:**  
  Radius 1.5, height **(3.2+3.0+1.6)=7.8**.  
  Positioned near front corners: X≈1.2 and X≈22.8, Y≈(1.0 − 40% diameter).  
- **Tower windows:** 2 per face (lower & upper) cut through.
- **Tower roofs:** octagonal cones, height 2.6, radius 1.5+0.25.
