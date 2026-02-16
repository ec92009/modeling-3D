#!/usr/bin/env python3
import math
import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix


def add_box(parts, x0, x1, y0, y1, z0, z1):
    if x1 <= x0 or y1 <= y0 or z1 <= z0:
        return
    b = trimesh.creation.box(extents=[x1 - x0, y1 - y0, z1 - z0])
    b.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
    parts.append(b)


def add_tri_prism_x(parts, A, B, C, x0, x1):
    ay, az = A
    by, bz = B
    cy, cz = C
    v = np.array([
        [x0, ay, az], [x0, by, bz], [x0, cy, cz],
        [x1, ay, az], [x1, by, bz], [x1, cy, cz],
    ], float)
    f = np.array([
        [0, 2, 1], [3, 4, 5],
        [0, 1, 4], [0, 4, 3],
        [1, 2, 5], [1, 5, 4],
        [2, 0, 3], [2, 3, 5],
    ], int)
    parts.append(trimesh.Trimesh(vertices=v, faces=f, process=False))


def add_roof(parts, xo, yo, L, W, wall_top_z, pitch_deg, overhang, roof_t, gable=True):
    xr0, xr1 = xo - overhang, xo + L + overhang
    yr0, yr1 = yo - overhang, yo + W + overhang
    Wr = yr1 - yr0
    run = Wr / 2.0
    rise = run * math.tan(math.radians(pitch_deg))
    slant = math.sqrt(run * run + rise * rise)

    panel = trimesh.creation.box(extents=[xr1 - xr0, slant, roof_t])
    p1 = panel.copy()
    p1.apply_transform(rotation_matrix(math.radians(pitch_deg), [1, 0, 0]))
    p1.apply_translation([(xr0 + xr1) / 2, yr0 + Wr / 4, wall_top_z + rise / 2])
    parts.append(p1)

    p2 = panel.copy()
    p2.apply_transform(rotation_matrix(math.radians(180 - pitch_deg), [1, 0, 0]))
    p2.apply_translation([(xr0 + xr1) / 2, yr0 + 3 * Wr / 4, wall_top_z + rise / 2])
    parts.append(p2)

    if gable:
        # Clean triangular gable panels (replaces stair-step stripe artifacts)
        rz = wall_top_z + (W / 2.0) * math.tan(math.radians(pitch_deg))
        wt = min(0.35, L * 0.05)
        A = (yo, wall_top_z)
        B = (yo + W, wall_top_z)
        C = (yo + W / 2.0, rz)

        # left gable triangular panel
        add_tri_prism_x(parts, A, B, C, xo, xo + wt)
        # right gable triangular panel
        add_tri_prism_x(parts, A, B, C, xo + L - wt, xo + L)


def add_shell_with_front_openings(parts, xo, yo, L, W, z0, z1, wall_t, openings_front, openings_back=None):
    """Hollow box shell with segmented front/back walls for openings."""
    if openings_back is None:
        openings_back = []

    # side walls
    add_box(parts, xo, xo + wall_t, yo + wall_t, yo + W - wall_t, z0, z1)
    add_box(parts, xo + L - wall_t, xo + L, yo + wall_t, yo + W - wall_t, z0, z1)

    # front wall segmented around openings
    openings = sorted(openings_front, key=lambda o: o[0])
    z_levels = sorted(set([z0, z1] + [o[2] for o in openings] + [o[3] for o in openings]))

    for za, zb in zip(z_levels[:-1], z_levels[1:]):
        if zb <= za:
            continue
        cur = xo
        for xa, xb, oa, ob in openings:
            if zb <= oa or za >= ob:
                continue
            if xa > cur:
                add_box(parts, cur, xa, yo, yo + wall_t, za, zb)
            cur = max(cur, xb)
        if cur < xo + L:
            add_box(parts, cur, xo + L, yo, yo + wall_t, za, zb)

    # back wall segmented around openings (at y=yo+W)
    openings_b = sorted(openings_back, key=lambda o: o[0])
    z_levels_b = sorted(set([z0, z1] + [o[2] for o in openings_b] + [o[3] for o in openings_b]))

    for za, zb in zip(z_levels_b[:-1], z_levels_b[1:]):
        if zb <= za:
            continue
        cur = xo
        for xa, xb, oa, ob in openings_b:
            if zb <= oa or za >= ob:
                continue
            if xa > cur:
                add_box(parts, cur, xa, yo + W - wall_t, yo + W, za, zb)
            cur = max(cur, xb)
        if cur < xo + L:
            add_box(parts, cur, xo + L, yo + W - wall_t, yo + W, za, zb)


def build(target_width_mm=180.0):
    # Main door is 2m wide (user requirement)
    main_door_w_m = 2.0

    # Mansion reference dimensions (meters)
    L, W = 24.0, 8.0
    h1, h2 = 3.2, 3.0
    floor_t, wall_t = 0.30, 0.20

    center_proj_d = 2.0
    side_wing_d = 1.5

    s = target_width_mm / L
    Ls, Ws = L * s, W * s
    h1s, h2s = h1 * s, h2 * s
    floor_ts, wall_ts = floor_t * s, wall_t * s

    parts = []

    x0, y0 = 0.0, 0.0
    x1, y1 = Ls, Ws

    # Base plinth + ground floor slab (extended footprint)
    base_x0 = x0 - side_wing_d * s - 0.8 * s
    base_x1 = x1 + side_wing_d * s + 0.8 * s
    base_y0 = y0 - 4.0 * s
    base_y1 = y1 + 1.0 * s
    add_box(parts, base_x0, base_x1, base_y0, base_y1, 0, floor_ts)

    # Openings layout on main front wall
    zgw0, zgw1 = floor_ts + 1.1 * s, floor_ts + 2.2 * s
    ztw0, ztw1 = floor_ts + h1s + 0.9 * s, floor_ts + h1s + 1.9 * s

    door_w = main_door_w_m * s
    door_h = 2.8 * s
    door_x0 = (x0 + x1) / 2 - door_w / 2
    door_x1 = (x0 + x1) / 2 + door_w / 2
    door_z0 = floor_ts + 0.15 * s
    door_z1 = floor_ts + door_h

    ground_window_centers = [4.5, 6.0, 18.0, 19.5]
    second_window_centers = [5.2, 7.0, 17.0, 18.8]

    openings_main_front = [(door_x0, door_x1, door_z0, door_z1)]
    for c in ground_window_centers:
        openings_main_front.append((x0 + (c - 0.6) * s, x0 + (c + 0.6) * s, zgw0, zgw1))

    # Back façade: mirrored windows + rear double door opposite main door
    rear_door_w = 1.9 * s
    rear_door_h = 2.5 * s
    rear_door_x0 = (x0 + x1) / 2 - rear_door_w / 2
    rear_door_x1 = (x0 + x1) / 2 + rear_door_w / 2
    rear_door_z0 = floor_ts + 0.15 * s
    rear_door_z1 = floor_ts + rear_door_h

    openings_main_back = [(rear_door_x0, rear_door_x1, rear_door_z0, rear_door_z1)]
    for c in ground_window_centers:
        openings_main_back.append((x0 + (c - 0.6) * s, x0 + (c + 0.6) * s, zgw0, zgw1))

    # Ground floor main shell
    add_shell_with_front_openings(parts, x0, y0, Ls, Ws, floor_ts, floor_ts + h1s, wall_ts, openings_main_front, openings_main_back)

    # 2nd floor shell (setback)
    inset = 0.35 * s
    openings_second_front = []
    openings_second_back = []
    for c in second_window_centers:
        o = (x0 + (c - 0.45) * s, x0 + (c + 0.45) * s, ztw0, ztw1)
        openings_second_front.append(o)
        openings_second_back.append(o)

    # central arched-top area approximated by tall opening rectangle
    openings_second_front.append(((x0 + 8.0 * s + x0 + 16.0 * s) / 2 - 1.0 * s,
                                  (x0 + 8.0 * s + x0 + 16.0 * s) / 2 + 1.0 * s,
                                  floor_ts + h1s + 0.9 * s,
                                  floor_ts + h1s + 2.1 * s))
    openings_second_back.append(((x0 + 8.0 * s + x0 + 16.0 * s) / 2 - 1.0 * s,
                                 (x0 + 8.0 * s + x0 + 16.0 * s) / 2 + 1.0 * s,
                                 floor_ts + h1s + 0.9 * s,
                                 floor_ts + h1s + 2.1 * s))

    add_shell_with_front_openings(
        parts,
        x0 + inset,
        y0 + inset,
        Ls - 2 * inset,
        Ws - 2 * inset,
        floor_ts + h1s,
        floor_ts + h1s + h2s,
        wall_ts,
        openings_second_front,
        openings_second_back,
    )

    # Transition belt to eliminate seam between lower and upper levels (front/back/sides)
    belt_z0 = floor_ts + h1s - 0.03 * s
    belt_z1 = floor_ts + h1s + 0.10 * s
    add_box(parts, x0, x1, y0, y0 + inset + 0.03 * s, belt_z0, belt_z1)                # front
    add_box(parts, x0, x1, y1 - inset - 0.03 * s, y1, belt_z0, belt_z1)                # back
    add_box(parts, x0, x0 + inset + 0.03 * s, y0, y1, belt_z0, belt_z1)                # left
    add_box(parts, x1 - inset - 0.03 * s, x1, y0, y1, belt_z0, belt_z1)                # right

    # Side wings as hollow blocks (no explicit openings to keep robust)
    add_shell_with_front_openings(
        parts,
        x0 - side_wing_d * s,
        y0 + 0.6 * s,
        3.8 * s + side_wing_d * s,
        Ws - 1.2 * s,
        floor_ts,
        floor_ts + h1s,
        wall_ts,
        [],
    )
    add_shell_with_front_openings(
        parts,
        x1 - 3.8 * s,
        y0 + 0.6 * s,
        3.8 * s + side_wing_d * s,
        Ws - 1.2 * s,
        floor_ts,
        floor_ts + h1s,
        wall_ts,
        [],
    )

    # Central front projection (framed mass, keeps entrance clear)
    cx0, cx1 = x0 + 8.0 * s, x0 + 16.0 * s
    proj_y0, proj_y1 = y0 - center_proj_d * s, y0 + 0.5 * s
    proj_z0, proj_z1 = floor_ts, floor_ts + h1s + 1.0 * s

    # clear opening zone for entrance volume (fully through front projection)
    ent_x0 = (cx0 + cx1) / 2 - 2.15 * s
    ent_x1 = (cx0 + cx1) / 2 + 2.15 * s
    ent_z0 = floor_ts
    ent_z1 = floor_ts + 3.35 * s

    # replace solid side walls with one octagonal column per side + upper beam
    col_side_r = 0.34 * s
    col_side_h = proj_z1 - proj_z0
    col_side_y = (proj_y0 + proj_y1) / 2.0

    left_col = trimesh.creation.cylinder(radius=col_side_r, height=col_side_h, sections=8)
    left_col.apply_translation([ent_x0 - 0.40 * s, col_side_y, proj_z0 + col_side_h / 2.0])
    parts.append(left_col)

    right_col = trimesh.creation.cylinder(radius=col_side_r, height=col_side_h, sections=8)
    right_col.apply_translation([ent_x1 + 0.40 * s, col_side_y, proj_z0 + col_side_h / 2.0])
    parts.append(right_col)

    # top beam over entrance opening
    add_box(parts, ent_x0, ent_x1, proj_y0, proj_y1, ent_z1, proj_z1)

    # Entrance portico: refined proportions (slimmer columns, cleaner balcony)
    col_w = 0.42 * s
    col_d = 0.42 * s
    col_h0 = floor_ts + 0.35 * s
    col_h1 = floor_ts + 3.85 * s

    front_col_y0, front_col_y1 = y0 - 2.10 * s, y0 - 2.10 * s + col_d
    rear_col_y0, rear_col_y1 = y0 - 1.15 * s, y0 - 1.15 * s + col_d

    # front pair (octagonal)
    fcx_l = cx0 + 0.80 * s + col_w / 2
    fcx_r = cx1 - 0.80 * s - col_w / 2
    fcy = (front_col_y0 + front_col_y1) / 2
    fr = col_w * 0.55
    fcol_l = trimesh.creation.cylinder(radius=fr, height=col_h1 - col_h0, sections=8)
    fcol_l.apply_translation([fcx_l, fcy, (col_h0 + col_h1) / 2])
    parts.append(fcol_l)
    fcol_r = trimesh.creation.cylinder(radius=fr, height=col_h1 - col_h0, sections=8)
    fcol_r.apply_translation([fcx_r, fcy, (col_h0 + col_h1) / 2])
    parts.append(fcol_r)

    # rear pair
    add_box(parts, cx0 + 1.55 * s, cx0 + 1.55 * s + col_w, rear_col_y0, rear_col_y1, col_h0, col_h1)
    add_box(parts, cx1 - 1.55 * s - col_w, cx1 - 1.55 * s, rear_col_y0, rear_col_y1, col_h0, col_h1)

    # capitals / bases for visual detail
    cap_h = 0.10 * s
    for (xa, xb, ya0, ya1) in [
        (cx0 + 0.80 * s, cx0 + 0.80 * s + col_w, front_col_y0, front_col_y1),
        (cx1 - 0.80 * s - col_w, cx1 - 0.80 * s, front_col_y0, front_col_y1),
        (cx0 + 1.55 * s, cx0 + 1.55 * s + col_w, rear_col_y0, rear_col_y1),
        (cx1 - 1.55 * s - col_w, cx1 - 1.55 * s, rear_col_y0, rear_col_y1),
    ]:
        add_box(parts, xa - 0.05 * s, xb + 0.05 * s, ya0 - 0.05 * s, ya1 + 0.05 * s, col_h0 - cap_h, col_h0)
        add_box(parts, xa - 0.05 * s, xb + 0.05 * s, ya0 - 0.05 * s, ya1 + 0.05 * s, col_h1, col_h1 + cap_h)

    # entablature + balcony slab
    add_box(parts, cx0 + 0.55 * s, cx1 - 0.55 * s, y0 - 2.22 * s, y0 - 0.62 * s, col_h1 + cap_h, col_h1 + cap_h + 0.30 * s)
    balcony_z0 = floor_ts + 3.95 * s
    # extend balcony floor to the sides and farther back
    add_box(parts, cx0 + 0.55 * s, cx1 - 0.55 * s, y0 - 2.18 * s, y0 - 0.58 * s, balcony_z0, balcony_z0 + 0.20 * s)

    # balcony handrail all around (front + left + right) with balusters
    rail_z0 = balcony_z0 + 0.20 * s
    rail_z1 = rail_z0 + 0.52 * s

    # front segment
    bx0, bx1 = cx0 + 0.72 * s, cx1 - 0.72 * s
    by0, by1 = y0 - 2.14 * s, y0 - 1.98 * s
    add_box(parts, bx0, bx1, by0, by1, rail_z1 - 0.08 * s, rail_z1)
    n_bal = 13
    b_w = 0.10 * s
    span = (bx1 - bx0 - n_bal * b_w) / (n_bal + 1)
    cx = bx0 + span
    for _ in range(n_bal):
        add_box(parts, cx, cx + b_w, by0 + 0.02 * s, by1 - 0.02 * s, rail_z0 + 0.02 * s, rail_z1 - 0.08 * s)
        cx += b_w + span

    # side segments (left/right)
    sx_th = 0.14 * s
    sy0 = y0 - 2.14 * s
    sy1 = y0 - 0.60 * s

    # left side top rail
    lx0, lx1 = cx0 + 0.90 * s, cx0 + 0.90 * s + sx_th
    add_box(parts, lx0, lx1, sy0, sy1, rail_z1 - 0.08 * s, rail_z1)
    # right side top rail
    rx0, rx1 = cx1 - 0.90 * s - sx_th, cx1 - 0.90 * s
    add_box(parts, rx0, rx1, sy0, sy1, rail_z1 - 0.08 * s, rail_z1)

    # left/right balusters
    n_side = 7
    b_d = 0.09 * s
    gap = (sy1 - sy0 - n_side * b_d) / (n_side + 1)
    ycur = sy0 + gap
    for _ in range(n_side):
        add_box(parts, lx0 + 0.02 * s, lx1 - 0.02 * s, ycur, ycur + b_d, rail_z0 + 0.02 * s, rail_z1 - 0.08 * s)
        add_box(parts, rx0 + 0.02 * s, rx1 - 0.02 * s, ycur, ycur + b_d, rail_z0 + 0.02 * s, rail_z1 - 0.08 * s)
        ycur += b_d + gap

    # rear balcony segment with OPENING at balcony door (no rail blocking doorway)
    back_y0, back_y1 = y0 - 0.66 * s, y0 - 0.52 * s
    door_gap_pad = 0.20 * s
    gap_x0 = door_x0 - door_gap_pad
    gap_x1 = door_x1 + door_gap_pad

    # top rail split left/right of door opening
    add_box(parts, bx0, gap_x0, back_y0, back_y1, rail_z1 - 0.08 * s, rail_z1)
    add_box(parts, gap_x1, bx1, back_y0, back_y1, rail_z1 - 0.08 * s, rail_z1)

    # balusters left of opening
    n_back = 11
    back_bw = 0.09 * s
    left_w = max(0.0, gap_x0 - bx0)
    right_w = max(0.0, bx1 - gap_x1)

    if left_w > 3 * back_bw:
        n_left = max(2, int(n_back * (left_w / (left_w + right_w + 1e-6))))
        left_span = (left_w - n_left * back_bw) / (n_left + 1)
        xcur = bx0 + left_span
        for _ in range(n_left):
            add_box(parts, xcur, xcur + back_bw, back_y0 + 0.02 * s, back_y1 - 0.02 * s, rail_z0 + 0.02 * s, rail_z1 - 0.08 * s)
            xcur += back_bw + left_span

    # balusters right of opening
    if right_w > 3 * back_bw:
        n_right = max(2, n_back - max(2, int(n_back * (left_w / (left_w + right_w + 1e-6)))))
        right_span = (right_w - n_right * back_bw) / (n_right + 1)
        xcur = gap_x1 + right_span
        for _ in range(n_right):
            add_box(parts, xcur, xcur + back_bw, back_y0 + 0.02 * s, back_y1 - 0.02 * s, rail_z0 + 0.02 * s, rail_z1 - 0.08 * s)
            xcur += back_bw + right_span

    # corner posts tie front/side/rear rails together
    post_w = 0.12 * s
    add_box(parts, lx0 - 0.01 * s, lx0 - 0.01 * s + post_w, by0, by0 + post_w, rail_z0 + 0.02 * s, rail_z1)
    add_box(parts, rx1 - post_w + 0.01 * s, rx1 + 0.01 * s, by0, by0 + post_w, rail_z0 + 0.02 * s, rail_z1)
    add_box(parts, lx0 - 0.01 * s, lx0 - 0.01 * s + post_w, back_y1 - post_w, back_y1, rail_z0 + 0.02 * s, rail_z1)
    add_box(parts, rx1 - post_w + 0.01 * s, rx1 + 0.01 * s, back_y1 - post_w, back_y1, rail_z0 + 0.02 * s, rail_z1)

    # Roofs
    main_roof_z = floor_ts + h1s + h2s
    wing_roof_z = floor_ts + h1s
    # Lower the main roof slightly for better proportion
    add_roof(parts, x0 + inset, y0 + inset, Ls - 2 * inset, Ws - 2 * inset,
             main_roof_z - 0.30 * s, 35, 0.35 * s, 0.22 * s, gable=True)
    add_roof(parts, x0 - side_wing_d * s, y0 + 0.6 * s, 3.8 * s + side_wing_d * s, Ws - 1.2 * s,
             wing_roof_z, 30, 0.25 * s, 0.20 * s, gable=True)
    add_roof(parts, x1 - 3.8 * s, y0 + 0.6 * s, 3.8 * s + side_wing_d * s, Ws - 1.2 * s,
             wing_roof_z, 30, 0.25 * s, 0.20 * s, gable=True)

    # Roof-wall sealing bands to remove tiny daylight gaps at eaves/intersections
    seal_h = 0.14 * s
    add_box(parts, x0 + inset - 0.05 * s, x1 - inset + 0.05 * s, y0 + inset - 0.05 * s, y1 - inset + 0.05 * s,
            (main_roof_z - 0.30 * s) - seal_h, (main_roof_z - 0.30 * s) + 0.03 * s)
    add_box(parts, x0 - side_wing_d * s - 0.03 * s, x0 + 3.8 * s + 0.03 * s, y0 + 0.6 * s - 0.03 * s, y1 - 0.6 * s + 0.03 * s,
            wing_roof_z - seal_h, wing_roof_z + 0.03 * s)
    add_box(parts, x1 - 3.8 * s - 0.03 * s, x1 + side_wing_d * s + 0.03 * s, y0 + 0.6 * s - 0.03 * s, y1 - 0.6 * s + 0.03 * s,
            wing_roof_z - seal_h, wing_roof_z + 0.03 * s)

    # Octagonal towers with OPEN windows (2 per wall: lower + upper) + octagonal roofs
    tower_r = 1.5 * s
    tower_h = h1s + h2s + 1.6 * s
    cone_h = 2.6 * s
    tower_wall_t = 0.18 * s

    def add_oct_tower_shell_open(parts, tx, ty, base_z, height, r, wall_t):
        n = 8
        side_len = 2.0 * r * math.sin(math.pi / n)
        panel_r = r - wall_t / 2.0

        win_w = 0.46 * side_len
        win_h = 0.95 * s
        low_z0, low_z1 = base_z + 1.0 * s, base_z + 1.0 * s + win_h
        high_z0, high_z1 = base_z + 4.3 * s, base_z + 4.3 * s + win_h

        z_levels = [base_z, low_z0, low_z1, high_z0, high_z1, base_z + height]

        def add_panel_piece(a_mid, x0, x1, z0, z1):
            if x1 <= x0 or z1 <= z0:
                return
            b = trimesh.creation.box(extents=[x1 - x0, wall_t, z1 - z0])
            # local center in panel coords
            cx_local = (x0 + x1) / 2.0
            cz_local = (z0 + z1) / 2.0
            b.apply_translation([cx_local, 0.0, cz_local])

            R = rotation_matrix(a_mid + math.pi / 2.0, [0, 0, 1])
            b.apply_transform(R)
            b.apply_translation([tx + panel_r * math.cos(a_mid), ty + panel_r * math.sin(a_mid), 0.0])
            parts.append(b)

        for i in range(n):
            a_mid = -math.pi + (2.0 * math.pi * (i + 0.5)) / n
            for z0, z1 in zip(z_levels[:-1], z_levels[1:]):
                if z1 <= z0:
                    continue
                in_low = not (z1 <= low_z0 or z0 >= low_z1)
                in_high = not (z1 <= high_z0 or z0 >= high_z1)
                if in_low or in_high:
                    add_panel_piece(a_mid, -side_len / 2.0, -win_w / 2.0, z0, z1)
                    add_panel_piece(a_mid,  win_w / 2.0,  side_len / 2.0, z0, z1)
                else:
                    add_panel_piece(a_mid, -side_len / 2.0, side_len / 2.0, z0, z1)

    tower_forward = 0.40 * (2.0 * tower_r)  # 40% of tower diameter
    for tx in [x0 + 1.2 * s, x1 - 1.2 * s]:
        ty = y0 + 1.0 * s - tower_forward

        add_oct_tower_shell_open(parts, tx, ty, floor_ts, tower_h, tower_r, tower_wall_t)

        # Octagonal roof cap
        roof = trimesh.creation.cone(radius=tower_r + 0.25 * s, height=cone_h, sections=8)
        cmin, _ = roof.bounds
        roof.apply_translation([tx, ty, floor_ts + tower_h - cmin[2]])
        parts.append(roof)

    # Chimneys
    add_box(parts, x0 + 6.0 * s, x0 + 6.8 * s, y0 + 5.3 * s, y0 + 6.0 * s,
            floor_ts + h1s + h2s + 1.2 * s, floor_ts + h1s + h2s + 2.3 * s)
    add_box(parts, x1 - 6.8 * s, x1 - 6.0 * s, y0 + 5.3 * s, y0 + 6.0 * s,
            floor_ts + h1s + h2s + 1.2 * s, floor_ts + h1s + h2s + 2.3 * s)

    # Front terrace + broader ceremonial stairs
    terrace_x0, terrace_x1 = cx0 + 0.75 * s, cx1 - 0.75 * s
    terrace_y0, terrace_y1 = y0 - 2.10 * s, y0 - 0.78 * s
    terrace_z0, terrace_z1 = floor_ts, floor_ts + 0.48 * s
    add_box(parts, terrace_x0, terrace_x1, terrace_y0, terrace_y1, terrace_z0, terrace_z1)

    # bridge landing from terrace to main wall to eliminate front gap strip
    bridge_w = 4.8 * s
    bx0 = (cx0 + cx1) / 2 - bridge_w / 2
    bx1 = (cx0 + cx1) / 2 + bridge_w / 2
    add_box(parts, bx0, bx1, terrace_y1, y0 + 0.02 * s, floor_ts, terrace_z1)

    # staircase centered at entry (top step lands exactly at terrace)
    stairs_w = 5.8 * s
    sx0 = (cx0 + cx1) / 2 - stairs_w / 2
    sx1 = (cx0 + cx1) / 2 + stairs_w / 2
    n_steps = 8
    step_h = (terrace_z1 - floor_ts) / n_steps
    step_d = 0.19 * s
    sy_front = terrace_y0 - n_steps * step_d
    for i in range(n_steps):
        zt = floor_ts + i * step_h
        yb = sy_front + i * step_d
        add_box(parts, sx0, sx1, yb, yb + step_d + 0.008 * s, floor_ts, zt + step_h)

    # handrails / balustrades
    rail_h = 0.48 * s
    left_rx0, left_rx1 = sx0 - 0.24 * s, sx0 - 0.08 * s
    right_rx0, right_rx1 = sx1 + 0.08 * s, sx1 + 0.24 * s
    rail_y0, rail_y1 = sy_front + 0.42 * s, terrace_y1

    # side parapets
    add_box(parts, left_rx0, left_rx1, rail_y0, rail_y1, floor_ts + 0.04 * s, floor_ts + rail_h)
    add_box(parts, right_rx0, right_rx1, rail_y0, rail_y1, floor_ts + 0.04 * s, floor_ts + rail_h)

    # terrace front rails removed (they were creating visual artifacts in slicer)
    # Door surround details: segmented frame (keeps doorway open) + pediment + sconces
    frame_pad = 0.22 * s
    fy0, fy1 = y0 - 0.08 * s, y0 + 0.18 * s
    # left and right jambs
    add_box(parts, door_x0 - frame_pad, door_x0 - 0.02 * s, fy0, fy1, door_z0 - 0.05 * s, door_z1 + 0.10 * s)
    add_box(parts, door_x1 + 0.02 * s, door_x1 + frame_pad, fy0, fy1, door_z0 - 0.05 * s, door_z1 + 0.10 * s)
    # lintel
    add_box(parts, door_x0 - frame_pad, door_x1 + frame_pad, fy0, fy1, door_z1 - 0.02 * s, door_z1 + 0.10 * s)
    # top cap over door
    add_box(parts, door_x0 - 0.35 * s, door_x1 + 0.35 * s, y0 - 0.18 * s, y0 + 0.22 * s, door_z1 + 0.05 * s, door_z1 + 0.30 * s)
    # sconces
    add_box(parts, door_x0 - 0.45 * s, door_x0 - 0.30 * s, y0 - 0.10 * s, y0 + 0.02 * s, door_z0 + 0.85 * s, door_z0 + 1.35 * s)
    add_box(parts, door_x1 + 0.30 * s, door_x1 + 0.45 * s, y0 - 0.10 * s, y0 + 0.02 * s, door_z0 + 0.85 * s, door_z0 + 1.35 * s)

    mesh = trimesh.util.concatenate(parts)
    pre = mesh.bounds[1] - mesh.bounds[0]
    mesh.apply_translation(-mesh.bounds[0])
    k = 180.0 / mesh.bounds[1][0]
    mesh.apply_scale(k)
    final = mesh.bounds[1] - mesh.bounds[0]

    m_to_mm = s * k
    ratio = 1000.0 / m_to_mm
    return mesh, pre, final, k, m_to_mm, ratio


if __name__ == '__main__':
    out = '/Users/rookcohen/.openclaw/workspace/mansion_entrance_v3_bolder_fit180.stl'
    m, pre, final, k, m_to_mm, ratio = build(180.0)
    m.export(out)
    print('Wrote:', out)
    print(f'Original bbox (mm): {pre[0]:.2f} x {pre[1]:.2f} x {pre[2]:.2f}')
    print(f'Final bbox (mm):    {final[0]:.2f} x {final[1]:.2f} x {final[2]:.2f}')
    print(f'Applied final scale multiplier: {k:.6f}')
    print(f'1 meter = {m_to_mm:.3f} mm in STL')
    print(f'Effective ratio: 1:{ratio:.2f}')
