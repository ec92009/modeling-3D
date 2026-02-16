#!/usr/bin/env python3
"""
House + annex + tower STL generator.

Default model:
- Main body: 7m x 5m x 3m
- Annex: 6m x 4m x 3m
- Tower: cylinder Ø5m, 6m high, conical roof
- Tower door: 1x2m (recess)
- Tower window: 1x1m, 1m above door (recess)
- Roof pitch (house): 30°
- Overhang: 0.2m
- Floor extension: 1m all around

Examples:
  python3 house_generator.py --output ./house.stl
  python3 house_generator.py --fit-cube-mm 180 --output ./house_fit180.stl
"""

import argparse
import math
import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix


def build_model(target_main_length_mm=180.0):
    s = target_main_length_mm / 7.0

    # house dims
    L1, W1, H1 = 7.0 * s, 5.0 * s, 3.0 * s
    L2, W2, H2 = 6.0 * s, 4.0 * s, 3.0 * s

    wall_t = 0.2 * s
    floor_t = 0.3 * s
    ceil_t = 0.2 * s
    roof_t = 0.2 * s
    roof_overhang = 0.2 * s
    floor_ext = 1.0 * s
    pitch = 30.0

    door_w, door_h = 1.0 * s, 2.0 * s
    win_w, win_h = 1.0 * s, 1.0 * s

    parts = []

    def add_box(x0, x1, y0, y1, z0, z1):
        if x1 <= x0 or y1 <= y0 or z1 <= z0:
            return
        b = trimesh.creation.box(extents=[x1 - x0, y1 - y0, z1 - z0])
        b.apply_translation([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2])
        parts.append(b)

    def add_tri_prism_x(A, B, C, x0, x1):
        ay, az = A
        by, bz = B
        cy, cz = C
        v = [[x0, ay, az], [x0, by, bz], [x0, cy, cz], [x1, ay, az], [x1, by, bz], [x1, cy, cz]]
        f = [[0, 2, 1], [3, 4, 5], [0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4], [2, 0, 3], [2, 3, 5]]
        parts.append(trimesh.Trimesh(vertices=np.array(v, float), faces=np.array(f, int), process=False))

    def add_roof_and_gables(xo, yo, L, W, H):
        xr0, xr1 = xo - roof_overhang, xo + L + roof_overhang
        yr0, yr1 = yo - roof_overhang, yo + W + roof_overhang
        Lr, Wr = xr1 - xr0, yr1 - yr0

        run = Wr / 2
        rise = run * math.tan(math.radians(pitch))
        slant = math.sqrt(run * run + rise * rise)

        panel = trimesh.creation.box(extents=[Lr, slant, roof_t])

        p1 = panel.copy()
        p1.apply_transform(rotation_matrix(math.radians(pitch), [1, 0, 0]))
        p1.apply_translation([(xr0 + xr1) / 2, yr0 + Wr / 4, H + rise / 2])
        parts.append(p1)

        p2 = panel.copy()
        p2.apply_transform(rotation_matrix(math.radians(180 - pitch), [1, 0, 0]))
        p2.apply_translation([(xr0 + xr1) / 2, yr0 + 3 * Wr / 4, H + rise / 2])
        parts.append(p2)

        # ridge cap
        ridge_w, ridge_h = 0.25 * s, 0.12 * s
        ry = yo + W / 2
        rz = H + (W / 2) * math.tan(math.radians(pitch))
        add_box(xr0, xr1, ry - ridge_w / 2, ry + ridge_w / 2, rz, rz + ridge_h)

        # gable infill (x ends)
        A, B, C = (yo, H), (yo + W, H), (yo + W / 2, rz)
        add_tri_prism_x(A, B, C, xo, xo + wall_t)
        add_tri_prism_x(A, B, C, xo + L - wall_t, xo + L)

    def add_body(xo, yo, L, W, H, front_windows, rear_windows, with_front_door):
        zf, zc = floor_t, H - ceil_t

        add_box(xo, xo + L, yo, yo + W, 0, floor_t)
        add_box(xo, xo + L, yo, yo + W, H - ceil_t, H)

        add_box(xo, xo + wall_t, yo + wall_t, yo + W - wall_t, zf, zc)
        add_box(xo + L - wall_t, xo + L, yo + wall_t, yo + W - wall_t, zf, zc)

        zwb, zwt = zf + win_h, zf + 2 * win_h

        gaps = []
        if with_front_door:
            gaps.append((xo + 20.0, xo + 20.0 + door_w, zf, zf + door_h))

        if front_windows == 1:
            gaps.append((xo + L - 25.0 - win_w, xo + L - 25.0, zwb, zwt))
        elif front_windows > 1:
            margin, span = xo + 25.0, L - 50.0
            step = span / (front_windows + 1)
            for i in range(front_windows):
                cx = margin + step * (i + 1)
                gaps.append((cx - win_w / 2, cx + win_w / 2, zwb, zwt))

        for z0, z1 in [(zf, zwb), (zwb, zwt), (zwt, zc)]:
            active = sorted([(a, b) for a, b, g0, g1 in gaps if not (g1 <= z0 or g0 >= z1)])
            cur = xo
            for a, b in active:
                if a > cur:
                    add_box(cur, a, yo, yo + wall_t, z0, z1)
                cur = max(cur, b)
            if cur < xo + L:
                add_box(cur, xo + L, yo, yo + wall_t, z0, z1)

        rw = []
        if rear_windows > 0:
            margin, span = xo + 25.0, L - 50.0
            step = span / (rear_windows + 1)
            for i in range(rear_windows):
                cx = margin + step * (i + 1)
                rw.append((cx - win_w / 2, cx + win_w / 2))

        add_box(xo, xo + L, yo + W - wall_t, yo + W, zf, zwb)
        cur = xo
        for a, b in rw:
            if a > cur:
                add_box(cur, a, yo + W - wall_t, yo + W, zwb, zwt)
            cur = max(cur, b)
        if cur < xo + L:
            add_box(cur, xo + L, yo + W - wall_t, yo + W, zwb, zwt)
        add_box(xo, xo + L, yo + W - wall_t, yo + W, zwt, zc)

        add_roof_and_gables(xo, yo, L, W, H)

    # house bodies
    x1, y1 = 0.0, 0.0
    x2, y2 = L1, 0.0
    add_body(x1, y1, L1, W1, H1, 1, 2, True)
    add_body(x2, y2, L2, W2, H2, 2, 2, False)

    # tower
    tower_r = (5.0 * s) / 2
    tower_h = 6.0 * s
    cone_h = 2.5 * s

    cx = x1 - tower_r + wall_t
    cy = y1 + W1 / 2

    # Build hollow tower shell with explicit openings (no boolean backend required)
    inner_r = max(tower_r - wall_t, tower_r * 0.7)
    nseg = 96

    # Opening vertical spans
    door_z0 = floor_t
    door_z1 = floor_t + door_h

    win_top_z0 = floor_t + door_h + 1.0 * s
    win_top_z1 = win_top_z0 + win_h

    # back lower window aligned to first-floor window band of main house
    # (same as zwb..zwt in house body: zf+win_h .. zf+2*win_h)
    win_back_low_z0 = floor_t + win_h
    win_back_low_z1 = floor_t + 2.0 * win_h

    # Opening angular spans around front/back normals
    # front = -90° (toward -Y), back = +90° (toward +Y)
    def ang_span(width):
        return max(0.03, min(math.pi / 3, width / (2.0 * tower_r)))

    door_da = ang_span(door_w)
    win_da = ang_span(win_w)

    z_breaks = sorted(set([floor_t, door_z0, door_z1, win_back_low_z0, win_back_low_z1, win_top_z0, win_top_z1, tower_h]))

    def wrap_pi(a):
        while a <= -math.pi:
            a += 2 * math.pi
        while a > math.pi:
            a -= 2 * math.pi
        return a

    def in_span(a, center, half):
        return abs(wrap_pi(a - center)) <= half

    def is_open(a_mid, z0, z1):
        # front openings
        if in_span(a_mid, -math.pi / 2, door_da) and not (z1 <= door_z0 or z0 >= door_z1):
            return True
        if in_span(a_mid, -math.pi / 2, win_da) and not (z1 <= win_top_z0 or z0 >= win_top_z1):
            return True
        # back openings (opposite side)
        if in_span(a_mid, math.pi / 2, win_da) and not (z1 <= win_back_low_z0 or z0 >= win_back_low_z1):
            return True
        if in_span(a_mid, math.pi / 2, win_da) and not (z1 <= win_top_z0 or z0 >= win_top_z1):
            return True
        return False

    shell_parts = []

    def wall_wedge(a0, a1, z0, z1):
        c0, s0 = math.cos(a0), math.sin(a0)
        c1, s1 = math.cos(a1), math.sin(a1)

        vo0 = [cx + tower_r * c0, cy + tower_r * s0]
        vo1 = [cx + tower_r * c1, cy + tower_r * s1]
        vi0 = [cx + inner_r * c0, cy + inner_r * s0]
        vi1 = [cx + inner_r * c1, cy + inner_r * s1]

        v = np.array([
            [vo0[0], vo0[1], z0], [vo1[0], vo1[1], z0], [vo1[0], vo1[1], z1], [vo0[0], vo0[1], z1],
            [vi0[0], vi0[1], z0], [vi1[0], vi1[1], z0], [vi1[0], vi1[1], z1], [vi0[0], vi0[1], z1],
        ], dtype=float)

        f = np.array([
            [0, 1, 2], [0, 2, 3],  # outer face
            [5, 4, 7], [5, 7, 6],  # inner face (reversed)
            [0, 4, 5], [0, 5, 1],  # bottom ring segment
            [3, 2, 6], [3, 6, 7],  # top ring segment
            [0, 3, 7], [0, 7, 4],  # side at a0
            [1, 5, 6], [1, 6, 2],  # side at a1
        ], dtype=int)

        return trimesh.Trimesh(vertices=v, faces=f, process=False)

    for i in range(nseg):
        a0 = -math.pi + (2 * math.pi * i) / nseg
        a1 = -math.pi + (2 * math.pi * (i + 1)) / nseg
        amid = wrap_pi((a0 + a1) * 0.5)

        for j in range(len(z_breaks) - 1):
            z0, z1 = z_breaks[j], z_breaks[j + 1]
            if z1 <= z0:
                continue
            if is_open(amid, z0, z1):
                continue
            shell_parts.append(wall_wedge(a0, a1, z0, z1))

    tower_shell = trimesh.util.concatenate(shell_parts)

    collar_h = 0.6 * s
    collar = trimesh.creation.cylinder(radius=tower_r, height=collar_h, sections=128)
    collar.apply_translation([cx, cy, tower_h - collar_h / 2])

    cone = trimesh.creation.cone(radius=tower_r + roof_overhang, height=cone_h, sections=128)
    # User requirement: cone base starts exactly at tower top (6m)
    cmin, cmax = cone.bounds
    cone.apply_translation([cx, cy, tower_h - cmin[2]])

    tower_combo = trimesh.util.concatenate([tower_shell, collar, cone])
    parts.append(tower_combo)

    # unified floor
    min_x = min(x1 - floor_ext, x2 - floor_ext, cx - tower_r - floor_ext)
    max_x = max(x1 + L1 + floor_ext, x2 + L2 + floor_ext, cx + tower_r + floor_ext)
    min_y = min(y1 - floor_ext, y2 - floor_ext, cy - tower_r - floor_ext)
    max_y = max(y1 + W1 + floor_ext, y2 + W2 + floor_ext, cy + tower_r + floor_ext)
    add_box(min_x, max_x, min_y, max_y, 0, floor_t)

    return trimesh.util.concatenate(parts), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='house_model.stl')
    ap.add_argument('--fit-cube-mm', type=float, default=None)
    args = ap.parse_args()

    mesh, s = build_model()

    orig_size = mesh.bounds[1] - mesh.bounds[0]
    k = 1.0
    if args.fit_cube_mm and args.fit_cube_mm > 0:
        k = args.fit_cube_mm / float(orig_size.max())
        mesh.apply_translation(-mesh.bounds[0])
        mesh.apply_scale(k)

    mesh.export(args.output)

    final_size = mesh.bounds[1] - mesh.bounds[0]
    m_to_mm = s * k
    ratio = 1000.0 / m_to_mm

    print(f'Wrote: {args.output}')
    print(f'Original bbox (mm): {orig_size[0]:.2f} x {orig_size[1]:.2f} x {orig_size[2]:.2f}')
    print(f'Final bbox (mm):    {final_size[0]:.2f} x {final_size[1]:.2f} x {final_size[2]:.2f}')
    print(f'Applied final scale multiplier: {k:.6f}')
    print(f'1 meter = {m_to_mm:.3f} mm in STL')
    print(f'Effective ratio: 1:{ratio:.2f}')


if __name__ == '__main__':
    main()
