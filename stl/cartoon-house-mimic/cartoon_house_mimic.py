#!/usr/bin/env python3
import math
import trimesh
import numpy as np


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


def build(target_main_length_mm=180.0):
    s = target_main_length_mm / 8.0  # 8m facade
    L, W, H = 8.0 * s, 5.5 * s, 3.0 * s
    floor_t = 0.30 * s   # 30 cm floor thickness
    wall_t = 0.20 * s    # 20 cm wall thickness
    roof_t = 0.2 * s
    overhang = 0.35 * s
    pitch = 30.0

    parts = []

    x0, y0 = 0.0, 0.0
    x1, y1 = L, W

    # Hollow main body: floor slab + wall shell
    add_box(parts, x0, x1, y0, y1, 0, floor_t)

    z0, z1 = floor_t, H

    # left and right walls (no openings)
    add_box(parts, x0, x0 + wall_t, y0 + wall_t, y1 - wall_t, z0, z1)
    add_box(parts, x1 - wall_t, x1, y0 + wall_t, y1 - wall_t, z0, z1)

    # front wall (door + window openings)
    front_openings = [
        (x0 + 1.70 * s, x0 + 2.50 * s, z0 + 0.15 * s, z0 + 2.10 * s),  # door opening
        (x0 + 3.90 * s, x0 + 5.50 * s, z0 + 1.20 * s, z0 + 2.25 * s),  # front window opening
    ]
    front_openings.sort(key=lambda t: t[0])

    for za, zb in [(z0, z0 + 0.15 * s), (z0 + 0.15 * s, z0 + 1.20 * s), (z0 + 1.20 * s, z0 + 2.10 * s), (z0 + 2.10 * s, z0 + 2.25 * s), (z0 + 2.25 * s, z1)]:
        cur = x0
        for xa, xb, oa, ob in front_openings:
            if not (zb <= oa or za >= ob):
                if xa > cur:
                    add_box(parts, cur, xa, y0, y0 + wall_t, za, zb)
                cur = max(cur, xb)
        if cur < x1:
            add_box(parts, cur, x1, y0, y0 + wall_t, za, zb)

    # back wall (solid)
    add_box(parts, x0, x1, y1 - wall_t, y1, z0, z1)

    # right-side window opening on side wall (near front-right corner)
    side_open = (x0 + 6.90 * s, x0 + 7.90 * s, z0 + 1.20 * s, z0 + 2.25 * s)
    xa, xb, za, zb = side_open
    add_box(parts, xa, xb, y1 - wall_t, y1, z0, za)
    add_box(parts, xa, xb, y1 - wall_t, y1, zb, z1)
    add_box(parts, x0 + wall_t, xa, y1 - wall_t, y1, za, zb)
    add_box(parts, xb, x1 - wall_t, y1 - wall_t, y1, za, zb)

    # Main gable roof
    xr0, xr1 = x0 - overhang, x1 + overhang
    yr0, yr1 = y0 - overhang, y1 + overhang
    Wr = yr1 - yr0
    run = Wr / 2
    rise = run * math.tan(math.radians(pitch))
    slant = math.sqrt(run * run + rise * rise)

    panel = trimesh.creation.box(extents=[xr1 - xr0, slant, roof_t])
    p1 = panel.copy()
    p1.apply_transform(trimesh.transformations.rotation_matrix(math.radians(pitch), [1, 0, 0]))
    p1.apply_translation([(xr0 + xr1) / 2, yr0 + Wr / 4, H + rise / 2])
    parts.append(p1)

    p2 = panel.copy()
    p2.apply_transform(trimesh.transformations.rotation_matrix(math.radians(180 - pitch), [1, 0, 0]))
    p2.apply_translation([(xr0 + xr1) / 2, yr0 + 3 * Wr / 4, H + rise / 2])
    parts.append(p2)

    # Gables
    rz = H + (W / 2) * math.tan(math.radians(pitch))
    add_tri_prism_x(parts, (y0, H), (y1, H), (y0 + W / 2, rz), x0, x0 + wall_t)
    add_tri_prism_x(parts, (y0, H), (y1, H), (y0 + W / 2, rz), x1 - wall_t, x1)

    # Porch platform
    porch_w = 2.6 * s
    porch_d = 1.2 * s
    porch_h = 0.35 * s
    px0 = x0 + 1.1 * s
    px1 = px0 + porch_w
    py0 = y0 - porch_d
    py1 = y0
    add_box(parts, px0, px1, py0, py1, 0.05 * s, porch_h)

    # Porch posts
    post_w = 0.18 * s
    add_box(parts, px0 + 0.15 * s, px0 + 0.15 * s + post_w, py0 + 0.15 * s, py0 + 0.15 * s + post_w, porch_h, H - 0.2 * s)
    add_box(parts, px1 - 0.15 * s - post_w, px1 - 0.15 * s, py0 + 0.15 * s, py0 + 0.15 * s + post_w, porch_h, H - 0.2 * s)

    # Porch shed roof
    add_box(parts, px0 - 0.2 * s, px1 + 0.2 * s, py0 - 0.05 * s, py1 + 0.15 * s, H - 0.35 * s, H - 0.1 * s)

    # Chimney
    add_box(parts, x0 + 6.2 * s, x0 + 6.8 * s, y0 + 3.8 * s, y0 + 4.4 * s, H + 0.4 * s, H + 1.8 * s)

    # (Openings are actual cut-through gaps in front and side walls; no recessed filler blocks.)

    mesh = trimesh.util.concatenate(parts)
    pre_size = mesh.bounds[1] - mesh.bounds[0]
    mesh.apply_translation(-mesh.bounds[0])
    k = 180.0 / mesh.bounds[1][0]
    mesh.apply_scale(k)
    m_to_mm = s * k
    ratio = 1000.0 / m_to_mm
    return mesh, pre_size, k, m_to_mm, ratio


if __name__ == '__main__':
    out = '/Users/rookcohen/.openclaw/workspace/cartoon_house_mimic_hollow_open_fit180.stl'
    m, pre_size, k, m_to_mm, ratio = build(180)
    m.export(out)
    size = m.bounds[1] - m.bounds[0]
    print('Wrote:', out)
    print(f'Original bbox (mm): {pre_size[0]:.2f} x {pre_size[1]:.2f} x {pre_size[2]:.2f}')
    print(f'Final bbox (mm):    {size[0]:.2f} x {size[1]:.2f} x {size[2]:.2f}')
    print(f'Applied final scale multiplier: {k:.6f}')
    print(f'1 meter = {m_to_mm:.3f} mm in STL')
    print(f'Effective ratio: 1:{ratio:.2f}')
