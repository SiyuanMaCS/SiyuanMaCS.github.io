from pathlib import Path

import math

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

import artifact_io


W, H = 256, 144
FRAMES = 20
FPS = 4
SS = 4
SW, SH = W * SS, H * SS
CASE_DIR = Path(__file__).resolve().parent


def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def ease_out(value):
    value = clamp(value)
    return 1.0 - (1.0 - value) ** 2.4


def soft_noise(width, height, cells, seed):
    rng = np.random.default_rng(seed)
    small = rng.random((max(2, height // cells), max(2, width // cells)))
    image = Image.fromarray((small * 255).astype(np.uint8), "L")
    image = image.resize((width, height), Image.Resampling.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def radial_gradient(width, height, center_x, center_y, radius):
    yy, xx = np.mgrid[0:height, 0:width]
    distance = np.sqrt((xx - center_x) ** 2 + (yy - center_y) ** 2)
    return np.clip(1.0 - distance / radius, 0.0, 1.0)


def composite_mask(base, layer, mask):
    mask = np.asarray(mask, dtype=np.float32)[..., None] / 255.0
    return base * (1.0 - mask) + np.asarray(layer, dtype=np.float32) * mask


def ellipse_points(cx, cy, rx, ry, angle=0.0, count=144):
    points = []
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    for index in range(count):
        theta = 2.0 * math.pi * index / count
        lx = rx * math.cos(theta)
        ly = ry * math.sin(theta)
        x = cx + lx * cos_a + ly * sin_a
        y = cy - lx * sin_a + ly * cos_a
        points.append((x, y))
    return points


def local_to_world(cx, cy, lx, ly, angle):
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return (
        cx + lx * cos_a + ly * sin_a,
        cy - lx * sin_a + ly * cos_a,
    )


def make_window_scene():
    width, height = SW, SH
    yy, xx = np.mgrid[0:height, 0:width]
    image = np.zeros((height, width, 3), dtype=np.float32)

    # Deep black exterior with a few restrained stars.
    image[:] = np.array([2.0, 4.0, 7.0], dtype=np.float32)
    rng = np.random.default_rng(91)
    for _ in range(58):
        x = int(rng.uniform(0, width))
        y = int(rng.uniform(0, height))
        value = float(rng.uniform(35, 175))
        radius = int(rng.integers(1, 4))
        glow = radial_gradient(width, height, x, y, radius * 2.4)
        image += glow[..., None] * np.array([value, value * 0.92, value * 0.78])

    # The curved blue limb of Earth enters through the lower half of the window.
    earth_cx, earth_cy = width * 0.54, height * 1.23
    earth_r = height * 0.87
    earth_d = np.sqrt((xx - earth_cx) ** 2 + (yy - earth_cy) ** 2)
    earth_mask = np.clip((earth_r + 2.5 - earth_d) / 4.0, 0.0, 1.0)
    latitude = np.clip((earth_cy - yy) / earth_r, 0.0, 1.0)
    ocean = np.stack(
        [
            10.0 + 12.0 * latitude,
            47.0 + 42.0 * latitude,
            90.0 + 50.0 * latitude,
        ],
        axis=-1,
    )
    cloud_field = soft_noise(width, height, 22, 7)
    cloud_field = np.asarray(
        Image.fromarray((cloud_field * 255).astype(np.uint8), "L").filter(
            ImageFilter.GaussianBlur(5.0)
        ),
        dtype=np.float32,
    ) / 255.0
    cloud_field = np.clip((cloud_field - 0.47) * 2.6, 0.0, 0.65)
    earth = ocean * (1.0 - cloud_field[..., None]) + np.array(
        [205.0, 218.0, 222.0]
    ) * cloud_field[..., None]
    limb = np.exp(-((earth_d - earth_r) / (height * 0.008)) ** 2)
    earth += limb[..., None] * np.array([95.0, 155.0, 205.0])
    image = composite_mask(image, earth, earth_mask * 255)

    return Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), "RGB")


def build_background():
    yy, xx = np.mgrid[0:SH, 0:SW]
    nx = xx / max(1, SW - 1)
    ny = yy / max(1, SH - 1)

    top = np.array([58.0, 63.0, 64.0])
    bottom = np.array([19.0, 23.0, 25.0])
    blend = (0.58 * ny + 0.42 * smoothstep(ny))[..., None]
    base = top * (1.0 - blend) + bottom * blend
    base += (0.5 - nx)[..., None] * np.array([8.0, 9.0, 10.0])

    texture = soft_noise(SW, SH, 9, 12) - 0.5
    brushed = soft_noise(SW, SH, 2, 33) - 0.5
    base += texture[..., None] * 7.5
    base += brushed[..., None] * np.array([2.2, 2.0, 1.8])
    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")

    draw = ImageDraw.Draw(image, "RGBA")

    # Interior wall panels, seams and structural ribs.
    panel_right = int(SW * 0.46)
    panel_split = int(SH * 0.61)
    draw.rectangle((panel_right, 0, SW, panel_split), fill=(67, 69, 66, 56))
    draw.rectangle((panel_right, panel_split, SW, SH), fill=(31, 35, 36, 76))
    draw.rectangle((0, int(SH * 0.75), SW, SH), fill=(20, 25, 27, 82))

    seam_color = (9, 12, 13, 175)
    highlight_color = (156, 159, 153, 62)
    for fraction in (0.09, 0.31, 0.45, 0.72, 0.88):
        x = int(SW * fraction)
        draw.line((x, 0, x, SH), fill=seam_color, width=3 * SS)
        draw.line((x + 2 * SS, 0, x + 2 * SS, SH), fill=highlight_color, width=1 * SS)

    for fraction in (0.17, 0.44, 0.63, 0.83):
        y = int(SH * fraction)
        draw.line((0, y, SW, y), fill=(8, 11, 12, 115), width=2 * SS)
        draw.line((0, y + 2 * SS, SW, y + 2 * SS), fill=(143, 146, 140, 36), width=SS)

    # Curved ribs and a shallow hatch lip give the frame a station-module read.
    rib_x = int(SW * 0.39)
    draw.rounded_rectangle(
        (rib_x - 9 * SS, -10 * SS, rib_x + 10 * SS, SH + 10 * SS),
        radius=9 * SS,
        fill=(33, 38, 39, 150),
        outline=(123, 127, 123, 62),
        width=2 * SS,
    )
    draw.line(
        (rib_x - 4 * SS, 0, rib_x - 4 * SS, SH),
        fill=(176, 179, 172, 48),
        width=2 * SS,
    )
    draw.arc(
        (-SW * 0.38, SH * 0.04, SW * 0.22, SH * 1.03),
        272,
        82,
        fill=(134, 137, 131, 76),
        width=3 * SS,
    )

    # Lower equipment rail with clipped, out-of-focus control blocks.
    rail_y = int(SH * 0.80)
    draw.rounded_rectangle(
        (int(SW * 0.42), rail_y, SW, rail_y + 7 * SS),
        radius=3 * SS,
        fill=(82, 85, 80, 150),
        outline=(174, 176, 166, 56),
        width=SS,
    )
    for index, x_frac in enumerate((0.52, 0.61, 0.79, 0.91)):
        x = int(SW * x_frac)
        w = int(SW * (0.055 + 0.01 * (index % 2)))
        draw.rounded_rectangle(
            (x, rail_y + 10 * SS, x + w, rail_y + 24 * SS),
            radius=2 * SS,
            fill=(37, 42, 42, 135),
            outline=(103, 108, 104, 58),
            width=SS,
        )

    # Small status lamps, kept soft and secondary.
    for x_frac, color in ((0.64, (83, 207, 177)), (0.87, (230, 168, 83))):
        cx = int(SW * x_frac)
        cy = int(SH * 0.885)
        draw.ellipse(
            (cx - 2 * SS, cy - 2 * SS, cx + 2 * SS, cy + 2 * SS),
            fill=(*color, 210),
        )

    # Window housing.
    wx, wy = int(SW * 0.225), int(SH * 0.35)
    rx, ry = int(SW * 0.145), int(SH * 0.245)
    draw.ellipse(
        (wx - rx - 8 * SS, wy - ry - 8 * SS, wx + rx + 8 * SS, wy + ry + 8 * SS),
        fill=(44, 48, 48, 240),
    )
    draw.ellipse(
        (wx - rx - 4 * SS, wy - ry - 4 * SS, wx + rx + 4 * SS, wy + ry + 4 * SS),
        fill=(133, 138, 136, 240),
    )
    draw.ellipse(
        (wx - rx, wy - ry, wx + rx, wy + ry),
        fill=(22, 26, 28, 255),
    )

    window_scene = make_window_scene()
    window_mask = Image.new("L", (SW, SH), 0)
    mask_draw = ImageDraw.Draw(window_mask)
    mask_draw.ellipse((wx - rx, wy - ry, wx + rx, wy + ry), fill=255)
    mask_crop = window_mask.crop((wx - rx, wy - ry, wx + rx, wy + ry))
    scene_crop = window_scene.resize((2 * rx, 2 * ry), Image.Resampling.LANCZOS)
    image.paste(scene_crop, (wx - rx, wy - ry), mask_crop)

    draw = ImageDraw.Draw(image, "RGBA")
    draw.arc(
        (wx - rx, wy - ry, wx + rx, wy + ry),
        188,
        350,
        fill=(220, 225, 220, 132),
        width=2 * SS,
    )
    draw.arc(
        (wx - rx + 2 * SS, wy - ry + 2 * SS, wx + rx - 2 * SS, wy + ry - 2 * SS),
        8,
        166,
        fill=(4, 7, 8, 190),
        width=2 * SS,
    )
    draw.line(
        (
            wx - rx * 0.55,
            wy - ry * 0.64,
            wx + rx * 0.18,
            wy + ry * 0.54,
        ),
        fill=(214, 229, 235, 28),
        width=4 * SS,
    )
    for angle in (0.22, 1.78, 3.32, 4.83):
        bx = wx + math.cos(angle) * (rx + 6 * SS)
        by = wy + math.sin(angle) * (ry + 6 * SS)
        draw.ellipse(
            (bx - 2 * SS, by - 2 * SS, bx + 2 * SS, by + 2 * SS),
            fill=(38, 42, 42, 210),
            outline=(183, 187, 181, 86),
            width=SS,
        )

    # Overhead luminaire and its broad reflected glow.
    light = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))
    light_draw = ImageDraw.Draw(light, "RGBA")
    light_draw.rounded_rectangle(
        (int(SW * 0.34), int(SH * 0.035), int(SW * 0.63), int(SH * 0.070)),
        radius=4 * SS,
        fill=(233, 232, 210, 120),
    )
    light_draw.rounded_rectangle(
        (int(SW * 0.37), int(SH * 0.042), int(SW * 0.60), int(SH * 0.057)),
        radius=2 * SS,
        fill=(255, 250, 220, 235),
    )
    bloom = light.filter(ImageFilter.GaussianBlur(12 * SS))
    image = Image.alpha_composite(image.convert("RGBA"), bloom)
    image = Image.alpha_composite(image, light)

    # Slight depth-of-field; the cup and water will remain sharp.
    image = image.filter(ImageFilter.GaussianBlur(1.4 * SS))
    arr = np.asarray(image.convert("RGB"), dtype=np.float32)
    arr += (np.random.default_rng(55).random((SH, SW, 1)) - 0.5) * 4.0
    return np.clip(arr, 0, 255)


def make_cup_geometry(frame_index, total_frames):
    t = frame_index / max(1, total_frames - 1)
    angle = math.radians(13.0 + 14.5 * smoothstep(t))
    cx, cy = 170.0 * SS, 91.0 * SS
    rim_half_width = 27.0 * SS
    base_half_width = 18.0 * SS
    half_height = 40.0 * SS
    rim_y = -half_height
    base_y = half_height
    water_y = (-7.0 + 8.0 * smoothstep(t)) * SS

    def body_points():
        return [
            local_to_world(cx, cy, -rim_half_width, rim_y, angle),
            local_to_world(cx, cy, rim_half_width, rim_y, angle),
            local_to_world(cx, cy, base_half_width, base_y, angle),
            local_to_world(cx, cy, -base_half_width, base_y, angle),
        ]

    def half_width_at(y):
        ratio = (y - rim_y) / (base_y - rim_y)
        return rim_half_width * (1.0 - ratio) + base_half_width * ratio

    water_half = half_width_at(water_y) - 1.5 * SS
    water_points = [
        local_to_world(cx, cy, -water_half, water_y, angle),
        local_to_world(cx, cy, water_half, water_y, angle),
        local_to_world(cx, cy, base_half_width - 1.5 * SS, base_y, angle),
        local_to_world(cx, cy, -base_half_width + 1.5 * SS, base_y, angle),
    ]

    return {
        "t": t,
        "angle": angle,
        "center": (cx, cy),
        "body": body_points(),
        "water": water_points,
        "water_y": water_y,
        "water_half": water_half,
        "rim_half": rim_half_width,
        "base_half": base_half_width,
        "half_height": half_height,
        "spout": local_to_world(
            cx, cy, -rim_half_width * 0.78, rim_y + 1.5 * SS, angle
        ),
    }


def mask_from_polygons(size, polygons, ellipses=()):
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        draw.polygon(polygon, fill=255)
    for ellipse in ellipses:
        draw.ellipse(ellipse, fill=255)
    return mask


def add_refracted_liquid(base, mask, tint, opacity, time_value):
    height, width = base.shape[:2]
    alpha = np.asarray(mask, dtype=np.float32) / 255.0
    if float(alpha.max()) <= 0.0:
        return base

    bbox = mask.getbbox()
    if bbox is None:
        return base
    x0, y0, x1, y1 = bbox
    pad = 8 * SS
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(width, x1 + pad), min(height, y1 + pad)

    roi = base[y0:y1, x0:x1]
    roi_alpha = alpha[y0:y1, x0:x1]
    local_h, local_w = roi.shape[:2]
    yy, xx = np.mgrid[0:local_h, 0:local_w]
    shift_x = np.rint(
        (2.6 * SS) * np.sin(yy * 0.043 + time_value * 1.7)
        + (0.9 * SS) * np.sin(xx * 0.097 - time_value)
    ).astype(np.int32)
    shift_y = np.rint(
        (1.8 * SS) * np.cos(xx * 0.035 - time_value * 1.2)
        + (0.6 * SS) * np.sin(yy * 0.111 + time_value)
    ).astype(np.int32)
    sample_x = np.clip(np.arange(local_w)[None, :] + shift_x, 0, width - 1)
    sample_y = np.clip(np.arange(local_h)[:, None] + shift_y, 0, height - 1)
    refracted = base[sample_y, sample_x]

    tint_arr = np.asarray(tint, dtype=np.float32)
    liquid = refracted * 0.62 + tint_arr * 0.38
    liquid += np.array([10.0, 18.0, 21.0], dtype=np.float32)
    blend = np.clip(roi_alpha * opacity, 0.0, 0.78)[..., None]
    base[y0:y1, x0:x1] = roi * (1.0 - blend) + liquid * blend
    return base


def add_edge(mask, color, width, opacity, blur=0.0):
    layer = Image.new("RGBA", mask.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")
    draw.bitmap((0, 0), mask, fill=(*color, opacity))
    layer.putalpha(ImageChops.subtract(layer.getchannel("A"), mask))
    if width > 0:
        eroded = mask.filter(ImageFilter.MinFilter(2 * width + 1))
        layer.putalpha(ImageChops.subtract(mask, eroded))
    if blur > 0:
        layer = layer.filter(ImageFilter.GaussianBlur(blur))
    return layer


def render_stream_mask(geometry, frame_index):
    t = geometry["t"]
    p0 = np.array(geometry["spout"], dtype=np.float64)
    p1 = p0 + np.array([-22.0 * SS, 0.5 * SS])
    p2 = p0 + np.array([-56.0 * SS, 16.0 * SS])
    p3 = p0 + np.array([-84.0 * SS, 42.0 * SS])

    progress = 0.13 + 0.87 * ease_out(t)
    samples = 70
    centers = []
    widths = []
    for index in range(samples):
        s = progress * index / (samples - 1)
        inv = 1.0 - s
        point = (
            inv**3 * p0
            + 3.0 * inv**2 * s * p1
            + 3.0 * inv * s**2 * p2
            + s**3 * p3
        )
        tangent = (
            3.0 * inv**2 * (p1 - p0)
            + 6.0 * inv * s * (p2 - p1)
            + 3.0 * s**2 * (p3 - p2)
        )
        tangent /= max(1e-6, np.linalg.norm(tangent))
        normal = np.array([-tangent[1], tangent[0]])
        ripple = (
            1.25 * SS * math.sin(17.0 * s - t * 3.2)
            + 0.55 * SS * math.sin(39.0 * s + 0.8)
        )
        point = point + normal * ripple
        width = (5.6 - 2.5 * s) * SS
        width *= 1.0 + 0.13 * math.sin(28.0 * s + frame_index * 0.22)
        centers.append(point)
        widths.append(width)

    left = []
    right = []
    for index, (point, width) in enumerate(zip(centers, widths)):
        prev_point = centers[max(0, index - 1)]
        next_point = centers[min(samples - 1, index + 1)]
        tangent = next_point - prev_point
        tangent /= max(1e-6, np.linalg.norm(tangent))
        normal = np.array([-tangent[1], tangent[0]])
        left.append(tuple(point + normal * width))
        right.append(tuple(point - normal * width))

    mask = Image.new("L", (SW, SH), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(left + list(reversed(right)), fill=255)
    tip = centers[-1]
    tip_r = widths[-1] * 1.18
    draw.ellipse(
        (tip[0] - tip_r, tip[1] - tip_r, tip[0] + tip_r, tip[1] + tip_r),
        fill=255,
    )

    # Surface-tension bulbs along the moving ribbon.
    for index in range(4, samples - 2, 11):
        if index / samples > progress - 0.06:
            continue
        r = widths[index] * (1.10 + 0.15 * math.sin(index))
        point = centers[index]
        draw.ellipse(
            (point[0] - r, point[1] - r, point[0] + r, point[1] + r),
            fill=255,
        )
    return mask.filter(ImageFilter.GaussianBlur(0.55 * SS))


def droplet_tracks(geometry):
    spout = geometry["spout"]
    specs = [
        (-58, 25, 0.17, 3.3, 0.1),
        (-72, 8, 0.28, 4.8, 1.2),
        (-48, 38, 0.34, 5.5, 2.4),
        (-86, 31, 0.43, 2.7, 0.8),
        (-39, 16, 0.50, 2.3, 2.0),
        (-64, 52, 0.56, 4.0, 1.6),
        (-92, 16, 0.61, 3.6, 2.8),
        (-43, 48, 0.66, 2.2, 0.4),
        (-77, 58, 0.70, 3.1, 2.2),
        (-104, 39, 0.75, 2.5, 1.1),
        (-54, 66, 0.79, 2.0, 2.7),
        (-31, 34, 0.83, 1.7, 1.8),
        (-88, 73, 0.87, 2.8, 0.2),
        (-112, 57, 0.90, 2.1, 2.4),
    ]
    tracks = []
    for dx, dy, birth, radius, phase in specs:
        tracks.append(
            {
                "offset": np.array([dx * SS, dy * SS]),
                "birth": birth,
                "radius": radius * SS,
                "phase": phase,
            }
        )
    return spout, tracks


def render_droplets(geometry, frame_index):
    spout, tracks = droplet_tracks(geometry)
    t = geometry["t"]
    mask = Image.new("L", (SW, SH), 0)
    draw = ImageDraw.Draw(mask)
    centers = []
    for track in tracks:
        age = (t - track["birth"]) / max(1e-6, 1.0 - track["birth"])
        if age <= 0.0:
            continue
        travel = smoothstep(age) * 0.78 + age * 0.22
        wobble = np.array(
            [
                3.0 * SS * math.sin(age * 8.0 + track["phase"]),
                2.2 * SS * math.cos(age * 7.0 + track["phase"]),
            ]
        )
        center = spout + track["offset"] * travel + wobble * min(1.0, age * 2.0)
        r = track["radius"] * (0.70 + 0.30 * min(1.0, age * 2.2))
        sx = 1.0 + 0.15 * math.sin(frame_index * 0.7 + track["phase"])
        sy = 1.0 - 0.12 * math.sin(frame_index * 0.7 + track["phase"])
        points = ellipse_points(
            center[0],
            center[1],
            r * sx,
            r * sy,
            math.radians(12.0 * math.sin(track["phase"])),
            40,
        )
        draw.polygon(points, fill=255)
        centers.append((center, r))
    mask = mask.filter(ImageFilter.GaussianBlur(0.35 * SS))
    return mask, centers


def add_stream_highlights(image, stream_mask, geometry, frame_index):
    t = geometry["t"]
    p0 = np.array(geometry["spout"], dtype=np.float64)
    p1 = p0 + np.array([-22.0 * SS, 0.5 * SS])
    p2 = p0 + np.array([-56.0 * SS, 16.0 * SS])
    p3 = p0 + np.array([-84.0 * SS, 42.0 * SS])
    progress = 0.13 + 0.87 * ease_out(t)
    highlight = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))
    draw = ImageDraw.Draw(highlight, "RGBA")
    upper = []
    core = []
    for index in range(55):
        s = progress * index / 54.0
        inv = 1.0 - s
        point = (
            inv**3 * p0
            + 3.0 * inv**2 * s * p1
            + 3.0 * inv * s**2 * p2
            + s**3 * p3
        )
        tangent = (
            3.0 * inv**2 * (p1 - p0)
            + 6.0 * inv * s * (p2 - p1)
            + 3.0 * s**2 * (p3 - p2)
        )
        tangent /= max(1e-6, np.linalg.norm(tangent))
        normal = np.array([-tangent[1], tangent[0]])
        wave = 1.0 * SS * math.sin(17.0 * s - t * 3.2)
        width = (5.6 - 2.5 * s) * SS
        upper.append(
            tuple(
                point
                + normal * (wave + width * 0.42)
                + np.array([0.0, -0.5 * SS])
            )
        )
        core.append(
            tuple(point + normal * (wave - width * 0.06))
        )
    draw.line(upper, fill=(235, 250, 250, 112), width=max(1, int(0.65 * SS)))
    draw.line(core, fill=(173, 228, 232, 64), width=max(1, int(0.45 * SS)))
    highlight = highlight.filter(ImageFilter.GaussianBlur(0.55 * SS))
    alpha = ImageChops.multiply(
        highlight.getchannel("A"),
        stream_mask.filter(ImageFilter.MaxFilter(5)),
    )
    highlight.putalpha(alpha)
    return Image.alpha_composite(image.convert("RGBA"), highlight)


def add_droplet_highlights(image, droplet_mask, centers):
    highlight = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))
    draw = ImageDraw.Draw(highlight, "RGBA")
    for center, radius in centers:
        hx = center[0] - radius * 0.30
        hy = center[1] - radius * 0.34
        hr = max(0.55 * SS, radius * 0.22)
        draw.ellipse(
            (hx - hr, hy - hr, hx + hr, hy + hr),
            fill=(245, 253, 252, 180),
        )
        px = center[0] - radius * 0.08
        py = center[1] - radius * 0.52
        pr = max(0.4 * SS, radius * 0.08)
        draw.ellipse(
            (px - pr, py - pr, px + pr, py + pr),
            fill=(255, 255, 255, 220),
        )
    highlight = highlight.filter(ImageFilter.GaussianBlur(0.6 * SS))
    highlight.putalpha(
        ImageChops.multiply(
            highlight.getchannel("A"),
            droplet_mask.filter(ImageFilter.MaxFilter(3)),
        )
    )
    return Image.alpha_composite(image.convert("RGBA"), highlight)


def render_cup(base, geometry, frame_index):
    t = geometry["t"]
    angle = geometry["angle"]
    cx, cy = geometry["center"]
    rim_y = -geometry["half_height"]
    base_y = geometry["half_height"]
    water_y = geometry["water_y"]

    body_mask = Image.new("L", (SW, SH), 0)
    draw_mask = ImageDraw.Draw(body_mask)
    draw_mask.polygon(geometry["body"], fill=255)
    draw_mask.polygon(
        ellipse_points(
            *local_to_world(cx, cy, 0.0, rim_y, angle),
            geometry["rim_half"],
            5.5 * SS,
            angle,
        ),
        fill=255,
    )
    draw_mask.polygon(
        ellipse_points(
            *local_to_world(cx, cy, 0.0, base_y, angle),
            geometry["base_half"],
            4.0 * SS,
            angle,
        ),
        fill=255,
    )

    water_mask = Image.new("L", (SW, SH), 0)
    water_draw = ImageDraw.Draw(water_mask)
    water_draw.polygon(geometry["water"], fill=255)
    water_center = local_to_world(cx, cy, 0.0, water_y, angle)
    water_draw.polygon(
        ellipse_points(
            *water_center,
            geometry["water_half"],
            4.2 * SS,
            angle,
        ),
        fill=255,
    )
    water_mask = ImageChops.multiply(water_mask, body_mask)
    water_mask = water_mask.filter(ImageFilter.GaussianBlur(0.35 * SS))

    # Glass body: a faint blue-gray veil plus a stronger refractive edge.
    base = add_refracted_liquid(
        base,
        body_mask,
        tint=(95, 112, 116),
        opacity=0.18,
        time_value=t,
    )
    base = add_refracted_liquid(
        base,
        water_mask,
        tint=(49, 137, 154),
        opacity=0.72,
        time_value=t + 0.8,
    )

    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB").convert("RGBA")

    # Water color volume and darker lower meniscus.
    water_tint = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))
    water_draw = ImageDraw.Draw(water_tint, "RGBA")
    water_draw.bitmap((0, 0), water_mask, fill=(51, 155, 173, 38))
    lower_water = Image.new("L", (SW, SH), 0)
    lower_draw = ImageDraw.Draw(lower_water)
    lower_draw.polygon(geometry["water"], fill=255)
    lower_water = ImageChops.multiply(lower_water, water_mask)
    lower_tint = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))
    ImageDraw.Draw(lower_tint, "RGBA").bitmap(
        (0, 0), lower_water, fill=(11, 79, 100, 30)
    )
    image = Image.alpha_composite(image, water_tint)
    image = Image.alpha_composite(image, lower_tint)

    # The water surface catches a thin elliptical specular line.
    surface = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))
    surface_draw = ImageDraw.Draw(surface, "RGBA")
    surface_points = ellipse_points(
        *water_center,
        geometry["water_half"] * 0.92,
        4.1 * SS,
        angle,
        120,
    )
    surface_draw.line(
        surface_points[:62],
        fill=(205, 250, 250, 105),
        width=max(1, int(0.65 * SS)),
        joint="curve",
    )
    surface = surface.filter(ImageFilter.GaussianBlur(0.45 * SS))
    image = Image.alpha_composite(image, surface)

    # Cup silhouette and rim.
    body_edge = add_edge(body_mask, (176, 204, 204), 1 * SS, 92, 0.35 * SS)
    image = Image.alpha_composite(image, body_edge)

    rim_mask = Image.new("L", (SW, SH), 0)
    rim_draw = ImageDraw.Draw(rim_mask)
    rim_points = ellipse_points(
        *local_to_world(cx, cy, 0.0, rim_y, angle),
        geometry["rim_half"],
        5.5 * SS,
        angle,
    )
    rim_draw.polygon(rim_points, fill=255)
    inner_rim = ellipse_points(
        *local_to_world(cx, cy, 0.0, rim_y + 0.4 * SS, angle),
        geometry["rim_half"] - 2.2 * SS,
        3.7 * SS,
        angle,
    )
    rim_draw.polygon(inner_rim, fill=0)
    rim_layer = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))
    rim_layer.paste((218, 235, 232, 205), (0, 0), rim_mask)
    rim_layer = rim_layer.filter(ImageFilter.GaussianBlur(0.25 * SS))
    image = Image.alpha_composite(image, rim_layer)

    # A cool inner lip and warm reflected glint separate glass from water.
    glint = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))
    glint_draw = ImageDraw.Draw(glint, "RGBA")
    upper_rim = ellipse_points(
        *local_to_world(cx, cy, 0.0, rim_y - 0.1 * SS, angle),
        geometry["rim_half"] * 0.90,
        4.9 * SS,
        angle,
        120,
    )
    glint_draw.line(
        upper_rim[28:86],
        fill=(255, 252, 230, 190),
        width=max(1, int(0.65 * SS)),
        joint="curve",
    )
    glint = glint.filter(ImageFilter.GaussianBlur(0.65 * SS))
    image = Image.alpha_composite(image, glint)

    # Vertical specular streak and secondary caustic inside the cup.
    streak = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))
    streak_draw = ImageDraw.Draw(streak, "RGBA")
    for lx, lw, opacity in ((-18.0, 1.7, 108), (11.0, 0.9, 52)):
        points = [
            local_to_world(cx, cy, (lx - lw) * SS, -29.0 * SS, angle),
            local_to_world(cx, cy, (lx + lw) * SS, -29.0 * SS, angle),
            local_to_world(cx, cy, (lx + lw * 1.4) * SS, 27.0 * SS, angle),
            local_to_world(cx, cy, (lx - lw * 1.2) * SS, 27.0 * SS, angle),
        ]
        streak_draw.polygon(points, fill=(236, 250, 247, opacity))
    streak = streak.filter(ImageFilter.GaussianBlur(0.75 * SS))
    streak_alpha = ImageChops.multiply(streak.getchannel("A"), body_mask)
    streak.putalpha(streak_alpha)
    image = Image.alpha_composite(image, streak)

    # Base ellipse and a soft contact glint.
    base_rim = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))
    base_rim_draw = ImageDraw.Draw(base_rim, "RGBA")
    base_points = ellipse_points(
        *local_to_world(cx, cy, 0.0, base_y, angle),
        geometry["base_half"],
        4.0 * SS,
        angle,
    )
    base_rim_draw.line(
        base_points[12:126],
        fill=(211, 225, 220, 135),
        width=max(1, int(0.65 * SS)),
        joint="curve",
    )
    base_rim = base_rim.filter(ImageFilter.GaussianBlur(0.5 * SS))
    image = Image.alpha_composite(image, base_rim)

    # A handful of tiny suspended bubbles inside the water.
    bubbles = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))
    bubble_draw = ImageDraw.Draw(bubbles, "RGBA")
    bubble_specs = (
        (-9.0, 21.0, 1.15),
        (5.0, 11.0, 0.8),
        (-2.0, 30.0, 0.65),
        (12.0, 25.0, 0.55),
    )
    for index, (lx, ly, radius) in enumerate(bubble_specs):
        drift = 1.2 * math.sin(frame_index * 0.42 + index * 1.7)
        bx, by = local_to_world(cx, cy, (lx + drift) * SS, ly * SS, angle)
        br = radius * SS
        bubble_draw.ellipse(
            (bx - br, by - br, bx + br, by + br),
            outline=(218, 248, 248, 115),
            width=max(1, int(0.35 * SS)),
        )
        bubble_draw.ellipse(
            (bx - br * 0.45, by - br * 0.45, bx, by),
            fill=(250, 255, 255, 140),
        )
    bubbles.putalpha(
        ImageChops.multiply(bubbles.getchannel("A"), water_mask)
    )
    image = Image.alpha_composite(image, bubbles)
    return np.asarray(image.convert("RGB"), dtype=np.float32)


def add_environmental_glints(image, frame_index):
    t = frame_index / (FRAMES - 1)
    layer = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer, "RGBA")

    # Tiny specular sparks on droplets and the moving ribbon.
    for index in range(6):
        x = (64 + 21 * index + 4 * math.sin(index * 1.8 + t * 4.0)) * SS
        y = (58 + 8 * math.sin(index * 2.3 + t * 3.0)) * SS
        radius = (0.6 + 0.35 * (index % 3)) * SS
        alpha = int(48 + 30 * math.sin(index + t * 5.0) ** 2)
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(232, 251, 250, alpha),
        )
    layer = layer.filter(ImageFilter.GaussianBlur(1.0 * SS))
    return np.asarray(
        Image.alpha_composite(
            Image.fromarray(np.clip(image, 0, 255).astype(np.uint8), "RGB").convert(
                "RGBA"
            ),
            layer,
        ).convert("RGB"),
        dtype=np.float32,
    )


def postprocess(image, frame_index):
    arr = np.asarray(image, dtype=np.float32)
    small = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").resize(
        (W, H), Image.Resampling.LANCZOS
    )
    arr = np.asarray(small, dtype=np.float32)

    # Restrained bloom from the brightest liquid and window highlights.
    luminance = arr.max(axis=2)
    bright = np.clip((luminance - 178.0) / 77.0, 0.0, 1.0)
    glow_image = Image.fromarray(
        np.clip(arr * bright[..., None], 0, 255).astype(np.uint8), "RGB"
    )
    glow = glow_image.filter(ImageFilter.GaussianBlur(2.2))
    glow_arr = np.asarray(glow, dtype=np.float32)
    arr = 255.0 - (255.0 - arr) * (255.0 - glow_arr * 0.17) / 255.0

    yy, xx = np.mgrid[0:H, 0:W]
    nx = (xx - (W - 1) / 2.0) / ((W - 1) / 2.0)
    ny = (yy - (H - 1) / 2.0) / ((H - 1) / 2.0)
    vignette = np.clip(1.0 - 0.17 * (nx**2 * 0.7 + ny**2), 0.78, 1.0)
    arr *= vignette[..., None]

    # Warm highlight roll-off and cool shadows.
    arr[..., 0] = arr[..., 0] * 1.018 + 1.2
    arr[..., 2] = arr[..., 2] * 1.025 + 0.8
    arr = np.where(arr > 224.0, 224.0 + (arr - 224.0) * 0.74, arr)

    rng = np.random.default_rng(400 + frame_index)
    grain = rng.normal(0.0, 2.3, (H, W, 1))
    arr += grain
    return np.clip(arr, 0, 255).astype(np.uint8)


def render_frame(background, frame_index):
    geometry = make_cup_geometry(frame_index, FRAMES)
    base = background.copy()
    image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB").convert("RGBA")

    stream_mask = render_stream_mask(geometry, frame_index)
    stream_base = add_refracted_liquid(
        base,
        stream_mask,
        tint=(52, 151, 170),
        opacity=0.78,
        time_value=frame_index * 0.17,
    )
    image = Image.fromarray(
        np.clip(stream_base, 0, 255).astype(np.uint8), "RGB"
    ).convert("RGBA")
    stream_tint = Image.new("RGBA", (SW, SH), (0, 0, 0, 0))
    stream_tint_draw = ImageDraw.Draw(stream_tint, "RGBA")
    stream_tint_draw.bitmap((0, 0), stream_mask, fill=(45, 151, 171, 40))
    image = Image.alpha_composite(image, stream_tint)
    image = add_stream_highlights(image, stream_mask, geometry, frame_index)

    droplet_mask, droplet_centers = render_droplets(geometry, frame_index)
    droplet_base = add_refracted_liquid(
        np.asarray(image.convert("RGB"), dtype=np.float32),
        droplet_mask,
        tint=(57, 160, 180),
        opacity=0.84,
        time_value=frame_index * 0.24 + 2.0,
    )
    image = Image.fromarray(
        np.clip(droplet_base, 0, 255).astype(np.uint8), "RGB"
    ).convert("RGBA")
    droplet_ring = add_edge(droplet_mask, (192, 235, 238), 1 * SS, 115, 0.3 * SS)
    image = Image.alpha_composite(image, droplet_ring)
    image = add_droplet_highlights(image, droplet_mask, droplet_centers)

    cup_base = np.asarray(image.convert("RGB"), dtype=np.float32)
    cup_base = render_cup(cup_base, geometry, frame_index)
    cup_base = add_environmental_glints(cup_base, frame_index)
    return postprocess(cup_base, frame_index)


def main():
    background = build_background()
    frames = np.stack(
        [render_frame(background, index) for index in range(FRAMES)],
        axis=0,
    )
    artifact_io.save(frames, CASE_DIR)
    print(f"Wrote {FRAMES} frames to {CASE_DIR}")


if __name__ == "__main__":
    main()
