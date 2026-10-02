import argparse
import json
import math
import os
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
IMAGE_WIDTH = 750
IMAGE_HEIGHT = 500
FOCAL_LENGTH_MM = 50
SENSOR_WIDTH_MM = 36
SENSOR_HEIGHT_MM = 24
CAMERA_DISTANCE = 10
PERTURB_DEGREES = 45
MAIN_DIRECTION = (1.0, 1.0, 1.0)
VIEW_ORDER = ['main', 'right', 'left', 'up', 'down']
VIEW_COLORS = {
    'baseline': (20, 112, 196),
    'increase_label_gap': (34, 139, 34),
    'long_leader_arc': (185, 92, 20),
    'short_leader_arc': (132, 72, 180),
    'arc_uniformity': (215, 50, 80),
}
PRESENTATION_RE = re.compile(r'^(label_|leader_)', re.I)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', default=str(ROOT / 'experiments' / 'moe_layout_llm_candidates.json'))
    p.add_argument('--outputDir', default=str(ROOT / 'experiments' / 'moe_layout_llm_candidate_previews'))
    return p.parse_args()


def add(a, b): return tuple(a[i] + b[i] for i in range(3))
def sub(a, b): return tuple(a[i] - b[i] for i in range(3))
def scale(a, v): return tuple(a[i] * v for i in range(3))
def dot(a, b): return sum(a[i] * b[i] for i in range(3))
def cross(a, b): return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

def norm(a, fallback=(1.0, 0.0, 0.0)):
    length = math.sqrt(max(0.0, dot(a, a)))
    return scale(a, 1 / length) if length > 1e-9 else tuple(fallback)


def frame_from_direction(direction, up_hint=(0.0, 1.0, 0.0)):
    outward = norm(direction)
    forward = scale(outward, -1)
    right = norm(cross(forward, norm(up_hint)), (1.0, 0.0, 0.0))
    up = norm(cross(right, forward), (0.0, 1.0, 0.0))
    return {'outward': outward, 'forward': forward, 'right': right, 'up': up}


def rotate_toward(direction, screen_axis, degrees):
    radians = degrees * math.pi / 180
    return norm(add(scale(norm(direction), math.cos(radians)), scale(norm(screen_axis), math.sin(radians))))


MAIN_FRAME = frame_from_direction(MAIN_DIRECTION)
VIEW_DIRECTIONS = {
    'main': MAIN_FRAME['outward'],
    'right': rotate_toward(MAIN_FRAME['outward'], MAIN_FRAME['right'], PERTURB_DEGREES),
    'left': rotate_toward(MAIN_FRAME['outward'], scale(MAIN_FRAME['right'], -1), PERTURB_DEGREES),
    'up': rotate_toward(MAIN_FRAME['outward'], MAIN_FRAME['up'], PERTURB_DEGREES),
    'down': rotate_toward(MAIN_FRAME['outward'], scale(MAIN_FRAME['up'], -1), PERTURB_DEGREES),
}


def camera_for_bounds(bounds, view):
    target = tuple(bounds.get('center') or [0, 0, 0])
    frame = frame_from_direction(VIEW_DIRECTIONS.get(view, VIEW_DIRECTIONS['main']), MAIN_FRAME['up'])
    eye = add(target, scale(frame['outward'], CAMERA_DISTANCE))
    return {**frame, 'eye': eye, 'target': target}


def project_point(point, camera):
    rel = sub(tuple(point), camera['eye'])
    depth = max(1e-6, dot(rel, camera['forward']))
    x = dot(rel, camera['right']) / depth * FOCAL_LENGTH_MM / (SENSOR_WIDTH_MM * 0.5)
    y = dot(rel, camera['up']) / depth * FOCAL_LENGTH_MM / (SENSOR_HEIGHT_MM * 0.5)
    return ((x + 1) * 0.5 * (IMAGE_WIDTH - 1), (1 - (y + 1) * 0.5) * (IMAGE_HEIGHT - 1), depth)


def parse_clean_faces(obj_path):
    vertices = []
    faces = []
    skip_group = False
    skip_material = False
    try:
        lines = obj_path.read_text(encoding='utf-8', errors='ignore').splitlines()
    except FileNotFoundError:
        return vertices, faces
    for line in lines:
        parts = line.strip().split()
        if not parts or parts[0].startswith('#'):
            continue
        if parts[0] == 'v' and len(parts) >= 4:
            try:
                vertices.append(tuple(float(v) for v in parts[1:4]))
            except ValueError:
                pass
            continue
        if parts[0] in ('g', 'o'):
            skip_group = any(PRESENTATION_RE.match(name or '') for name in parts[1:])
            skip_material = False
            continue
        if parts[0] == 'usemtl':
            skip_material = bool(parts[1:] and PRESENTATION_RE.match(parts[1]))
            continue
        if parts[0] == 'f' and len(parts) >= 4 and not skip_group and not skip_material:
            indices = []
            for token in parts[1:]:
                try:
                    raw = int(token.split('/')[0])
                except ValueError:
                    continue
                idx = raw - 1 if raw > 0 else len(vertices) + raw
                if 0 <= idx < len(vertices):
                    indices.append(idx)
            if len(indices) >= 3:
                faces.append(indices)
    return vertices, faces


def render_clean_object(source_obj, bounds, view):
    image = Image.new('RGBA', (IMAGE_WIDTH, IMAGE_HEIGHT), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    vertices, faces = parse_clean_faces(ROOT / source_obj.replace('/', os.sep))
    camera = camera_for_bounds(bounds, view)
    projected = [project_point(v, camera) for v in vertices]
    polygons = []
    light = norm((0.4, 0.7, 1.0))
    for face in faces:
        pts3 = [vertices[i] for i in face]
        pts2 = [(projected[i][0], projected[i][1]) for i in face]
        if len(pts2) < 3:
            continue
        a, b, c = pts3[0], pts3[1], pts3[2]
        normal = norm(cross(sub(b, a), sub(c, a)), (0.0, 0.0, 1.0))
        shade = int(178 + 52 * max(0.0, dot(normal, light)))
        avg_depth = sum(projected[i][2] for i in face) / len(face)
        polygons.append((avg_depth, pts2, shade))
    for _, pts2, shade in sorted(polygons, key=lambda item: item[0], reverse=True):
        draw.polygon(pts2, fill=(shade, shade, shade, 255), outline=(132, 140, 148, 110))
    return image


def px(point, width, height):
    return ((point['x'] + 1) * 0.5 * width, (1 - point['y']) * 0.5 * height)


def clamp_box(x0, y0, x1, y1, width, height):
    return (max(0, min(width - 1, x0)), max(0, min(height - 1, y0)), max(0, min(width - 1, x1)), max(0, min(height - 1, y1)))


def draw_candidate(base_image, labels, mode, out_path):
    image = base_image.convert('RGBA')
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    color = VIEW_COLORS.get(mode, (30, 90, 180))
    try:
        font = ImageFont.truetype('arial.ttf', 14)
        small = ImageFont.truetype('arial.ttf', 11)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    for label in labels:
        ax, ay = px(label['anchor'], width, height)
        cx, cy = px(label['center'], width, height)
        hw = max(8, label['width'] * width / 2)
        hh = max(7, label['height'] * height / 2)
        x0, y0, x1, y1 = clamp_box(cx - hw, cy - hh, cx + hw, cy + hh, width, height)
        draw.line((ax, ay, cx, cy), fill=color + (230,), width=2)
        draw.ellipse((ax - 3, ay - 3, ax + 3, ay + 3), fill=(255, 220, 40, 255), outline=(60, 60, 60, 255))
        draw.rounded_rectangle((x0, y0, x1, y1), radius=5, fill=(255, 255, 255, 218), outline=color + (255,), width=2)
        text = str(label.get('text') or label.get('id') or '')[:28]
        draw.text((x0 + 4, y0 + 3), text, fill=(10, 20, 30, 255), font=font)
    draw.rounded_rectangle((8, 8, 310, 34), radius=4, fill=(255, 255, 255, 224), outline=color + (255,), width=2)
    draw.text((16, 14), f'Clean OBJ + candidate: {mode}', fill=color + (255,), font=small)
    composed = Image.alpha_composite(image, overlay).convert('RGB')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    composed.save(out_path, 'PNG')


def main():
    args = parse_args()
    data = json.loads(Path(args.input).read_text(encoding='utf-8'))
    out_dir = Path(args.outputDir).resolve()
    previews = []
    for row in data['rows']:
        sample_dir = out_dir / f"{row['category']}_{row['sample_id']}"
        clean_bases = {view: render_clean_object(row['source_obj'], row.get('bounds') or {}, view) for view in data['image_order']}
        for candidate in row['candidates']:
            candidate_views = {}
            for view in data['image_order']:
                out_path = sample_dir / candidate['mode'] / f"{view}.png"
                draw_candidate(clean_bases[view], candidate['projected'][view], candidate['mode'], out_path)
                candidate_views[view] = str(out_path.relative_to(ROOT)).replace(os.sep, '/')
            previews.append({
                'category': row['category'],
                'sample_id': row['sample_id'],
                'candidate_id': candidate['candidate_id'],
                'mode': candidate['mode'],
                'style': candidate.get('style'),
                'metrics': candidate.get('metrics'),
                'clean_preview_base': 'source_obj_without_label_leader_layers_or_anchor_region_colors',
                'views': candidate_views,
            })
    manifest = {
        'version': 'moe_layout_llm_candidate_preview_manifest_v2_clean_obj',
        'source': str(Path(args.input).resolve().relative_to(ROOT)).replace(os.sep, '/'),
        'image_order': data['image_order'],
        'base_policy': 'render neutral clean OBJ first, then draw candidate labels; do not use dataset PNGs because they contain manual labels and leader lines',
        'count': len(previews),
        'previews': previews,
    }
    manifest_path = out_dir / 'manifest.json'
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'ok': True, 'manifest': str(manifest_path.relative_to(ROOT)).replace(os.sep, '/'), 'previews': len(previews)}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
