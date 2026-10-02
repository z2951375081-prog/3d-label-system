from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json, sys
root = Path(__file__).resolve().parents[1]
out = root / 'experiments' / 'llm_style_transfer_20261002'
render = out / 'rendered'
rows = json.loads((out / 'experiment_manifest.json').read_text(encoding='utf-8'))
items = []
for r in rows['manual']:
    sample = r['sample'].replace('/', '_')
    items.append((r['id'] + ' manual', root / r['main_image']))
for r in rows['bino']:
    sample = r['sample'].replace('/', '_')
    items.append((r['id'] + ' bino', render / (sample + '/bino/main.png')))
for r in rows['final']:
    sample = r['sample'].replace('/', '_')
    items.append((r['id'] + ' final', render / (sample + '/final/main.png')))
thumb_w, thumb_h = 430, 300
label_h = 34
cols = 3
rows_n = (len(items) + cols - 1) // cols
sheet = Image.new('RGB', (cols * thumb_w, rows_n * (thumb_h + label_h)), 'white')
draw = ImageDraw.Draw(sheet)
for i, (label, file) in enumerate(items):
    x = (i % cols) * thumb_w
    y = (i // cols) * (thumb_h + label_h)
    try:
        im = Image.open(file).convert('RGB')
        im.thumbnail((thumb_w - 10, thumb_h - 10))
        px = x + (thumb_w - im.width) // 2
        py = y + (thumb_h - im.height) // 2
        sheet.paste(im, (px, py))
    except Exception as e:
        draw.text((x + 8, y + 8), 'missing: ' + str(file), fill='red')
    draw.text((x + 8, y + thumb_h + 8), label, fill='black')
sheet.save(out / 'comparison_main_9.png')
# Per target: five-view 3-row comparison manual reference, bino, final.
for target in rows['bino']:
    sample = target['sample'].replace('/', '_')
    # Manual comparison is the target's own annotated reference view files.
    # Use existing dataset view paths from the manifest if present in the experiment manifest's source sample.
    # The main contact sheet above is the primary artifact; this section builds bino/final five-view contact.
    tiles = []
    for mode in ['bino', 'final']:
        for view in ['main', 'right', 'left', 'up', 'down']:
            file = render / (sample + '/' + mode + '/' + view + '.png')
            tiles.append((mode + ' ' + view, file))
    canvas = Image.new('RGB', (5 * 430, 2 * 334), 'white')
    d = ImageDraw.Draw(canvas)
    for j, (label, file) in enumerate(tiles):
        xx = (j % 5) * 430
        yy = (j // 5) * 334
        im = Image.open(file).convert('RGB')
        im.thumbnail((420, 300))
        canvas.paste(im, (xx + (430 - im.width)//2, yy + 4))
        d.text((xx + 8, yy + 306), label, fill='black')
    canvas.save(out / (sample + '_bino_final_10views.png'))
print(str(out / 'comparison_main_9.png'))
