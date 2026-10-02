from pathlib import Path
from PIL import Image, ImageDraw
root = Path(__file__).resolve().parents[1]
out = root / 'experiments' / 'llm_style_transfer_styles_20261002'
render = out / 'rendered'
targets = [('bino1','Lamp_14912'),('bino2','Laptop_9878'),('bino3','Scissors_10429')]
styles = ['bino','spherical','rectangular','surround']
views = ['main','right','left','up','down']
# Main-view contact sheet: Bino plus three explicit style outputs.
thumb_w, thumb_h, label_h = 430, 300, 34
canvas = Image.new('RGB', (3*thumb_w, 4*(thumb_h+label_h)), 'white')
d = ImageDraw.Draw(canvas)
for r, style in enumerate(styles):
    for c, (tid, folder) in enumerate(targets):
        file = render / folder / style / 'main.png'
        im = Image.open(file).convert('RGB'); im.thumbnail((thumb_w-10, thumb_h-10))
        x, y = c*thumb_w, r*(thumb_h+label_h)
        canvas.paste(im, (x+(thumb_w-im.width)//2, y+(thumb_h-im.height)//2))
        d.text((x+8, y+thumb_h+8), style + ' ' + tid, fill='black')
canvas.save(out / 'comparison_all_styles_main_12.png')
# Per-target five-view contact sheets: Bino + three style outputs.
for tid, folder in targets:
    canvas = Image.new('RGB', (5*thumb_w, 4*(thumb_h+label_h)), 'white')
    d = ImageDraw.Draw(canvas)
    for r, style in enumerate(styles):
        for c, view in enumerate(views):
            file = render / folder / style / (view + '.png')
            im = Image.open(file).convert('RGB'); im.thumbnail((thumb_w-10, thumb_h-10))
            x, y = c*thumb_w, r*(thumb_h+label_h)
            canvas.paste(im, (x+(thumb_w-im.width)//2, y+(thumb_h-im.height)//2))
            d.text((x+8, y+thumb_h+8), style + ' ' + view, fill='black')
    canvas.save(out / (folder + '_all_styles_20views.png'))
print(out / 'comparison_all_styles_main_12.png')
