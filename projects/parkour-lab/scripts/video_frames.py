"""Decode an MP4 and make an evenly spaced contact sheet for visual QA."""
import argparse
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image,ImageDraw
p=argparse.ArgumentParser();p.add_argument('video',type=Path);a=p.parse_args()
reader=imageio.get_reader(a.video)
n=reader.count_frames();meta=reader.get_meta_data()
sheet=Image.new('RGB',(960,540),'white');draw=ImageDraw.Draw(sheet)
for i in range(6):
    index=round(i*(n-1)/5)
    frame=Image.fromarray(reader.get_data(index));frame.thumbnail((320,240))
    x=(i%3)*320;y=(i//3)*270
    sheet.paste(frame,(x,y));draw.text((x+8,y+242),f'{index/meta["fps"]:.2f} s',fill='black')
reader.close();sheet.save(a.video.with_suffix('.png'))
print(dict(frames=n,fps=meta['fps'],seconds=n/meta['fps'],size=meta['size']))
