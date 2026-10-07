"""Where each emulator's window opens.

Dolphin persists render-window geometry in Dolphin.ini, so writing it before launch
pins every slot to a known place instead of letting six windows stack on each other.
The generated grid is saved to window-layout.json and is meant to be hand-edited.
"""
import configparser
import json
from pathlib import Path
from .config import ROOT

LAYOUT = ROOT / 'window-layout.json'
SCREEN = (1512, 982)      # logical points of the built-in 3024x1964 Retina display
MENU_BAR = 32             # leave the macOS menu bar clear
TITLE_BAR = 28            # Dolphin's own title bar sits above the render area
MARGIN = 4
GAP = 8


def grid(count, screen=SCREEN, columns=None, reserve=0):
    """A columns x rows grid of 4:3 render areas that fits on one screen.

    `reserve` keeps that many points free at the bottom for the companion window. The
    emulators shrink to make room rather than the companion overlapping them.
    """
    count = max(1, int(count))
    if columns:
        cols = columns
        rows = -(-count // cols)
    elif count == 1:
        cols, rows = 1, 1
    elif count == 2:
        cols, rows = 2, 1
    elif count == 3:
        cols, rows = 3, 1
    elif count == 4:
        cols, rows = 2, 2
    elif count in (5, 6):
        cols, rows = 3, 2
    elif count in (7, 8):
        cols, rows = 4, 2
    elif count in (9, 10, 11, 12):
        cols, rows = 4, 3
    else:
        cols = 4
        rows = -(-count // cols)

    width = (screen[0] - 2*MARGIN - GAP*(cols-1)) // cols
    height = width * 3 // 4
    # Shrink to fit vertically if the rows would run off the bottom.
    available = screen[1] - MENU_BAR - MARGIN - GAP*(rows-1) - TITLE_BAR*rows - max(0, int(reserve))
    if rows*height > available:
        height = available // rows
        width = height * 4 // 3
    boxes = []
    for index in range(count):
        row = index // cols
        items_in_row = min(cols, count - row * cols)
        row_col = index % cols

        # Center the row horizontally if it has fewer windows than cols
        row_width = items_in_row * width + (items_in_row - 1) * GAP
        start_x = (screen[0] - row_width) // 2

        x = start_x + row_col * (width + GAP)
        y = MENU_BAR + row * (height + TITLE_BAR + GAP)
        boxes.append({'x': x, 'y': y, 'width': width, 'height': height})
    return boxes


def load(count, path=None, reserve=0):
    """Saved layout for this emulator count, generating and storing one if absent.

    Layouts that reserve a companion band are stored under their own key, so turning
    the companion on never rewrites a plain layout somebody hand-edited.
    """
    path = Path(path or LAYOUT)
    try: saved = json.loads(path.read_text())
    except (OSError, ValueError): saved = {}
    key = f'{int(count)}+panel' if reserve else str(int(count))
    boxes = saved.get(key)
    if not isinstance(boxes, list) or len(boxes) != count:
        boxes = grid(count, reserve=reserve)
        saved[key] = boxes
        try: path.write_text(json.dumps(saved, indent=2, sort_keys=True))
        except OSError: pass
    return boxes


def write_geometry(ini_path, box):
    """Pin one emulator's render window. Dolphin reads these from [Display] and [Interface]."""
    if not box: return
    path = Path(ini_path)
    parser = configparser.ConfigParser()
    parser.read(path)
    if not parser.has_section('Display'): parser.add_section('Display')
    parser.set('Display', 'RenderWindowXPos', str(int(box['x'])))
    parser.set('Display', 'RenderWindowYPos', str(int(box['y'])))
    parser.set('Display', 'RenderWindowWidth', str(int(box['width'])))
    parser.set('Display', 'RenderWindowHeight', str(int(box['height'])))
    parser.set('Display', 'RenderWindowAutoSize', 'False')

    if not parser.has_section('Interface'): parser.add_section('Interface')
    parser.set('Interface', 'MainWindowPosX', str(int(box['x'])))
    parser.set('Interface', 'MainWindowPosY', str(int(box['y'])))
    parser.set('Interface', 'MainWindowWidth', str(int(box['width'])))
    parser.set('Interface', 'MainWindowHeight', str(int(box['height'])))
    with path.open('w') as handle: parser.write(handle)


def visualizer_space(boxes, screen=SCREEN, minimum=(420, 180)):
    """Largest free rectangle, including Dolphin title bars and a small gutter.

    Return None when the saved layout leaves no usable space. Never move the
    user's emulator boxes just to accommodate the companion window.
    """
    left, top, right, bottom = MARGIN, MENU_BAR, screen[0]-MARGIN, screen[1]-MARGIN
    occupied = []
    for b in boxes:
        x1, y1 = max(left, b['x']-GAP), max(top, b['y']-GAP)
        x2 = min(right, b['x']+b['width']+GAP)
        y2 = min(bottom, b['y']+b['height']+TITLE_BAR+GAP)
        if x2 > x1 and y2 > y1: occupied.append((x1, y1, x2, y2))
    xs = sorted({left, right, *(v for b in occupied for v in (b[0], b[2]))})
    ys = sorted({top, bottom, *(v for b in occupied for v in (b[1], b[3]))})
    best, area = None, 0
    for x1 in xs:
        for x2 in xs:
            if x2-x1 < minimum[0]: continue
            for y1 in ys:
                for y2 in ys:
                    size = (x2-x1)*(y2-y1)
                    if y2-y1 < minimum[1] or size <= area: continue
                    if any(x1 < b[2] and x2 > b[0] and y1 < b[3] and y2 > b[1] for b in occupied): continue
                    best, area = dict(x=x1, y=y1, width=x2-x1, height=y2-y1), size
    return best
