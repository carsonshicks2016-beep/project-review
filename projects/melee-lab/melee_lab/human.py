"""Local human input. Server timestamps and a short lease prevent stuck controls."""
import time
from pathlib import Path
import melee
from .catalog import read_json

BUTTONS = ('A', 'B', 'X', 'Y', 'Z', 'L', 'R', 'START', 'D_UP', 'D_DOWN', 'D_LEFT', 'D_RIGHT')
INPUT_TIMEOUT = .35

class HumanInput:
    def __init__(self, path):
        self.path = Path(path)
        self.connected = False

    def apply(self, controller):
        data = read_json(self.path)
        age = time.time() - data.get('received_at', 0)
        self.connected = 0 <= age < INPUT_TIMEOUT
        controller.release_all()
        if not self.connected:
            return
        for name in data.get('buttons', []):
            if name in BUTTONS:
                controller.press_button(melee.Button['BUTTON_' + name])
        for field, button in [('main', melee.Button.BUTTON_MAIN), ('c', melee.Button.BUTTON_C)]:
            x, y = data.get(field, [.5, .5])
            controller.tilt_analog(button, max(0, min(1, x)), max(0, min(1, y)))
        for name, value in zip(('L', 'R'), data.get('triggers', [0, 0])):
            controller.press_shoulder(melee.Button['BUTTON_' + name], value)
