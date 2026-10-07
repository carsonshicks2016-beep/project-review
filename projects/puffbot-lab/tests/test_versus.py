"""Me vs AI: the human's controller bindings (no emulator needed)."""
from __future__ import annotations

import configparser

import pytest

from puffbot import dolphin
from puffbot.config import Config

DEV = 'SDL/0/Xbox One S Controller'


def test_switch_layout_puts_buttons_where_a_switch_pro_has_them():
    b = dolphin.human_bindings(DEV, 'switch')
    assert b['Buttons/A'] == f'`Button A` | `{DEV}:Button E`'        # right face button
    assert b['Buttons/B'].endswith(':Button S`')                     # bottom
    assert b['Buttons/X'].endswith(':Button N`') and b['Buttons/Y'].endswith(':Button W`')
    x = dolphin.human_bindings(DEV, 'xbox')
    assert x['Buttons/A'].endswith(':Button S`') and x['Buttons/B'].endswith(':Button E`')
    with pytest.raises(ValueError):
        dolphin.human_bindings(DEV, 'ps5')


def test_every_binding_keeps_the_pipe_so_the_harness_can_pick_characters():
    for key, expr in dolphin.human_bindings(DEV, 'switch').items():
        first = expr.split(' | ')[0]
        assert first.startswith('`') and 'SDL' not in first, key
        assert DEV in expr, key
    b = dolphin.human_bindings(DEV, 'switch')
    assert b['Main Stick/Up'].endswith(':Left Y+`') and b['C-Stick/Up'].endswith(':Right Y+`') and b['Triggers/L-Analog'].endswith(':Trigger L`')
    assert b['Buttons/Z'].count(DEV) == 2                             # either bumper


def test_bind_human_rewrites_port_1_only(tmp_path):
    cfg = Config(iso='/dev/null')
    emu = dolphin.Emulator(cfg, 0, tmp_path / 'User', visible=True, human={'device': DEV, 'layout': 'switch'})
    (tmp_path / 'User' / 'Config').mkdir(parents=True)
    ini = configparser.ConfigParser()
    for port in (1, 2):
        ini.add_section(f'GCPad{port}')
        ini.set(f'GCPad{port}', 'device', f'Pipe/0/slippibot{port}')
        ini.set(f'GCPad{port}', 'buttons/a', 'Button A')
    with open(tmp_path / 'User' / 'Config' / 'GCPadNew.ini', 'w') as f:
        ini.write(f)
    emu._bind_human()
    out = configparser.ConfigParser()
    out.read(tmp_path / 'User' / 'Config' / 'GCPadNew.ini')
    assert out['GCPad1']['buttons/a'] == f'`Button A` | `{DEV}:Button E`'
    assert out['GCPad1']['device'] == 'Pipe/0/slippibot1'
    assert out['GCPad2']['buttons/a'] == 'Button A'
    raw = (tmp_path / 'User' / 'Config' / 'GCPadNew.ini').read_text()
    assert 'Buttons/A' not in raw, 'no second, differently-cased copy of a key'
