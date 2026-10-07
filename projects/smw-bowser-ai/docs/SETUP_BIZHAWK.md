# BizHawk / EmuHawk Setup

Use the Windows EmuHawk build for the first implementation. The Lua bridge uses
BizHawk's `comm.socketServer*`, `joypad.set`, `mainmemory.read*`,
`gameinfo.getromhash`, and `emu.frameadvance` APIs.

## Checklist

1. Install BizHawk/EmuHawk on Windows.
2. Load a legal Super Mario World ROM.
3. Disable cheats.
4. Use a clean save for final validation.
5. Open **Tools -> Lua Console**.
6. Start the Python bridge server on the same machine or reachable LAN host.
7. Open `lua/smw_bridge.lua`.

## Final Evaluation Rules

- No savestate load/save after evaluation starts.
- No memory writes or RAM patching.
- No manual input after launch.
- Starworld entry aborts the run.
- Bowser must be defeated; credits warp without Bowser defeat is invalid.

## Port Direction

The Lua API calls this a "socket server", but from the Python side we listen on
the configured host/port and respond to each Lua frame message. Responses must
be prefixed as `LENGTH SPACE JSON`, matching BizHawk's response requirement.

