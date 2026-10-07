#!/bin/zsh
set -e
TASK_DIR='/Users/REVIEW_USER/FlyGarden/.runtime/file-tracker-diagnostics'
mkdir -p "$TASK_DIR"
STAMP=$(date '+%Y%m%d-%H%M%S')
REPORT="$TASK_DIR/$STAMP.txt"
print 'This collects memory diagnostics, then restarts only macOS file-change tracking.'
print 'It does not delete files, restart the Mac, or stop Fly Garden.'
print 'Your Mac administrator password is required; typing is hidden.'
sudo -v
TRACKER_PID=$(pgrep -x fseventsd | head -n 1)
{
 date
 ps -p "$TRACKER_PID" -o pid,%cpu,rss,vsz,etime,command
 sysctl vm.swapusage
 sudo vmmap -summary "$TRACKER_PID" || true
} > "$REPORT" 2>&1
print 'Restarting the file tracker...'
if ! sudo launchctl kickstart -k system/com.apple.fseventsd >> "$REPORT" 2>&1; then
 print 'macOS blocked the restart. The diagnostic is saved; no file cleanup was performed.'
 print "$REPORT"
 read 'REPLY?Press Return to close.'
 exit 1
fi
sleep 5
{
 print 'AFTER RESTART'
 date
 launchctl print system/com.apple.fseventsd
 sysctl vm.swapusage
 NEW_PID=$(pgrep -x fseventsd | head -n 1)
 sudo vmmap -summary "$NEW_PID" || true
} >> "$REPORT" 2>&1
print 'Restart completed. Recheck Activity Monitor in about one minute.'
print "Diagnostic saved to: $REPORT"
print 'Return to Codex and say “done” so I can verify the result and the experiment.'
read 'REPLY?Press Return to close.'
