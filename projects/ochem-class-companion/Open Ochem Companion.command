#!/bin/zsh
cd '/Users/REVIEW_USER/Documents/Ochem Class Companion'
if /usr/bin/curl -fsS http://127.0.0.1:8769/api/state >/dev/null 2>&1; then
  open http://127.0.0.1:8769
else
  (sleep 1; open http://127.0.0.1:8769) &
  /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 server.py
fi
