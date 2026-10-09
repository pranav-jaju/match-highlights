#!/bin/zsh
# Process every manifest match whose video is present: labels (auto) + features.
cd /Users/macbook/ResumeSelfProject
for m in match1 match2 match3; do
  [ -f cache/$m.emb.npz ] || .venv/bin/python -m matchreel features data/$m.m4a --no-commentary 2>&1 | grep -E "^\[|Error|Traceback"
done
while true; do
  todo=()
  for m in m04 m05 m06 m07 m08 m09 m10 m11 m12 m13 m14 m15 m16 m17; do
    [ -f data/${m}_video.mp4 ] && [ ! -f cache/$m.emb.npz -o ! -f cache/$m.done ] && todo+=$m
  done
  [ ${#todo} -eq 0 ] && { [ -f /tmp/dl_done ] && break; sleep 60; continue; }
  m=${todo[1]}
  echo "== $m $(date +%H:%M)"
  .venv/bin/python tools/process_match.py $m 2>&1 | grep -E "^\{|Error|Traceback" && touch cache/$m.done
done
echo ALL_DONE
