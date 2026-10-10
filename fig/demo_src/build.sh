#!/usr/bin/env bash
# Rebuilds fig/r2cc-demo.mp4 and fig/r2cc-demo.gif from template.html (storyboard, captions, layout) and data.json.
#   ./build.sh            render the video and the GIF
#   ./build.sh preview    only write demo.html, which plays in a browser with a scrubber
# Needs Node.js, ffmpeg and Google Chrome. data.json is made by data.py from the raw samples of the run.
set -euo pipefail
cd "$(dirname "$0")"
python3 -c 'import sys; t = open("template.html").read(); open("demo.html", "w").write(t.replace("/*DATA*/", open("data.json").read()))'
[[ "${1:-}" == preview ]] && { echo "wrote demo.html"; exit 0; }
[[ -d node_modules/playwright-core ]] || npm install --silent
rm -rf frames && node render.mjs 30 0 "" frames
ffmpeg -y -loglevel error -framerate 30 -i frames/%05d.png -c:v libx264 -preset slow -crf 24 -pix_fmt yuv420p \
  -movflags +faststart -tune animation ../r2cc-demo.mp4
ffmpeg -y -loglevel error -framerate 30 -i frames/%05d.png -vf \
  "fps=15,scale=1280:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle" \
  ../r2cc-demo.gif
rm -rf frames
ls -la ../r2cc-demo.mp4 ../r2cc-demo.gif
