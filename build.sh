#!/bin/sh
# Assemble the page from parts/: lab.html is the artifact source (no doctype), index.html is the standalone page.
cd "$(dirname "$0")"
cat parts/1-head.html parts/2-body.html parts/3-core.js parts/4-l23.js parts/5-l4a.js parts/6-l4b.js parts/7-l5.js > lab.html
{ printf '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
  awk '/^<\/style>/{print; print "</head>\n<body>"; next} {print}' lab.html
  printf '</body>\n</html>\n'; } > index.html
