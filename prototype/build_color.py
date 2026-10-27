"""把 color.html 打包成可双击打开的 standalone HTML。"""
import os
import re
import subprocess

os.chdir(os.path.dirname(os.path.abspath(__file__)))

html = open("color.html", encoding="utf-8").read()
code = re.search(r'<script type="module">(.*?)</script>', html, re.S).group(1)
code = code.replace("from 'three'", "from './vendor/three.module.js'")
code = code.replace("from 'three/addons/", "from './vendor/jsm/")
open("color_main.js", "w", encoding="utf-8").write(code)

subprocess.run(
    "npx esbuild color_main.js --bundle --format=iife --minify "
    "--alias:three=./vendor/three.module.js --outfile=bundle_color.js",
    shell=True, check=True,
)

html = re.sub(r'<script src="\./color_data\.js"></script>\s*', "", html)
html = re.sub(r'<script type="importmap">.*?</script>\s*', "", html, flags=re.S)
html = re.sub(r'<script type="module">.*?</script>\s*', "", html, flags=re.S)

data = open("color_data.js", encoding="utf-8").read().replace("</script>", "<\\/script>")
bundle = open("bundle_color.js", encoding="utf-8").read().replace("</script>", "<\\/script>")
inject = "<script>" + data + "</script>\n<script>" + bundle + "</script>\n"
html = html.replace("</body>", inject + "</body>")

open("color_standalone.html", "w", encoding="utf-8").write(html)
print("wrote color_standalone.html", round(os.path.getsize("color_standalone.html") / 1024), "KB")
