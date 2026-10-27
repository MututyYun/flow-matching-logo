"""把 cross.html 的场景代码 + Three.js + 数据打包成可双击打开的 standalone HTML。"""
import os
import re
import subprocess

os.chdir(os.path.dirname(os.path.abspath(__file__)))

html = open("cross.html", encoding="utf-8").read()
code = re.search(r'<script type="module">(.*?)</script>', html, re.S).group(1)
code = code.replace("from 'three'", "from './vendor/three.module.js'")
code = code.replace("from 'three/addons/", "from './vendor/jsm/")
open("cross_main.js", "w", encoding="utf-8").write(code)

subprocess.run(
    "npx esbuild cross_main.js --bundle --format=iife --minify "
    "--alias:three=./vendor/three.module.js --outfile=bundle.js",
    shell=True, check=True,
)

html = re.sub(r'<script src="\./cross_data\.js"></script>\s*', "", html)
html = re.sub(r'<script type="importmap">.*?</script>\s*', "", html, flags=re.S)
html = re.sub(r'<script type="module">.*?</script>\s*', "", html, flags=re.S)

data = open("cross_data.js", encoding="utf-8").read().replace("</script>", "<\\/script>")
bundle = open("bundle.js", encoding="utf-8").read().replace("</script>", "<\\/script>")
inject = "<script>" + data + "</script>\n<script>" + bundle + "</script>\n"
html = html.replace("</body>", inject + "</body>")

open("cross_standalone.html", "w", encoding="utf-8").write(html)
print("wrote cross_standalone.html", round(os.path.getsize("cross_standalone.html") / 1024), "KB")
