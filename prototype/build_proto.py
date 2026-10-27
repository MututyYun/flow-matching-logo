"""把 proto.html 打包成可双击打开的 standalone HTML（内联 Three.js / GSAP / 数据）。"""
import os
import re
import subprocess

os.chdir(os.path.dirname(os.path.abspath(__file__)))

html = open("proto.html", encoding="utf-8").read()
code = re.search(r'<script type="module">(.*?)</script>', html, re.S).group(1)
code = code.replace("from 'three'", "from './vendor/three.module.js'")
code = code.replace("from 'three/addons/", "from './vendor/jsm/")
open("proto_main.js", "w", encoding="utf-8").write(code)

subprocess.run(
    "npx esbuild proto_main.js --bundle --format=iife --minify "
    "--alias:three=./vendor/three.module.js --outfile=bundle_proto.js",
    shell=True, check=True,
)

html = re.sub(r'<script src="\./vendor/gsap\.min\.js"></script>\s*', "", html)
html = re.sub(r'<script src="\./data\.js"></script>\s*', "", html)
html = re.sub(r'<script type="importmap">.*?</script>\s*', "", html, flags=re.S)
html = re.sub(r'<script type="module">.*?</script>\s*', "", html, flags=re.S)

gsap = open("vendor/gsap.min.js", encoding="utf-8").read().replace("</script>", "<\\/script>")
data = open("data.js", encoding="utf-8").read().replace("</script>", "<\\/script>")
bundle = open("bundle_proto.js", encoding="utf-8").read().replace("</script>", "<\\/script>")
inject = ("<script>" + gsap + "</script>\n"
          "<script>" + data + "</script>\n"
          "<script>" + bundle + "</script>\n")
html = html.replace("</body>", inject + "</body>")

open("proto_standalone.html", "w", encoding="utf-8").write(html)
print("wrote proto_standalone.html", round(os.path.getsize("proto_standalone.html") / 1024 / 1024, 2), "MB")
