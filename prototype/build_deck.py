"""把三段视频与校徽图内嵌进 deck_template.html，产出离线单文件幻灯片。"""
import base64
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "少走弯路_FlowMatching.html")


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


html = open(os.path.join(HERE, "deck_template.html"), encoding="utf-8").read()
html = html.replace("__LOGO__", "data:image/webp;base64," + b64(os.path.join(HERE, "logo_small.webp")))
html = html.replace("__VIDEO1__", b64(os.path.join(HERE, "videos", "cross.mp4")))
html = html.replace("__VIDEO2__", b64(os.path.join(HERE, "videos", "three_layer.mp4")))
html = html.replace("__VIDEO3__", b64(os.path.join(HERE, "videos", "color.mp4")))

repo_file = os.path.join(ROOT, "repo_url.txt")
repo = open(repo_file, encoding="utf-8").read().strip() if os.path.exists(repo_file) else ""
if not repo:
    repo = "https://github.com/your-name/flow-matching-logo"
repo_short = repo.split("://", 1)[-1].rstrip("/")
html = html.replace("__REPO_SHORT__", repo_short)
html = html.replace("__REPO__", repo)

with open(OUT, "w", encoding="utf-8") as f:
    f.write(html)

print(f"wrote {OUT}  ({os.path.getsize(OUT) / 1024 / 1024:.1f} MB)")
