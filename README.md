# 少走弯路：用 Flow Matching + 一次 Reflow，把扩散采样压到 1 步

用 **Rectified Flow / Flow Matching** 把 8000 个二维高斯噪声点映射成湘潭大学校徽中央“三拱门”点云，再用**一次 Reflow** 消除随机配对造成的轨迹交叉，让模型 **1 步**就能生成目标形状。

配套的 11 页离线幻灯片（含三段 3D 演示视频）可直接双击播放，不需要联网、Python 或服务器。

仓库地址：<https://github.com/MututyYun/flow-matching-logo>

---

## 一眼看结果

| 方法 | 1 步 | 2 步 | 4 步 | 10 步 | 50 步 |
|---|---:|---:|---:|---:|---:|
| 1-RF（随机配对） | 0.5742 | 0.1722 | 0.0580 | 0.0279 | 0.0151 |
| **2-RF（Reflow 一次）** | **0.0173** | 0.0160 | 0.0155 | 0.0152 | 0.0151 |

> 指标为 Chamfer Distance（越小越好，50 步结果作为参照）。

| 指标 | 1-RF | 2-RF |
|---|---:|---:|
| 随机配对交叉率 | 21.43% | **0.25%** |
| 平均多绕路比例 | 58.7% | **0.0079%** |

模型：`VelocityMLP(hidden=320, depth=5)`，**433,282 参数**；输入 `(x, t)`，输出二维速度场。

训练配置：

```
n_points      = 8000
n_pairs       = 32000
steps1/steps2 = 12000 / 12000
batch         = 8192
lr            = 2e-3
```

---

## 直接看幻灯片

双击根目录的 **`少走弯路_FlowMatching.html`**（约 32.5 MB）。

- **完全离线**：三段 MP4、校徽图、字体与全部脚本都已内嵌在单个 HTML 里，不请求任何外部资源。
- **环境要求**：Windows / macOS / Linux 上的 Chrome 或 Edge（需支持 H.264 视频解码）。教室 Windows 10/11 自带的 Edge 即可，**不需要 WebGL**。
- **操作**：`←` `→` / 空格翻页，`F` 全屏，`N` 显示演讲者备注，右下角可重播视频。
- 复制到 U 盘或直接发到教室电脑，双击打开即可。

---

## 仓库结构

```
logo_diffusion_demo/
├── 少走弯路_FlowMatching.html   # 最终交付：离线单文件幻灯片
├── README.md
├── HANDOFF.md                    # 项目交接记录（设计决策、坑、复现细节）
├── repo_url.txt                  # 幻灯片末页显示的仓库地址，换成你的即可
├── target/                       # 校徽原图与抠图
│   ├── target.jpg
│   ├── target_rgba.png
│   └── target_rgba_full.png
├── data/
│   ├── target_points.npz         # 8000 个目标点（含亮度/饱和度）
│   ├── rf_pairs.npz              # 32000 组噪声→目标配对
│   ├── rf_trajectory.npz         # 1-RF / 2-RF 轨迹与少步采样结果
│   └── rf_stats.json             # 交叉率、曲率、Chamfer、训练曲线
├── checkpoints/
│   ├── rf1.pt                    # 1-RF 权重
│   └── rf2.pt                    # 2-RF（Reflow 后）权重
├── scripts/
│   ├── prepare.py                # 抠图 + 三拱门掩码 + 采样 8000 点
│   ├── rf_model.py               # VelocityMLP
│   ├── train_rf.py               # 训练 1-RF → 生成配对 → Reflow → 2-RF
│   └── plot_rf_validate.py       # 只重画验证图
├── docs/
│   ├── 教案文案.md                # 课堂讲稿草稿
│   └── preview/                  # 数据与训练预览图
├── prototype/                    # Three.js 视频场景与幻灯片构建工具
│   ├── cross.html / proto.html / color.html
│   ├── export_*.py               # npz → 前端数据
│   ├── build_*.py                # 打包 standalone 场景 / 幻灯片
│   ├── export_*_video.js         # 逐帧截图（puppeteer-core + Edge）
│   ├── shot_deck.js              # 幻灯片逐页截图
│   ├── deck_template.html        # 幻灯片模板
│   ├── build_deck.py             # 注入视频/图片 → 单文件 HTML
│   ├── videos/                   # 三段已合成 MP4
│   ├── vendor/                   # Three.js / GSAP 本地副本
│   └── logo_small.webp
└── ddpm/                         # 早期 DDPM 方案备份（未用于最终交付）
```

---

## 复现

### 1. 环境

```bash
pip install torch numpy matplotlib opencv-python pillow
# 首次 prepare 若没有 target_rgba_full.png，会调用 rembg 抠图：
pip install rembg
```

训练自动使用 CUDA，无 GPU 时回退 CPU（会慢很多）。

### 2. 数据与模型

```bash
python scripts/prepare.py
python scripts/train_rf.py --steps1 12000 --steps2 12000 --reflow-pairs 32000
```

产物：`data/*.npz`、`data/rf_stats.json`、`checkpoints/rf1.pt`、`checkpoints/rf2.pt`、`docs/preview/*.png`。

只重画验证图：

```bash
python scripts/plot_rf_validate.py
```

### 3. 重做演示视频（可选）

需要 Node.js、Edge 和 ffmpeg：

```bash
cd prototype
npm install

# 数据：npz → 浏览器可读的 JS
python export_cross.py
python export_proto.py
python export_color.py

# 打包成可双击的 standalone 场景
python build_cross.py
python build_proto.py
python build_color.py

# 逐帧截图（输出 frames_cross/ frames_proto/ frames_color/）
node export_video.js
node export_proto_video.js
node export_color_video.js

# 合成 MP4
ffmpeg -framerate 30 -i frames_cross/f%04d.png -c:v libx264 -pix_fmt yuv420p -crf 20 videos/cross.mp4
ffmpeg -framerate 30 -i frames_proto/f%04d.png -c:v libx264 -pix_fmt yuv420p -crf 20 videos/three_layer.mp4
ffmpeg -framerate 30 -i frames_color/f%04d.png -c:v libx264 -pix_fmt yuv420p -crf 20 videos/color.mp4
```

### 4. 重新生成幻灯片

```bash
cd prototype
python build_deck.py
```

`build_deck.py` 会读取 `prototype/videos/` 里的三段 MP4、`logo_small.webp` 和根目录的 `repo_url.txt`，把所有资源 base64 内嵌，输出根目录的 `少走弯路_FlowMatching.html`。

改末页仓库地址：编辑根目录 `repo_url.txt`（一行完整 URL），再跑一次 `build_deck.py`。

---

## 小提示

- `prototype/node_modules/` 未提交，首次跑 puppeteer 脚本前执行 `npm install`。
- 逐帧截图会占用几百 MB，`frames_*` 目录已在 `.gitignore` 中忽略。
- 视频合成使用 H.264 + `yuv420p`，保证 Windows 自带播放器与浏览器都能解码。
- 若只想在别的电脑放幻灯片，**只需要那一个 HTML 文件**，其余目录都不必带。

---

有兴趣一起研究深度学习可以随时找我。源码、模型权重、点云数据和全部导出工具都在这个仓库里，有需要自取；觉得还不错，点个 star 喵。
