# 交接文档：湘潭大学三拱门 Flow Matching 课堂演示

> 最后更新：2026-10-27
> **注意**：`D:\comfyUI\HANDOFF.md` 是另一个 hand_lora 项目的文档，**不要覆盖**。

---

## 0. 一句话

用 **Flow Matching（Rectified Flow + 一次 Reflow）** 把 8000 个二维高斯噪声点映射成湘大校徽中央的“三拱门”点云，交付一个**课堂电脑双击即开的离线单文件 HTML（11 页）**，用三段预渲染 3D 视频讲清：

**随机配对 → 连线交叉 → 轨迹弯曲 → Reflow 拉直 → 1 步生成。**

---

## 1. 最终交付物

| 文件 | 说明 |
|---|---|
| `少走弯路_FlowMatching.html` | **唯一需要带到教室的文件**，约 32.5 MB，内嵌三段 MP4、校徽图、全部 CSS/JS |
| `README.md` | 项目说明、结果表、目录结构、复现步骤 |
| `HANDOFF.md` | 本文档 |
| `repo_url.txt` | 幻灯片末页显示的仓库地址（一行 URL），改完重跑 `python prototype/build_deck.py` |

课堂要求（已验证）：

- 完全离线，不请求网络；不依赖 Python / Node / ffmpeg / 服务器。
- 教室电脑**不需要 WebGL**：三段动画已渲染成 H.264 MP4，幻灯片只负责播放。
- Chrome / Edge 可播；Windows 10/11 自带 Edge 即可。
- 操作：`←` `→` / 空格翻页，`F` 全屏，`N` 演讲者备注，视频右下角重播。

---

## 2. 用户约束与偏好（务必遵守）

- 始终用中文回复；技术术语、代码、命令可保留英文。
- 最终交付必须离线、双击即开；教室电脑不跑 Python，也不跑 Three.js/WebGL。
- **8000 点已定稿**，不再重训 15000 点版本。所有数据、权重、统计、视频都基于 8000 点。
- 不要起本地 HTTP 服务器验证，用户自己打开文件看效果。
- 不要盲目调相机参数；用户更愿意肉眼检查构图。
- 用户偏快；不要长时间打磨非核心细节。
- 不要覆盖 `D:\comfyUI\HANDOFF.md`（hand_lora 项目）。

---

## 3. 最终实验数据（8000 点）

模型：`VelocityMLP(hidden=320, depth=5)`，输入 `(x, t)`，输出二维速度；**433,282 参数**。

```
n_points      = 8000
n_pairs       = 32000
steps1/steps2 = 12000 / 12000
batch         = 8192
lr            = 2e-3
EMA           = 0.999
```

| 指标 | 1-RF | 2-RF |
|---|---:|---:|
| 配对连线交叉率 | 0.2143（21.43%） | 0.0025（0.25%） |
| 平均多绕路 | +58.7%（中位 1.43×） | +0.0079%（中位 1.000004×） |
| 1 步 Chamfer | 0.5742 | **0.0173** |
| 2 步 Chamfer | 0.1722 | 0.0160 |
| 4 步 Chamfer | 0.0580 | 0.0155 |
| 10 步 Chamfer | 0.0279 | 0.0152 |
| 50 步 Chamfer | 0.0151 | 0.0151 |

结论：1-RF 1 步塌缩成小点团；2-RF 1 步已是清晰三拱门。数据源：`data/rf_stats.json`。

**讲稿注意**：连续 ODE 轨迹不会真正相交。课堂上说的“交叉”是 **随机配对连线**（`x0_i → x1_i`）的交叉，以及少步 Euler 折线；不要说成 ODE 解相交。

---

## 4. 目录结构

```
logo_diffusion_demo/
├── 少走弯路_FlowMatching.html   # 最终交付
├── README.md / HANDOFF.md / repo_url.txt
├── target/                       # target.jpg、target_rgba.png、target_rgba_full.png
├── data/                         # target_points.npz、rf_pairs.npz、rf_trajectory.npz、rf_stats.json
├── checkpoints/                  # rf1.pt、rf2.pt
├── scripts/                      # prepare.py、rf_model.py、train_rf.py、plot_rf_validate.py
├── docs/                         # 教案文案.md、preview/*.png
├── prototype/
│   ├── videos/                   # cross.mp4 17MB、three_layer.mp4 3.4MB、color.mp4 4.7MB
│   ├── cross.html / proto.html / color.html
│   ├── export_cross.py / export_proto.py / export_color.py   # npz → *_data.js
│   ├── build_cross.py / build_proto.py / build_color.py      # → *_standalone.html
│   ├── export_video.js / export_proto_video.js / export_color_video.js  # 逐帧截图
│   ├── deck_template.html / build_deck.py / shot_deck.js
│   ├── logo_small.webp、vendor/、package.json
│   └── node_modules/             # 不提交；跑 puppeteer 前 npm install
└── ddpm/                         # 早期 DDPM 归档，未用于最终交付
```

`prototype/` 下的 `*_data.js`、`*_main.js`、`bundle*.js`、`*_standalone.html`、`frames_*/`、`shots/` 都是生成物，已在 `.gitignore` 中忽略。

---

## 5. 幻灯片结构（11 页）

| # | 内容 | 备注 |
|---|---|---|
| 1 | 封面 | 点阵校徽 + 标题 |
| 2 | 扩散原理 | 噪声 → 图像的“流”；路径无数种 |
| 3 | 传统曲线路径 | 为什么需要多步 Euler |
| 4 | Flow Matching | 用直线构造路径 |
| 5 | 实验设置 | 8000 点、433,282 参数、训练配置 |
| 6 | 视频① | 随机配对、交叉、轨迹弯曲 |
| 7 | Reflow | 用 1-RF 生成配对再训练 |
| 8 | 视频② | 1-RF vs 2-RF 少步对比 |
| 9 | Chamfer | 1/2/4/10/50 步数据表 |
| 10 | 视频③ | 灰度 LUT 色板染色 |
| 11 | 总结 | 三句话 + 仓库卡片（地址来自 `repo_url.txt`） |

技术实现：1920×1080 固定舞台 + CSS `transform: scale()` 适配投影仪；三段 MP4 以 base64 内嵌，翻到对应页才创建 Blob URL 播放；切换页面暂停/重播；`N` 键显示 `data-note` 讲稿。

---

## 6. 复现流程

### 6.1 数据与模型（Python）

```bash
cd D:/comfyUI/logo_diffusion_demo
python scripts/prepare.py
python scripts/train_rf.py --steps1 12000 --steps2 12000 --reflow-pairs 32000
python scripts/plot_rf_validate.py   # 可选：只重画验证图
```

- Python 依赖：`torch numpy matplotlib opencv-python pillow`；首次抠图需要 `rembg`。
- 训练自动用 CUDA，无 GPU 回退 CPU。
- RTX 4050 Laptop 6GB 上两阶段合计约 6~8 分钟。

### 6.2 视频（Node + Edge + ffmpeg）

```bash
cd prototype
npm install
python export_cross.py && python build_cross.py
python export_proto.py && python build_proto.py
python export_color.py && python build_color.py
node export_video.js
node export_proto_video.js
node export_color_video.js
ffmpeg -framerate 30 -i frames_cross/f%04d.png       -c:v libx264 -pix_fmt yuv420p -crf 20 videos/cross.mp4
ffmpeg -framerate 30 -i frames_proto/f%04d.png       -c:v libx264 -pix_fmt yuv420p -crf 20 videos/three_layer.mp4
ffmpeg -framerate 30 -i frames_color/f%04d.png       -c:v libx264 -pix_fmt yuv420p -crf 20 videos/color.mp4
```

- 每段 360 帧 / 30fps / 12 秒 / 1920×1080。
- 逐帧截图用**有头 Edge**（`headless:false`，窗口移到屏幕外），因为需要真实 GPU 渲染。所有脚本路径均已改为基于 `__dirname`，仓库换位置也能跑。
- 颜色视频的相机与色板参数在 `prototype/color.html`：`YAW0=0.70, PITCH=0.38, dist=7.6-1.2*e`；底噪亮度 `0.09+hash*0.26`；色板亮度节点 `0.035/0.19/0.45/0.68/0.92`。

### 6.3 幻灯片

```bash
cd prototype
python build_deck.py
```

`build_deck.py` 读取 `deck_template.html`，把 `logo_small.webp`、`videos/*.mp4` 和根目录 `repo_url.txt` 注入后写出根目录 HTML。改末页地址后重跑即可。

### 6.4 截图验证（可选）

```bash
cd prototype
node shot_deck.js        # 全部 11 页
node shot_deck.js 10     # 只截第 11 页（0 基）
```

输出在 `prototype/shots/deck_XX.png`。需要 `npm install`。

---

## 7. 设计决策记录

1. **用预渲染视频，不做实时 WebGL**：教室电脑性能与稳定性不可控；单文件也不依赖 GPU。实时 Three.js 版本仅作为 `prototype/*.html` 的开发场景保留。
2. **8000 点定稿**：15000 点方案已放弃，不再重训。
3. **颜色是前端 LUT 映射**：模型只学生成位置（2D）；颜色由目标图 `target_sat/light` + 色板色相决定，因此换色板不需要重跑模型。视频③展示灰度 → 色板染色。
4. **“交叉”讲配对连线**：见第 3 节讲稿注意。
5. **视频③只做视觉加分**：核心叙事仍是第 6、8 页的轨迹对比。

---

## 8. 已知限制 / 后续可做

- **2-RF 泛化未量化**：`train_rf.py` 里有 fresh-noise 采样代码，但当前 `rf_stats.json` 没有 `fresh_noise` 字段。若要在课堂外承诺“任意噪声 1 步可用”，先补跑该测试；否则只讲“同一批噪声粒子的传输”。
- 2-RF 在固定 32000 组 reflow 配对上训练，存在记忆配对的风险；课堂演示足够，不算严格泛化实验。
- `ddpm/` 是早期方案归档，未接入最终演示，保留作对比/回退。
- 若要把 HTML 发到 GitHub Pages/网页，可只上传 `少走弯路_FlowMatching.html`；32.5 MB 在 GitHub 单文件限制内。

---

## 9. 运行环境（本次生成时）

- Windows 11，Python 3.14.2，PyTorch 2.14.0+cu126
- GPU：RTX 4050 Laptop 6GB
- Node.js + `puppeteer-core`，使用系统 Edge（`C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe`）
- ffmpeg 用于合成 H.264 MP4
