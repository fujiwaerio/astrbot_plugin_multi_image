# AstrBot 多模型生图插件

一个指令打通七个生图渠道，并且**专门解决「中转站跑不通」的问题**。

| 渠道 | 接的是什么 | 典型模型 |
| --- | --- | --- |
| `gpt` | **通用 OpenAI 兼容位**：任何说 OpenAI 协议的服务 | `gpt-image-1`、`flux-1.1-pro`、中转站上的任意生图模型 |
| `gemini` | Gemini 原生协议（失败自动回退 OpenAI 兼容层） | `gemini-2.5-flash-image`、`imagen-*` |
| `comfyui` | 本地部署的 ComfyUI，支持多套工作流模板（只有自己跑了它才用得上） | 自建的工作流 |
| `seedream` | 火山方舟 `/api/v3` | `doubao-seedream-4-0-250828` |
| `grok` | xAI 协议 | `grok-imagine-image-2.0` |
| `jimeng` | 即梦（sessionid 方案） | `jimeng-4.0`、`nanobanana` |
| `qwen` | 阿里云百炼原生协议 | `qwen-image-2.0`、`wan2.7-image` |

支持**图生图 / 垫图**，支持把参考图和提示词一起发送。

---

## 30 秒上手

1. **安装**：把插件目录放进 AstrBot 的 `data/plugins/`，重启 AstrBot，或在 WebUI 的
   「插件」页点一次「重载插件」。依赖只有 `httpx`（一般随 AstrBot 一起装好）。
2. **配一个渠道**：打开 WebUI 的「插件 → 多模型生图」配置页。
   七个渠道是**并列的可选槽位**，不是必须配齐——**配好任意一个就能生图**。
3. **开画**：

   ```
   /draw 一只坐在窗边的橘猫，柔和自然光
   ```

配好之后发 `/draw_models` 可以看到哪些渠道可用、哪些还没配。

> **「未配置」是正常状态，不是故障。** 七个渠道的凭据分属不同厂商
> （OpenAI、Google、xAI、火山方舟、阿里云百炼……），手里只有其中一两个很常见，
> 其余留空不影响使用。插件把「未配置」和「配置不完整」（填了 Key 但还缺东西）
> 分开显示，就是为了避免把「还没配」误读成「坏了」。
>
> ⚠️ 唯一的前提是**默认渠道得是配好的那一个**：`default_provider` 默认指向 `gpt`，
> 如果只配好了别的渠道，直接发 `/draw 一只猫` 会提示「渠道未配置完整」。
> 三种改法——指令里写明渠道（`/draw seedream 一只猫`）、群里用
> `/draw_switch seedream`、或把配置页的「默认生图渠道」改成配好的那个。

### 每个渠道需要准备什么

| 渠道 | 需要准备 |
| --- | --- |
| `comfyui` | **只有自己机器上跑着 ComfyUI 时才用得上**：填它的地址，再加一套工作流（不需要 Key） |
| `seedream` | 火山方舟的 API Key + 已开通的模型 ID |
| `qwen` | 阿里云百炼的 API Key |
| `gpt` / `gemini` / `grok` | 各家的官方 Key，或一个**真正带生图模型**的中转站 |
| `jimeng` | 即梦 2API 服务地址 + `sessionid` |

**没装 ComfyUI 就把那一整块留空**，它和别的渠道互不相干。
插件不会去探测本机有没有 ComfyUI，也不会因此报错——那一块保持空着，
它就一直是「未配置」，不参与任何生图流程。

---

## 指令

### `/draw` —— 生图

```
/draw [渠道[@工作流]] [参数] <提示词>
```

别名：`/生图`、`/画图`。渠道名可以省略，也可以写别名：

| 标准名 | 可以写成 |
| --- | --- |
| `gpt` | `openai`、`chatgpt`、`dalle`、`gpt-image` |
| `gemini` | `google`、`imagen`、`nano`、`nanobanana`、`banana` |
| `comfyui` | `comfy`、`工作流` |
| `seedream` | `doubao`、`豆包`、`ark`、`方舟`、`seededit` |
| `grok` | `xai`、`x-ai` |
| `jimeng` | `即梦`、`dreamina`、`jm` |
| `qwen` | `万相`、`通义`、`百炼`、`dashscope`、`wanx` |

| 参数 | 说明 |
| --- | --- |
| `--size 1024x1024` | 输出尺寸（`-s` 亦可）。**像素写法各渠道通用** |
| `--size 2K` | 尺寸档位，Seedream 用（`1K`/`1.5K`/`2K`/`3K`/`4K`，大写 K） |
| `--width 800` / `--height 600` | 分别指定宽高（`-w` / `-h`） |
| `--ratio 16:9` | 画面比例（Gemini 原生、即梦用） |
| `--resolution 2k` | 分辨率档位（即梦用：`1k`/`2k`/`4k`） |
| `--n 2` | 生成数量（受 `max_count` 限制） |
| `--seed 123` | 随机种子 |
| `--negative 模糊,低质量` | 负向提示词（值里不能带空格） |

```bash
/draw gpt 一只坐在窗边的橘猫，柔和自然光，摄影风格
/draw gemini 未来城市夜景，电影感构图
/draw comfyui 水彩风格的山间小屋
/draw comfyui@zimage 赛博朋克少女 --size 2048x2048
/draw seedream 中国古风庭院，细腻插画
/draw grok 太空中的机械鲸鱼
/draw 即梦 春日樱花街道 --ratio 16:9
/draw 一只橘猫                       # 不写渠道，用当前默认渠道
```

#### 尺寸参数在各渠道的落地方式

各家的尺寸接口差异极大，插件做了统一转换。**`--size` 的像素写法会同时写入宽高**，
所以 `--size 1920x1080` 在所有渠道都生效：

| 渠道 | 实际发送的字段 | 说明 |
| --- | --- | --- |
| `gpt` | `size: "1024x1024"` | 原样透传；`gpt-image-1` 只支持 1024×1024 / 1536×1024 / 1024×1536 |
| `seedream` | `size: "2K"` 或 `"2048x2048"` | 档位与像素二选一，不可混用 |
| `gemini`（原生） | `generationConfig.imageConfig.aspectRatio` | **默认不发送**，见下方说明 |
| `gemini`（Imagen） | `parameters.aspectRatio` | 由宽高换算成 `1:1`/`4:3`/`16:9` 等 |
| `comfyui` | 尺寸节点的 `width`/`height`/`batch_size` | 也支持工作流里的 `{{width}}`/`{{height}}` 占位符 |
| `grok` | `size`（默认）或 `aspect_ratio` + `resolution` | 由配置项「尺寸参数写法」决定 |
| `jimeng` | `ratio` + `resolution` | 比例按宽高自动换算，也可用 `--ratio` 指定 |
| `qwen` | `parameters.size`（`*` 分隔） | **默认不发送**，交给服务端默认值 |

> **为什么 Gemini 原生协议默认不发比例**：`imageConfig.aspectRatio` 这个字段有个坑——
> 不支持它的模型传了会**直接报错**（而不是忽略）。需要时用 `--ratio 16:9` 单次指定，
> 或把渠道配置里的「始终发送画面比例」打开。
>
> **为什么百炼默认不发尺寸**：各模型支持的尺寸集合差异很大，与其猜错，不如让服务端
> 用默认值。代价是输出可能偏大——实测 `qwen-image-2.0` 不传 `size` 时返回
> **2048×2048 PNG，约 5.5 MB**。消息平台对图片体积敏感的话，建议填 `1328*1328`。

### `/draw_models` —— 查看渠道状态

别名：`/生图模型`、`/生图渠道`。列出每个渠道是「已就绪」「未配置」还是
「配置不完整」（并指出缺什么），同时显示当前会话实际会用的渠道；
如果配了 ComfyUI 的工作流，还会把可用的别名一并列出来。

**不受名单限制**，方便在被拒绝的群里自查配置。

### `/draw_switch` —— 切换本群渠道

别名：`/生图切换`、`/切换生图`、`/切换渠道`。

```
/draw_switch <渠道[@工作流]>      把本群默认渠道换成它
/draw_switch                     查看本群当前渠道和可切换的渠道
/draw_switch 默认                 清除本群设置，改回跟随全局默认渠道
```

**每个群各自独立**：A 群切到 `seedream` 不影响 B 群。设置写在插件数据目录的
`group_channels.json` 里，重启、重载插件后依然有效。

用处是：群里的人不用每次都写渠道名，直接 `/draw 一只猫` 就用本群约定的那个渠道；
而插件配置里的全局默认渠道可以另设一个。

渠道的选择顺序固定为：

```
1. /draw 里显式写的渠道      /draw gpt 一只猫
2. 本群 /draw_switch 设定的   /draw_switch seedream
3. 配置里的全局默认渠道        default_provider
```

也就是说，本群切到 A 之后，任何人都仍然可以临时用 `/draw B 提示词` 走别的渠道。

几条边界：

- 渠道名从前往后扫，**第一个认识的词**就算数，所以
  `/draw_switch 帮我换成 seedream` 这种自然说法也能用。
- 只能切到**当前已就绪**的渠道。切到一个没配好的渠道，会让之后每一次生图都失败，
  对群里其他人来说就是「机器人坏了」——所以插件直接拒绝，并说明缺什么。
- 权限与生图**完全一致**：被黑名单拦下的人改不了设置；`admin_only` 打开时
  只有管理员能切。
- 私聊里不能用（没有「群」这个概念），私聊直接在指令里写渠道即可。
- 可以连工作流一起固定：`/draw_switch comfyui@anime`，之后 `/draw 提示词`
  就会用这套工作流。

### 图生图 / 垫图

把参考图和提示词**一起发送**即可，无需额外参数：

```
[发送一张图片] /draw seedream 把背景换成雪山
```

各渠道的实现方式不同，插件会自动选择：

| 渠道 | 图生图走法 | 参考图传参 |
| --- | --- | --- |
| `gpt` | `POST /v1/images/edits`（multipart） | 文件流，多图用同名字段重复 |
| `gemini` | `:generateContent` 的 `inlineData` | Base64 |
| `seedream` | 同一个 `/images/generations` 接口的 `image` 字段 | Base64 data URI 数组 |
| `grok` | `POST /v1/images/edits`，**官方是 JSON** | `{"url": ..., "type": "image_url"}` |
| `jimeng` | `POST /v1/images/compositions` | 优先用图片原始 URL，否则 data URI |
| `comfyui` | 上传到 `/upload/image` 后写入 `LoadImage` 节点 | 文件名 |

参考图是否启用、数量上限、单张大小上限，分别由 `enable_reference`、
`max_references`、`reference_max_mb` 控制。

### 图像编辑模型的提示

有一类模型**必须**带参考图才能工作，光给文字它只会报错：

| 类型 | 模型名特征 | 例子 |
| --- | --- | --- |
| 图像编辑 | `*-edit`、`*image-edit*` | `qwen-image-edit-max`、`grok-imagine-image-1.0-edit` |
| 局部重绘 | `*seededit*`、`*inpaint*`、`*outpaint*` | `doubao-seededit-3-0-i2i-250628` |

选用了这类模型却没发参考图时，插件会在**发出请求之前**拦下来，并给出可操作的提示：

```
「qwen-image-edit-max」是图像编辑模型，必须带一张参考图才能用。
用法：把图片和提示词一起发给我，例如「[发一张图] /draw qwen 把背景换成雪山」
如果只是想用文字生成图片，请把 qwen 渠道的模型换成 qwen-image-2.0 或 qwen-image-3.0。
```

替代模型的建议是**按渠道**给的（Seedream 会建议换成 Seedream，而不是 SeedEdit）。
识别用的是词边界匹配，不会误伤 `gpt-image-1`、`gemini-2.5-flash-image` 这类
名字里带 `image` 但其实是文生图的模型。

---

## 中转站（聚合站）兼容性

NewAPI / OneAPI 这类聚合站对同一个接口的支持程度差异极大，「照文档写」经常直接报错。
插件把实际踩到的坑都做成了自动处理：

| 中转站的实际情况 | 插件的处理方式 |
| --- | --- |
| 用户填的地址有的是裸域名、有的带 `/v1`、有的是完整接口地址 | `resolve_endpoint` 归一化，不重复也不缺版本前缀 |
| 只实现了 `/v1/chat/completions`，图片写在回复正文里 | 自动回退到 chat 接口，并从 markdown / `<img>` / data URI 里抠图 |
| 没实现 `/v1/images/generations`，返回 404 / 501 | 识别为「接口不存在」，自动换协议重试 |
| 多传一个 `size` / `response_format` / `n` 就 400 | 上游点名哪个参数，就自动摘掉哪个参数重发（最多 3 个） |
| 把 Base64 直接塞在 `url` 字段里返回 | 按内容嗅探，自动识别为 Base64 图片 |
| 返回 `/files/a.png` 这种站内相对路径 | 基于 API 地址补全成绝对 URL |
| 图片链接需要鉴权才能下载 | 开启 `force_download` 后由插件下载成本地文件发送；鉴权头**只发给与接口同源的地址** |
| 上游返回网页形式的错误（Cloudflare 502 之类） | 只提取页面标题 + Ray ID + 一句通俗解释，不把整页 HTML 发到群里 |
| 需要自定义鉴权头、或走代理、或自签名证书 | `extra_headers` / `proxy` / `verify_ssl` 三个配置项 |
| 限流、网关抖动 | 408/409/425/429/5xx 指数退避重试，尊重 `Retry-After` |
| 鉴权失败（401/403） | 不会傻乎乎地换协议重试，直接如实报错 |

### 用中转站跑 GPT / Gemini / Grok 生图

这三家**官方**的生图接口各有自己的 Key，且价格不低。常见的两条路：

| 路线 | 需要准备 |
| --- | --- |
| 官方直连 | 各家官方 Key。`gpt` 填 `https://api.openai.com`；`gemini` 填 `https://generativelanguage.googleapis.com`；`grok` 填 `https://api.x.ai` |
| 走中转站 | 一个**真正带生图模型**的中转站地址 + Key，一个 Key 用多家模型 |

> ⚠️ **选第二条路时，先确认那家站真的有生图模型。**
> 很多中转站只代理对话模型，`GET {api_base}/v1/models` 列出来的全是文本模型，
> 生图模型一个都没有——这种情况下填进去只会报「模型不存在」。
> 把模型列表拉出来搜一下 `image` / `seedream` / `flux` 之类的关键词再配。

### 一个渠道只能配一个地址

七个渠道就是七个槽位，每个槽位一组 `api_base` + `api_key`，各自独立。
如果手上有**两个**同为 OpenAI 协议的服务（例如官方 OpenAI + 一个 FLUX 中转站），
`gpt` 槽一次只能填一个，切换时要改配置。

但**同一个站点上的多个模型**有现成的办法：把第二个模型填进 `gemini` 槽，
并把它的 `api_style` 设成 `openai`。此时这个槽会退化成标准 OpenAI 请求，
打的是同一个 `{api_base}/images/generations`：

| 槽位 | `api_base` | `model` | `api_style` |
| --- | --- | --- | --- |
| `gpt` | `https://站点地址/v1` | `gpt-image-2` | `auto` |
| `gemini` | `https://站点地址/v1` | `agnes-image-2.1-flash` | `openai` |

之后 `/draw gpt …` 和 `/draw gemini …` 就分别用这两个模型，互不干扰；
也可以用 `/draw_switch` 把某个群的默认渠道定成更快的那个。

> 这个办法对任何 OpenAI 兼容的站点都成立，不限于上表中的两个模型。
> 注意「能不能垫图」取决于**上游**是否实现了 `/images/edits`——
> 上例中的 `agnes` 没有实现（请求会挂住直到超时），需要图生图时仍要用 `gpt` 槽。

这是「槽位固定、开箱即用」的取舍。真需要三个以上同协议槽位，可以再加一个渠道位。

---

## 各渠道配置

### 通用字段

每个渠道都有的字段：

| 字段 | 说明 |
| --- | --- |
| `api_base` | API 地址。裸域名 / 带 `/v1` / 完整接口地址都能识别 |
| `api_key` | 密钥（即梦填 `sessionid`） |
| `model` | 模型名 |
| `api_style` | 调用协议，`auto` 表示按顺序自动回退 |
| `timeout` | 单次请求超时（秒） |
| `extra_headers` | 附加请求头，如 `{"X-Api-Key":"xxx"}` |
| `extra_body` | 附加请求体字段，如 `{"quality":"high"}` |
| `proxy` | HTTP(S) 代理，如 `http://127.0.0.1:7890` |
| `verify_ssl` | 自建自签名站点可关闭 |
| `force_download` | 开启后由插件把图片下载成临时文件再发送。渠道 Key **只发给与 `api_base` 同源的地址**：图床挂在第三方域名（对象存储 / CDN）时自动改用裸请求；若同源地址仍拒绝鉴权头，会去掉头再试一次 |

> 每个渠道的 `api_base` + `api_key` 就是**自定义 URL + Key** 的位置。
> 七个渠道各有一组，不需要再单独做一个「自定义渠道」。

### 先理解一件事：渠道是按「协议」划分的，不是按厂商

这一点理解错了会走很多弯路。

**`gpt` 渠道不只是给 GPT 用的。** 只要对方的站提供 OpenAI 协议的
`POST /v1/images/generations`（退一步，`/v1/chat/completions` 能出图也行），
那么不管它背后实际是 **FLUX、Stable Diffusion、Ideogram、Recraft** 还是别的什么，
**都填进 `gpt` 渠道就行**——不需要改代码，也不需要新增渠道。

对接中转站也是这个姿势：中转站通常一个地址代理几十家模型，
把地址和 Key 填进 `gpt`，`model` 写成它那边的模型名即可。

### `gpt`（通用 OpenAI 兼容位）

- `api_base`：官方 OpenAI 填 `https://api.openai.com`；中转站/第三方填其地址
- `model`：官方是 `gpt-image-1`；中转站可能是 `gpt-4o-image`、`dall-e-3`；
  第三方服务则是它自己的模型名（如 `flux-1.1-pro`、`sd3.5-large`）
- `api_style`：`auto` 先试图片接口，失败再回退对话接口
- 插件按模型名自动决定是否携带 `response_format`
  （`gpt-image-*` 不支持该字段，`dall-e-*` 需要它）
- `size`：留空则用全局默认宽高（1024×1024）

### `gemini`

- `api_base`：官方填 `https://generativelanguage.googleapis.com`
- `model`：`gemini-2.5-flash-image` 等图像模型走 `:generateContent`；
  名字含 `imagen` 的模型自动改走 `:predict`
- `api_style`：`auto` 的尝试顺序是
  原生协议 → 谷歌官方 OpenAI 兼容层（`/v1beta/openai`）→ 中转站 `/v1`
- `response_modalities`：默认 `TEXT,IMAGE`，纯图片模型可改成 `IMAGE`
- `send_aspect_ratio`：默认关，原因见上文「尺寸参数」

### `comfyui`

> 这一块**只对已经自己部署了 ComfyUI 的人有意义**。没装就整块留空，
> 它和别的渠道互不相干。地址默认是空的——插件不会去探测本机有没有 ComfyUI，
> 也不会因此报错；**判断这个渠道是否可用的唯一标准是有没有工作流**，
> 所以留空时它一直是「未配置」，不参与任何生图流程。

- `api_base`：**必须填**——填 ComfyUI 的根地址，如 `http://127.0.0.1:8188`
  （插件自己拼接 `/prompt`、`/history`、`/view`；不会默认去连本机 8188）
- `workflows`：工作流模板列表，可存多套，用 `/draw comfyui@别名` 切换
- `default_workflow`：不写 `@别名` 时用哪一套（留空用第一个模板）
- `max_wait`：最长等待时间，复杂工作流请调大（默认 300 秒）
- `api_key`：ComfyUI 套在带鉴权的反向代理后面时才需要

工作流有**两种注入方式**，任选其一：

1. **占位符**（推荐，最可控）：在导出的 JSON 里直接写
   `{{prompt}}`、`{{negative_prompt}}`、`{{width}}`、`{{height}}`、
   `{{seed}}`、`{{count}}`，插件会做 JSON 安全转义后替换。
2. **节点注入**：没有占位符时，插件按节点类型自动查找——
   第一个 `CLIPTextEncode` 作正向、第二个作负向，
   `EmptyLatentImage` / `EmptySD3LatentImage` 写宽高与批量，
   `KSampler` 写种子，`LoadImage` 写参考图文件名。
   也可以在模板里显式填写各节点的 id 覆盖自动查找。

> ⚠️ 工作流必须是 ComfyUI 界面里「保存（API 格式）」导出的 JSON，
> 不是普通的 `workflow.json`。

### `seedream`（火山方舟）

- `api_base`：国内 `https://ark.cn-beijing.volces.com/api/v3`；
  国际 `https://ark.ap-southeast.bytepluses.com/api/v3`
- `model`：需带日期后缀，如 `doubao-seedream-4-0-250828`
  （国际区不带 `doubao-` 前缀）
- `size`：档位 `1K`/`1.5K`/`2K`/`3K`/`4K`（大写 K），或 `2048x2048`
  这种像素写法，**两者不可混用**
- `watermark`：默认关。实测不显式传 `watermark` 时，方舟会在右下角加
  「AI生成」水印；插件默认发送 `watermark: false`，所以**正常使用不会带水印**
- `optimize_prompt`：启用官方提示词优化，默认关

> 方舟官方 4.0/4.5/5.x 的参数表里**没有 `n` 和 `seed`**。
> 插件默认只发送官方文档中的字段；只有在指令里显式指定 `--seed`、或 `--n` 大于 1 时
> 才会带上它们，一旦被上游拒绝会被自动摘掉重试。

### `grok`（xAI）

- `api_base`：官方填 `https://api.x.ai`；中转站填其地址
- `model`：官方当前为 `grok-imagine-image-2.0`；老中转站可能只认 `grok-imagine-1.0`
- `edit_model`：留空则图生图沿用同一个模型
- `size_mode`：`size` 发 OpenAI 风格尺寸（中转站常用）；
  `aspect_ratio` 发官方的 `aspect_ratio` + `resolution`
- `edit_mode`：`auto` 先按**官方的 JSON** 格式发图生图请求，
  失败再退回 multipart（部分中转站只实现了后者）

### `jimeng`（即梦）

> 和 ComfyUI 一样，这一块**只对已经自己搭了服务的人有意义**：
> 即梦官方没有开放普通用户直接调用的生图 API，社区做法是先在自己机器上跑一个
> `jimeng-2api` / `jimeng-free-api` 之类的转换服务，插件再连它。
> 地址默认是空的，没搭服务就把这一块留空。

- `api_base`：**必须填**——填自己搭的那个服务的地址，如 `http://localhost:5100`
- `api_key`：即梦官网的 `sessionid`。**支持填多个**（逗号或换行分隔），
  遇到限额或失效会自动轮换；国际站需加前缀，如 `us-`
- `model`：`jimeng-4.0`、`jimeng-3.1`、`nanobanana` 等
- `ratio` / `resolution`：留空则比例按图片宽高自动换算，分辨率默认 `2k`
- `sample_strength`：图生图采样强度，越大越偏离原图
- `reference_mode`：参考图传 `data_uri` 还是裸 `base64`

> 即梦的图生图接口原生只接受**图片 URL**。如果参考图来自网络，插件会直接透传
> 它的 URL；如果是本地图片，可以开启全局的「注册到回调文件服务」
> （`use_file_service`），借助 AstrBot 的 `callback_api_base` 换成一个可访问的 URL。

### `qwen`（阿里云百炼 / 通义万相）

百炼名下有一批生图模型（`qwen-image-*`、`wan2.7-image`、`z-image-turbo` 等），
但**它的图片生成不在 OpenAI 兼容层里**——往
`/compatible-mode/v1/images/generations` 发请求会直接 404，必须走原生 API。
插件把两条原生路径都实现好了：

| 路径 | 接口 | 适用模型 |
| --- | --- | --- |
| 同步多模态 | `POST /api/v1/services/aigc/multimodal-generation/generation` | `qwen-image-*`、`z-image-*` |
| 异步任务 | `POST /api/v1/services/aigc/text2image/image-synthesis` + 轮询 `/api/v1/tasks/{id}` | `wan*`、`wanx*` |

- `api_base`：填 `https://dashscope.aliyuncs.com`。
  **填成 `/compatible-mode/v1` 也没关系**，插件会自动还原成根地址
  （这个坑很容易踩：拼出来会变成 `/compatible-mode/v1/api/v1/...`）
- `model`：默认 `qwen-image-2.0`；`wan*` 系列自动改走异步任务路径
- ⚠️ `qwen-image-edit-max` / `qwen-image-edit-plus` 是**图像编辑模型，必须带参考图**，
  详见上文「图像编辑模型的提示」
- `size`：**默认不发送**，原因与代价见上文「尺寸参数」
- `api_style`：`auto` 按模型名自动选择，也可强制 `multimodal` / `text2image`
- 结果里直接带图就返回，带 `task_id` 就自动轮询，**两条路径共用一套逻辑**，
  所以在「这个模型到底走哪条」上判断失误也不会直接失败

---

## 全局配置

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `default_provider` | `gpt` | 不写渠道名时用哪个（群里可用 `/draw_switch` 覆盖） |
| `default_width` / `default_height` | 1024 | 默认尺寸 |
| `default_count` / `max_count` | 1 / 4 | 默认与最大生成数量 |
| `timeout` | 180 | 默认请求超时（秒） |
| `enable_reference` | 开 | 是否允许图生图 |
| `max_references` | 3 | 最多使用几张参考图 |
| `reference_max_mb` | 8 | 单张参考图大小上限（MB） |
| `use_file_service` | 关 | 把本地参考图注册到 AstrBot 回调文件服务 |
| `force_download` | 关 | 全局强制代为下载图片 |
| `admin_only` | 关 | 仅管理员可用 |
| `admin_bypass` | 关 | 管理员绕过四个名单与频率限制 |
| `rate_limit_enabled` | 关 | 启用频率限制，见下文 |
| `rate_limit_scope` | `user` | `user` 按人计数 / `group` 按群共享额度 |
| `rate_limit_window_minutes` | 10 | 统计窗口（分钟） |
| `rate_limit_max_calls` | 5 | 窗口内最多生成几次 |
| `group_whitelist` | 空 | 群白名单，见下文 |
| `group_blacklist` | 空 | 群黑名单，见下文 |
| `group_user_whitelist` | 空 | 群内个人白名单，见下文 |
| `group_user_blacklist` | 空 | 群内个人黑名单，见下文 |
| `one_task_per_session` | 开 | 同一会话同时只跑一个任务 |
| `cache_keep_minutes` | 30 | 本地图片缓存保留时长，0 表示不清理 |

### 持久化位置

数据存放在 `data/plugin_data/astrbot_plugin_multi_image/`，**不写插件自身目录**，
插件更新不会丢数据：

| 文件 | 内容 |
| --- | --- |
| `cache/` | 代下载的图片缓存，按 `cache_keep_minutes` 清理 |
| `group_channels.json` | 各群用 `/draw_switch` 设定的渠道，删掉即恢复全局默认 |

---

## 频率限制

> ⚠️ **建议开启。** 生图是一次几毛钱到几块钱的真实调用，群里连刷会很快把额度烧完。
> 默认关闭只是为了不改变既有安装的行为。

| 配置项 | 默认 | 说明 |
| --- | --- | --- |
| `rate_limit_enabled` | 关 | 总开关 |
| `rate_limit_scope` | `user` | `user` = 每人独立额度；`group` = 整个群共用一份额度 |
| `rate_limit_window_minutes` | 10 | 统计窗口 |
| `rate_limit_max_calls` | 5 | 窗口内最多生成几次 |

超过限制时回复：

```
生图太频繁了，请 4 分 12 秒 后再试。
当前限制：10 分钟内你最多生成 5 次。
```

### 几个设计取舍

- **只有成功受理的调用才计数。** 被拦下的重试不会把窗口往后推——否则越急的人
  越用不了，行为也不可预测（可以理解成固定窗口，而不是「每被拒一次就加罚」）。
- **`scope=group` 时私聊按人算。** 私聊没有群号，如果一律归到同一个键，
  所有私聊用户会共用一份额度，那就荒谬了。
- **频率限制排在并发占位之前。** 被限流的请求不该占用会话位，
  否则用户等完限流又会撞上「你已有一个任务在进行中」。
- **`admin_bypass` 打开时管理员不受限。** 这个开关同时管名单和频率限制，
  语义统一为「管理员不受约束」。
- 限流记录随插件卸载清空；过期条目在每次检查时顺带清理，不会无限增长。

### 和名单的关系

两者是**独立的两道闸**，按这个顺序过：

```
1. 名单（谁能用）       群黑名单 / 群白名单 / 群内个人名单
2. 频率限制（能用多快）  每人或每群 N 分钟 M 次
3. 并发占位（同时几个）  同一会话同时只跑一个任务
```

名单决定「这个人能不能用」，频率限制决定「能用多快」。
**只配名单不配频率限制，群友依然可以把额度刷光**，这两件事要一起做。

---

## 准入控制

四个名单，**填哪个哪个生效**，不需要额外的「模式」开关。判定顺序从上往下：

```
1. admin_only 开启  → 非管理员一律拒绝
2. admin_bypass 开启 → 管理员绕过下面全部名单
3. 群黑名单非空      → 只按黑名单判，白名单不参与
4. 群白名单非空      → 只按白名单判
5. 以上都为空        → 才看群内个人名单（按当前群取，黑名单优先）
6. 都没配            → 不限制
```

> **核心规则：群名单一旦启用（白或黑任一非空），群内个人名单就整体不生效。**
> 这样两层不会互相打架。

### 群白名单 / 群黑名单

- 类型：ID 列表，元素就是群号
- **两者只能填一个**。都填时以**黑名单**为准（白名单被忽略）
- 群黑名单：名单里的群被拒绝，其他群放行；私聊放行
- 群白名单：只有名单里的群放行，其他群和私聊都被拒绝

### 群内个人白名单 / 群内个人黑名单

- 类型：JSON 对象，**键是群号，值是成员号数组**
- 只在群白名单和群黑名单**都留空**时生效
- 同一个群里，黑名单非空则白名单不生效；都没配则该群不限制

```json
{
  "123456789": ["10001", "10002"]
}
```

### 写法兼容性

ID 列表支持多种写法，因为它们可能来自配置面板，也可能来自手改的 JSON：

| 写法 | 是否接受 |
| --- | --- |
| `["123", "456"]` | ✅ |
| `[123, 456]`（不加引号的数字） | ✅ 自动转成字符串比较 |
| `"123,456"` | ✅ |
| `"123，456、789"`（中文标点） | ✅ |
| `"123\n456"` | ✅ |
| `{"123456": "10001,10002"}`（个人名单值写成字符串） | ✅ |
| `{"123456": [10001, 10002]}` | ✅ |

> 之所以要处理这么多写法：JSON 里**不加引号的纯数字会被解析成 int**，
> 直接和 AstrBot 传来的字符串群号比较会永远不相等——名单看起来填了却完全不生效。
> 插件在比较前统一归一化成去空白的字符串。

### 几点说明

- **管理员默认不绕过名单。** 名单是明确写下的规则，默认对所有人一视同仁——
  否则会出现「这个群明明在黑名单里，为什么还能用」这种从配置上看不出来的行为。
  需要豁免时打开 `admin_bypass`（默认关）。
- `admin_bypass` **只对管理员生效**，普通成员照常受限；`admin_only` 是最外层门禁，
  非管理员即使开了 `admin_bypass` 也进不来。
- **取不到管理员身份时按普通成员处理。** 万一平台的 `is_admin()` 抛异常，
  宁可让限制照常生效，也不能反过来静默放行。
- `/draw_models`（查渠道状态）**不受名单限制**，方便在被拒的群里自查配置。
- `/draw_switch`（切换本群渠道）**受名单限制**：它改的是别人之后会用到的设置，
  不能让被拉黑的人还能动。
- 被拒绝时会记一条日志，包含群号、用户和原因，便于排查。

---

## 常见问题

**Q：填了地址和 Key，还是报「模型不存在」。**
先 `GET {api_base}/v1/models` 看看那个模型在不在列表里。中转站很常见的情况是
只代理对话模型，一个生图模型都没有。

**Q：`/draw` 提示「未指定渠道，且没有配置默认渠道」。**
`default_provider` 指向的渠道没配好，或者指令里的渠道名无法识别。
发 `/draw_models` 看实际可用的渠道。

**Q：`/draw_switch` 说「当前没有已就绪的渠道」。**
一个渠道都还没配好。到插件配置页填任意一个渠道的
`api_base` + `api_key` + `model` 即可，`/draw_models` 会列出每个渠道分别缺什么。

**Q：报错只有一句话，看不到上游返回的原文？**
这是有意的。上游偶尔会返回一整页网页当错误（Cloudflare 的 502 页面有 6 KB），
把它原样发到群里既看不懂也刷屏。插件只提取页面标题、Cloudflare Ray ID，
再补一句该状态码的通俗解释。真要报障时，**这条消息正好可以直接转给站长**——
Ray ID 就是对方查日志要的凭据。

**Q：`--size` 填了没用，出图尺寸还是老样子。**
尺寸是「请求」而不是「命令」：插件会把它原样发出去，但最终听上游的。
有的站点或模型直接忽略它（实测某站的 `gpt-image-2` 固定返回约 1300×1200，
`agnes` 固定 1024×1024，填 `512x512` 和 `1792x1024` 结果一模一样）。
遇到这种情况只能换模型或换站点。

**Q：开了 `force_download` 反而下载失败，报 401。**
这是 0.1.2 之前的问题：插件会把渠道 Key 一起发给图床，而第三方域名上的
对象存储会因此判成鉴权失败——公开可下的图片反而下不来。0.1.2 起，
Key 只发给与 `api_base` 同源的地址，并会在被拒时去掉鉴权头重试。

**Q：图片发不出去 / 群友看不到图。**
多半是图片太大。百炼与大尺寸模型很容易返回 5 MB 以上的 PNG，
填一个具体尺寸（如 `--size 1328*1328`）通常就能解决。

**Q：多行提示词被折叠成一行。**
这是 AstrBot 指令解析的行为，不是插件的问题。

**Q：`--flag 的值里想带空格。**
目前不支持。`--negative 模糊,低质量` 这种用逗号分隔的写法可以正常工作。

---

## 已知限制

- **即梦官方 AK/SK 接口未实现。** 即梦除了第三方的 `sessionid` 方案，还有火山引擎
  官方的 `visual.volcengineapi.com`（`Action=CVSync2AsyncSubmitTask`，走 AK/SK 的
  HMAC-SHA256 网关签名）。本插件只实现了 `sessionid` 方案——官方签名路径没有
  可验证的参考实现，为避免「看起来实现了其实跑不通」，暂未加入。
- **Grok 官方接口一次只用第一张参考图**（JSON 的 `image` 是单个对象）；
  多张参考图请走中转站的 multipart 路径。
- 消息里的多行提示词会被 AstrBot 折叠成一行。
- `--flag 值` 的值不能含空格。

---

## 项目结构

```text
astrbot_plugin_multi_image/
├── main.py                  插件入口：指令、参数解析、结果发送、缓存与群渠道设置
├── metadata.yaml            插件元数据
├── _conf_schema.json        配置面板 schema
├── requirements.txt         依赖（httpx）
├── README.md
├── LICENSE
├── logo.png
└── providers/
    ├── __init__.py          渠道注册表、别名与工厂
    ├── base.py              数据模型、异常、抽象基类
    ├── http.py              URL 归一化、重试、图片响应解析
    ├── openai_image.py      OpenAI 协议 + 中转站多协议回退
    ├── gemini.py            Gemini 原生 / Imagen / OpenAI 兼容层
    ├── comfyui.py           工作流模板 + 提交 + 轮询 + 取图
    ├── seedream.py          火山方舟
    ├── grok.py              xAI（JSON 与 multipart 双路径）
    ├── jimeng.py            即梦（sessionid 轮换）
    └── dashscope.py         阿里云百炼（同步多模态 + 异步任务）
```

## 开发

代码遵循 [ruff](https://docs.astral.sh/ruff/) 检查与格式化，开发仓库另外附有：

- **离线测试套件**：用 `httpx.MockTransport` 模拟上游，
  **不会发出真实请求、不消耗任何额度**；
- **加载校验脚本**：按 AstrBot 的真实方式导入插件，检查类发现、指令注册、
  `_conf_schema.json` 解析，以及配置里不含任何个人凭据；
- **打包与合规校验**：核对插件市场规范（元数据字段、目录结构、体积上限、
  包内无缓存与配置文件）。

这些脚本不在发布包内，只随开发仓库提供。

## 参考资料与致谢

### 官方文档

- [AstrBot 插件开发指南](https://docs.astrbot.app/dev/star/plugin-new.html)
- [AstrBot 插件配置](https://docs.astrbot.app/dev/star/guides/plugin-config.html)
- [AstrBot 消息的发送](https://docs.astrbot.app/dev/star/guides/send-message.html)
- [AstrBot 插件市场 JSON 规范](https://docs.astrbot.app/dev/plugin-market/2026-06-27.html)
- [火山方舟图像生成 API](https://www.volcengine.com/docs/82379/1330310)
- [xAI 图像生成](https://docs.x.ai/developers/model-capabilities/images/generation)
- [阿里云百炼通义万相](https://help.aliyun.com/zh/model-studio/)

### 接口契约的核对来源

本插件的**接口契约**（端点、字段名、错误码语义）是结合各家官方文档与若干开源
AstrBot 插件实现交叉核对后确定的，核对对象包括即梦 `sessionid` 方案的
`/v1/images/generations` 与 `/v1/images/compositions` 走法、火山方舟
`/api/v3/images/generations` 的字段与错误码语义、xAI 的
`/v1/images/generations` 与 `/v1/images/edits` 分工。

核对只涉及公开可见的 API 调用方式，**没有复制任何第三方项目的代码**。
本插件的渠道抽象层、多协议回退、参数松弛重试、响应抠图等均为独立实现。

## 作者

本插件的**全部代码由 AI 编码代理 DeepSeek Harness 生成**——模型 `deepseek-v4-flash`，
在 DSH（DeepSeek Harness）运行时里完成，从渠道适配、协议回退到测试与打包脚本。

仓库由 [fujiwaerio](https://github.com/fujiwaerio) 创建与维护，需求提出、方案取舍与
最终验收由仓库维护者负责。

声明：正如dsh所说从生成验证测试都是dsh干的，我只是一个玩AI的小白，如果有什么想法或者帮助我和报错可以来 QQ 1107471744

## 许可证

本项目采用 [MIT 许可证](LICENSE)，版权归 DeepSeek Harness 所有。
任何人都可以自由使用、修改、分发，包括商用，只需保留版权声明与许可证文本。

`logo.png` 为 256×256、1:1 的占位图标，可自行替换成自己的设计
（保持同样规格即可，AstrBot 插件市场会读取它）。
