对上面的方案进行评审

### **评审结论：总体方向正确，但不能原样实施；建议保留“页型标记 + 结构感知切分 + 检索降权”主线，修正目录页判定、表格处理、chunk 长度和检索权重的几个关键风险。**

## 一、总体评价

上面的方案解决了三个真实问题：

1. 目录页与正文混在一起，导致关键词密集的目录页被误召回。
2. 800 字 chunk 与 Qwen 只截取 600 字不匹配，可能把关键结论截断。
3. 图表、表格、正文被统一当成普通文本，造成数字密集型噪声。

“先分类、再切分、最后在检索层按类型处理”的方向优于简单地把目录页删除或把所有 chunk 统一改成 500 字。尤其是保留 `page_type`，能支持后续调参、错误分析和 B 榜文档级召回。

不过，原方案中有几处需要调整，否则可能出现**误删正文、召回率下降、表格解析失败、索引分数失真**的问题。

---

## 二、逐项评审

### 1. 目录页：应标记和降权，但不建议完全跳过

原方案同时提出两种思路：

- 目录页保留并标记 `toc`
- A 榜或证据检索时直接跳过

更稳妥的做法是：**保留目录 chunk，但默认不让它进入证据候选；文档级召回仍可以使用。**

原因是目录页有两种用途：

- 对具体题目作答：目录通常不是事实证据，应该排除。
- B 榜文档召回：目录包含文档主题、公司名、章节名，可以帮助判断文档是否相关。

因此建议分成两套索引：

```text
document_index
├── body/table/chart  # 用于证据检索
└── toc/cover         # 用于文档级召回或辅助定位
```

检索时不要仅用一个统一的 `rank_chunks`：

```python
def retrieve_evidence(...):
    candidates = [
        c for c in chunks
        if c["page_type"] not in {"toc", "cover"}
    ]
```

而文档级召回可以使用：

```python
def retrieve_documents(...):
    candidates = all_chunks
    # toc 可参与，但降低权重
```

这样不会让目录进入最终证据，也不会丢失它对 B 榜的主题提示价值。

---

### 2. 目录页识别规则过于简单，需要组合特征

原方案主要依赖：

```python
toc_lines / len(lines) > 0.4
```

以及：

```python
图表 + 数字
```

这不够稳。研究报告正文中也可能出现：

- 大量编号标题
- 图表标题
- 章节列表
- 参考文献
- 资料来源
- 多个带页码的引用

单一的“点线 + 页码”规则可能漏掉没有点线的目录，也可能误判正文。

建议采用**特征打分，而不是单条件判断**。

```python
def toc_score(text: str, page_no: int) -> float:
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if not lines:
        return 0.0

    score = 0.0

    toc_line_count = sum(
        bool(re.search(r'.{2,}(?:\.{2,}|…{2,}|·{2,})\s*\d{1,4}$', x))
        for x in lines
    )

    numbered_heading_count = sum(
        bool(re.match(r'^(?:[一二三四五六七八九十]+[、.]|\d+(?:\.\d+)*[、.])', x))
        for x in lines
    )

    chart_index_count = sum(
        bool(re.search(r'^(?:图|表|图表)\s*\d+', x))
        for x in lines
    )

    short_line_count = sum(len(x) <= 80 for x in lines)

    if toc_line_count >= 3:
        score += 0.5

    if toc_line_count / max(len(lines), 1) >= 0.25:
        score += 0.2

    if chart_index_count >= 4:
        score += 0.2

    if short_line_count / max(len(lines), 1) >= 0.7:
        score += 0.1

    # 目录页一般没有连续长正文
    if max(map(len, lines), default=0) < 180:
        score += 0.1

    return min(score, 1.0)
```

然后使用等级，而不是简单的 True/False：

```python
score = toc_score(text, page_no)

if score >= 0.75:
    page_type = "toc"
elif score >= 0.50:
    page_type = "toc_candidate"
else:
    page_type = "body"
```

推荐处理方式：

| 类型 | 证据检索 | 文档召回 | 处理 |
|---|---:|---:|---|
| `toc` | 排除 | 低权重保留 | 整页保留 |
| `toc_candidate` | 极低权重 | 保留 | 进入人工抽样检查 |
| `body` | 正常 | 正常 | 正常切分 |

这里的 `toc_candidate` 很重要，可以降低规则误判带来的损失。

---

### 3. “图表页”不应该按数字行占比直接判定

原方案中有类似：

```python
numeric_lines / len(lines) > 0.6
```

这个规则容易误伤：

- 财务报表
- 统计表格
- 含大量年度数据的正文
- 保险费率表
- 合同金额和期限表

这些内容虽然数字多，但正是高价值证据。

更好的做法是区分：

```text
page_type
├── body
├── table
├── figure
├── mixed
├── toc
└── cover
```

其中 `mixed` 不要强行判为图表页。只有当页面同时满足以下条件时，才判定为 `figure`：

- 数字或短标签比例高
- 有明显图表标题
- 文本句子比例低
- 没有可靠的表格结构
- 正文字符密度较低

```python
def classify_figure(page_text: str, has_table: bool) -> bool:
    if has_table:
        return False

    lines = [x.strip() for x in page_text.splitlines() if x.strip()]
    if not lines:
        return False

    figure_title = any(
        re.match(r'^(图|图表)\s*\d+', x)
        for x in lines
    )

    numeric_like = sum(
        bool(re.fullmatch(r'[\d\s%.,+\-()]+', x))
        for x in lines
    )

    sentence_like = sum(
        len(x) >= 30 and re.search(r'[。；！？]', x) is not None
        for x in lines
    )

    return (
        figure_title
        and numeric_like / len(lines) >= 0.4
        and sentence_like / len(lines) <= 0.3
    )
```

另外，“图表坐标轴数字过滤”不能简单把所有纯数字行都删掉。应当：

- 删除短、纯数字、无单位的刻度行；
- 保留带年份、百分比、金额单位的行；
- 保留靠近图表标题或数据标签的数值；
- 对删除结果进行日志记录。

例如：

```python
def is_axis_noise(line: str) -> bool:
    line = line.strip()

    if not line:
        return True

    # 纯数字短行，可能是坐标轴刻度
    if len(line) <= 12 and re.fullmatch(r'[\d\s.,+\-]+', line):
        return True

    # 含单位或年份，不要直接删除
    if re.search(r'(亿|万|万元|美元|欧元|人民币|%|同比|202\d)', line):
        return False

    return False
```

更稳妥的原则是：**宁可降权，也不要在预处理阶段直接丢失不确定信息。**

---

### 4. `find_tables()` 不应作为唯一表格方案

原方案把 `find_tables()` 作为结构化表格提取的主要手段，这个方向可以尝试，但不能假设所有图表都能被它识别。

它更适合：

- 有明显表格线的表格
- 规则排列的多列表格
- 财报和合同中的结构化表格

它不一定适合：

- 柱状图
- 折线图
- 饼图
- 无边框表格
- PDF 中由文本块模拟的表格
- 扫描图片表格

因此必须设计三级策略：

```text
优先级 1：find_tables()
优先级 2：文本块坐标恢复
优先级 3：原始文本兜底
```

建议保留三类产物，而不是只生成一个 table text：

```json
{
  "chunk_type": "table",
  "table_id": "doc-12-p4-t1",
  "page_no": 4,
  "bbox": [50, 120, 550, 420],
  "headers": ["年份", "收入", "增速"],
  "rows": [
    ["2024", "100", "5.2%"],
    ["2025", "110", "10.0%"]
  ],
  "text": "年份 | 收入 | 增速\n2024 | 100 | 5.2%\n2025 | 110 | 10.0%",
  "source": "pymupdf_find_tables"
}
```

同时保留 `bbox` 和原始行列结构，这对后续判断非常重要。

---

### 5. `table` 不应统一加权 1.2

原方案建议：

```python
table_weight = 1.2
```

这会有风险，因为不是所有题都适合表格优先：

- 观点题通常依赖正文解释；
- 因果判断依赖上下文；
- 表格中的数字可能没有主语和单位；
- 表格可能是目录或摘要中的低质量表格。

建议改成**题型和领域感知权重**：

| 领域/题型 | body | table | figure | toc |
|---|---:|---:|---:|---:|
| 财务报表 + 数值比较 | 1.0 | 1.3 | 0.8 | 0.05 |
| 金融合同 | 1.0 | 1.2 | 0.7 | 0.05 |
| 保险条款 | 1.1 | 1.2 | 0.7 | 0.05 |
| 监管法规 | 1.2 | 1.0 | 0.5 | 0.05 |
| 行业研报 | 1.0 | 1.0 | 0.8 | 0.05 |

并且应当把类型权重放在最后，而不是完全覆盖 BM25：

```python
final_score = (
    0.75 * normalized_bm25
    + 0.15 * option_entity_score
    + 0.10 * numeric_context_score
) * type_weight
```

最好不要把 `table` 直接放到 top 位置，而是使用多样性约束：

```text
top-k 中至少保留：
- 每个关键实体的一个候选
- 每个关键数值的一个候选
- 正文和表格各至少一个候选（如果存在）
```

否则表格权重过高，可能重新造成“数字碎片霸榜”。

---

### 6. 500～600 字不能作为硬目标，应该使用“语义单元 + 最大长度”

原方案将正文 chunk 上限从 800 改成 550 或 600，这一方向有依据，但不能简单地把所有 chunk 压到 500～600 字。

风险包括：

- 某些完整段落本身只有 100 字，强行合并会引入主题污染；
- 某些关键句在 600 字边界处被拆开；
- 研报的标题、观点、数据、结论可能跨多个文本块；
- 保险条款和法规条款通常需要保留整个条款，不能任意截断。

建议使用三级切分策略：

#### 一级：天然语义单元

优先按以下边界切分：

- 标题
- 段落
- 条款
- 表格
- 图注
- 资料来源

#### 二级：句子合并

同一语义单元超过上限时，按句号、分号、问号、感叹号切分。

#### 三级：硬切兜底

只有长句异常时，才按字符硬切，并保留少量 overlap。

推荐参数：

```python
TARGET_CHUNK_LEN = 450
MAX_CHUNK_LEN = 700
MIN_CHUNK_LEN = 80
OVERLAP_LEN = 60
```

这比固定 550 更稳：

- 目标长度约 450；
- 允许自然段达到 700；
- Qwen 提取 600 字时仍尽量覆盖完整句子；
- 超长段落才切分。

对于关键句，建议保存一个更短的 `evidence_text`：

```json
{
  "text": "完整 chunk 原文",
  "evidence_text": "围绕命中实体和数值的 300-600 字上下文"
}
```

不要把原始 chunk 本身截断后覆盖掉。检索使用完整文本，喂给 Qwen 使用证据窗口。

---

### 7. 目录页问题不是只靠 chunk 解决，还要修正“证据可信度”

即便目录页被降权，仍可能进入候选。因此 reasoner 还应该检查证据类型：

```python
def evidence_reliability(chunk):
    page_type = chunk.get("page_type")

    if page_type == "cover":
        return 0.1
    if page_type == "toc":
        return 0.1
    if page_type == "figure":
        return 0.6
    if page_type == "table":
        return 0.9
    return 1.0
```

但不要把 `toc` 直接判为 refute 或 support。正确的语义是：

```text
toc 证据只能用于定位，不能独立支持事实性结论
```

因此 reasoner 应输出：

```json
{
  "verdict": "insufficient",
  "confidence": 0.15,
  "reason": "仅命中目录标题，未找到正文事实"
}
```

这比简单降低分数更安全。

---

## 三、建议的最终数据结构

推荐把原先的简单 chunk 扩展为以下结构：

```json
{
  "chunk_id": "pack2_text09-p04-b02",
  "doc_id": "pack2_text09",
  "page_no": 4,
  "page_type": "body",
  "chunk_type": "paragraph",
  "section_path": [
    "自主IP驱动",
    "芯原股份"
  ],
  "text": "完整原始文本……",
  "evidence_text": "命中实体和数值附近的上下文……",
  "char_len": 486,
  "entities": ["芯原股份", "博通"],
  "numbers": [
    {
      "raw": "千亿",
      "value": null,
      "unit": "亿元",
      "sign": null
    }
  ],
  "source": {
    "method": "pymupdf_blocks",
    "bbox": [40, 120, 560, 360]
  },
  "retrieval": {
    "allow_evidence": true,
    "doc_recall_weight": 1.0,
    "evidence_weight": 1.0
  }
}
```

这里有两个关键点：

1. `entities`、`numbers` 由规则抽取，仅作为检索辅助，不能替代原文判断。
2. `allow_evidence` 与 `doc_recall_weight` 分开，解决目录页“不能作证据但可以帮助文档召回”的问题。

---

## 四、推荐实施顺序

不建议一次性同时改目录识别、表格提取、分词、reasoner、Qwen Prompt。应该使用隔离实验，否则无法知道提升来自哪里。

### 第一阶段：低风险清洗实验

只做：

- `page_type` 标注；
- `toc`、`cover` 证据检索排除；
- 正文按段落/句子切分；
- 保存旧版和新版 chunk；
- 不改变 reasoner 和 Qwen Prompt。

验证指标：

- ev5、ev7 是否不再进入最终证据；
- res_a_020 的封面页是否不再作为有效证据；
- 20 题证据召回率是否下降；
- top-k 中实体和数值覆盖率是否提高。

### 第二阶段：chunk 结构实验

只改变：

- `TARGET_CHUNK_LEN`
- `MAX_CHUNK_LEN`
- 是否保留 overlap
- 是否按标题和句子切分

建议至少比较：

| 版本 | target | max | overlap |
|---|---:|---:|---:|
| baseline | 600 | 800 | 0 |
| candidate-A | 450 | 650 | 50 |
| candidate-B | 500 | 700 | 60 |

不要只看总准确率，还要看错误类型是否退化。

### 第三阶段：表格和图表实验

先在有明确问题的 research 文档上验证：

- `find_tables()` 成功率；
- 表格行列是否正确；
- 数值、单位、年份是否保留；
- 图表坐标轴噪声是否减少；
- 表格题召回是否提高。

只有验证通过后，再推广到五个领域。

### 第四阶段：检索评分实验

把 page type 权重做成配置，而不是硬编码：

```yaml
retrieval:
  body: 1.0
  table: 1.1
  figure: 0.7
  toc: 0.05
  cover: 0.0
```

逐组跑实验，避免“table 固定 1.2”导致新的数字噪声问题。

---

## 五、必须增加的评估指标

仅看最终 15/20 或 20/20 不够，因为 chunk 优化可能偶然改变答案。建议建立以下指标。

### 1. 目录污染率

```text
toc_contamination_rate =
进入最终证据的 toc chunk 数 / 最终证据 chunk 总数
```

目标：证据级别为 0%。

### 2. 证据召回率

对每个选项统计：

```text
evidence_recall =
召回黄金证据的选项数 / 需要证据的选项数
```

特别关注：

- 韩国段落
- 宇信科技营收段落
- 博通与芯原的实体区分
- 2026 年电动车销量段落

### 3. 证据纯度

```text
evidence_precision =
真正支持或反驳该选项的 chunk 数 / 被选作该选项证据的 chunk 总数
```

这能衡量目录、封面、其他实体证据是否污染候选。

### 4. 关键字段覆盖率

对每个选项检查：

- 核心实体
- 年份
- 数值
- 单位
- 符号
- 方向词

例如：

```json
{
  "option": "C",
  "entity_hit": true,
  "year_hit": true,
  "value_hit": true,
  "unit_hit": false,
  "direction_hit": false
}
```

`020-C` 会显示实体命中，但核心数值未命中，这是比单一 BM25 分数更有用的诊断。

### 5. 证据截断损失率

统计：

```text
关键结论被 evidence_text 截断的题数 / 全部题数
```

这是验证 800 → 600 是否真的有效的关键指标。

### 6. 误杀率

目录/图表过滤不能只看噪声是否减少，还要看有效证据是否被过滤：

```text
有效证据被过滤率 =
被过滤的黄金证据数 / 黄金证据总数
```

目标是：

- 目录证据污染率接近 0；
- 有效正文证据被误删为 0；
- 表格证据误删率尽可能低。

---

## 六、最终建议

我建议采用下面这个修正版，而不是直接照搬原方案：

```text
PDF 页面
  ↓
页面分类：cover / toc / body / table / figure / mixed
  ↓
保留全部原始内容，但写入 page_type
  ↓
正文：段落优先，句子切分，target 450，max 650~700
表格：find_tables 优先，文本坐标恢复兜底
图表：保留标题、图注、带单位数值，过滤坐标轴噪声
目录：整页保留，但 evidence 检索排除
封面：保留元数据，但 evidence 检索排除
  ↓
双索引
  ├── document index：toc 可低权参与
  └── evidence index：仅 body/table/可靠 figure
  ↓
BM25 + 实体/数值/单位/方向特征
  ↓
选项感知精排
  ↓
reasoner 类型可信度检查
  ↓
Qwen 逐选项判断
```

### **最终评审意见**

- **保留**：页型标记、目录降权、正文结构切分、表格独立处理、检索层控制。
- **修改**：目录页不要物理删除；不要用单一比例判断图表页；不要把所有 chunk 固定压到 500～600 字；不要固定给表格统一加 1.2 权重。
- **新增**：`toc_candidate` 和 `mixed` 类型、双索引、证据窗口、字段覆盖率、误杀率评估。
- **暂缓**：全面接入 `find_tables()`。先在已知出错文档上验证，再扩展到全部领域。

这套修正版能同时解决 ev5/ev7 的目录污染、ev7 类图表目录干扰、020 的封面误证据，以及 001 中正文关键句被截断的问题，同时避免因过度过滤导致新的召回损失。

*内容由 AI 生成仅供参考*