## 代码结构

```
XSkill/
├── eval/
│   ├── infer_api.py              # Main inference entry point
│   ├── infer_api_utils.py        # Utility functions for inference pipeline
│   ├── run_api_exskill.sh        # Reference run script
│   ├── configs/
│   │   └── tool_configs.yaml     # Per-tool runtime configuration
│   ├── engine/                   # API calling, tool dispatch, context management
│   ├── exskill/                  # Experience & skill learning core
│   │   ├── experience_critique.py
│   │   ├── experience_manager.py
│   │   ├── experience_retriever.py
│   │   ├── skill_builder.py
│   │   ├── trajectory_summary.py
│   │   └── multimodal_analysis.py
│   ├── tools/                    # Tool implementations
│   │   ├── code_interpreter.py
│   │   ├── web_search.py
│   │   ├── image_search.py
│   │   ├── visit.py
│   │   └── zoom.py
│   ├── prompts/                  # Prompt templates
│   ├── search/                   # Search tree & node structures
│   └── utils/                    # Shared utilities
├── memory_bank/                  # Experience library & skill document (created at runtime)
├── output/                       # Per-sample inference outputs
├── logs/                         # Run logs
└── requirements.txt
```

---

## 安装

**Python 3.11** is recommended.

```bash
git clone https://github.com/XSkill-Agent/XSkill.git
cd XSkill
pip install -r requirements.txt
```


---

## 模型推理配置 .env
 
目前所有的api key写在.env中，运行代码前补充这些key后，运行：source ../.env.

```bash
export OPENAI_API_KEY=""    # 模型api key
export OPENAI_BASE_URL=""   # openai 兼容的地址
export REASONING_MODEL_NAME="o4-mini"  # 推理模型（主agent推理、工具调用决策、最终答案生成）
export VERIFIER_MODEL_NAME="gpt-4o"    # LLM as Judge 验证模型
export EXPERIENCE_MODEL_NAME="gpt-4o"  # 经验和skill生成模型（任务分解、经验重写、轨迹总结、经验生成、skill生成合并优化适应）
export EXPERIENCE_EMBEDDING_MODEL="text-embedding-3-small"  # 嵌入模型
export SERPAPI_KEY=""    # web_search和image_search tool
export JINA_API_KEY=""   # visit tool
``` 

## `eval/configs/tool_configs.yaml` 工具使用配置

```yaml
# visit tool — webpage content fetching
visit:
  max_content_length: 150000   # Max characters to extract per page
  timeout: 120                 # HTTP request timeout (seconds)
  api_key: ""                  # Optional: API key for VLM-based page summarization
  api_endpoint: ""             # Optional: endpoint for the above
  model_name: ""               # Optional: model name for the above

# image_search tool — reverse/visual image search
image_search:
  imgbb_api_key: ""            # Required: ImgBB API key for image hosting
  max_results: 5               # Max search results to return
  search_image_max_pixels: 1000000
  search_image_quality: 85
```

## 数据处理
Benchmark 数据文件（通过 `--input-file` 参数传递）是包含样本对象列表的 JSON 文件。每个样本都具有以下字段：

```json
[
  {
    "doc_id或question_id": "sample_001",
    "problem或question": "What is shown in <image>? Describe the object in detail.",
    "images": ["relative/path/to/image.jpg"],
    "solution": "A red bicycle parked against a wall."
  },
  ...
]
```

说明：
- 如果 `problem` 中存在 `<image>` 占位符，则会按照出现的顺序插入图片。
- 如果没有 `<image>` 占位符但 `images` 字段不为空，则会将列出的所有图片传递给模型。
- 仅包含文本内容的样本（没有 `images` 字段且没有 `<image>` 占位符）也是被支持的。
- `solution` 字段可以省略；没有基准答案的样本将获得分数 `0.0`。

---

## 说明

这个分支将代码和简短的文档存放在“mmagent/”目录中，而大型本地数据和生成的文件则存放在 Git 仓库之外：

```text
XSkill/
├── mmagent/          # Git-tracked code repository
├── benchmark/        # local demo/full benchmark data, not tracked
├── benchmark_full/   # downloaded/processed benchmark data, not tracked
├── logs/             # run logs, not tracked
├── memory_bank/      # generated experiences and skills, not tracked
│   ├── TIR-Bench/
│   ├── VisualToolBench/
│   └── ...
├── output/           # run outputs, not tracked
├── .env              # local API keys, not tracked
└── .env.local        # optional local API keys, not tracked
```

所有运行脚本都会通过“DATA_ROOT”或“XSKILL_DATA_ROOT”来解析数据根目录。默认情况下，数据根目录是“mmagent/”的父目录。

对于完整的benchmark 测试脚本，默认情况下memory是按benchmark进行划分的，例如“memory_bank/TIR-Bench/experiences.json”和“memory_bank/TIR-Bench/SKILL.md”。只有在您有意要重复使用或转移一个内存库时，才需要重写“MEMORY_ROOT”。

```bash
cd /mnt/f/XSkill/mmagent

# 若数据存储于其他位置，则可选择进行手动覆盖：
export DATA_ROOT=/mnt/f/XSkill
# or:
export XSKILL_DATA_ROOT=/mnt/f/XSkill
```

代码根目录: F:\XSkill\mmagent
数据根目录: F:\XSkill
也就是脚本默认会：

读取:  F:\XSkill\benchmark_full\...
写入:  F:\XSkill\output\...
写日志: F:\XSkill\logs\...
使用:  F:\XSkill\memory_bank\...
读取外层: F:\XSkill\.env / .env.local
同时也支持你手动改路径，不写死：

```bash
export DATA_ROOT=/mnt/f/XSkill 或
export XSKILL_DATA_ROOT=/mnt/f/XSkill

export BENCH_ROOT=/some/path/processed
export OUTPUT_ROOT=/some/path/output
export LOG_ROOT=/some/path/logs
export MEMORY_ROOT=/some/path/memory_bank/test
```
以后运行要进入 mmagent：

```bash
cd /mnt/f/XSkill/mmagent
source ../.env
bash scripts/run_full_benchmark_test.sh
```
或者 **训练 + 测试**：

```bash
cd /mnt/f/XSkill/mmagent
source ../.env
bash scripts/run_full_benchmark_train_test.sh
```

测试单独的benchmark:
```bash
BENCHMARK=TIR-Bench bash scripts/run_full_benchmark_train_test.sh
```
会使用：/mnt/f/XSkill/memory_bank/TIR-Bench/

换成：
```bash
BENCHMARK=AgentVista bash scripts/run_full_benchmark_train_test.sh
```
会使用：/mnt/f/XSkill/memory_bank/AgentVista/

如果做论文里的 OOD / transfer，比如用 MMSearch-Plus 的 memory 测 MMBrowseComp，可以显式覆盖：
```bash
BENCHMARK=MMBrowseComp \
MEMORY_ROOT=/mnt/f/XSkill/memory_bank/MMSearch-Plus \
bash scripts/run_full_benchmark_test.sh
```


## 数据流
```bash
benchmark JSON + image
   ↓
agent 推理
   ↓
样本目录:
   original_image.jpg 原始图片
   traj.jsonl 轨迹数据，里面包括：初始题目和 ground truth、每轮 assistant 输出、工具调用、工具参数、工具返回结果、最终回答
   metrics.json 指标
   injected_system_prompt.txt 注入系统提示词，会包含：base agent 指令、检索到的 experience、适配后的 skill
   ↓
trajectory_summary.py 总结轨迹:
   扫描图片
   生成 caption
   把 <image> / tool image 引用替换成 caption
   把 traj.jsonl + tool result + caption 拼成 exp_summary_prompt.txt
   ↓
LLM 总结轨迹:
   exp_summary_resp.txt
   ↓
experience_critique.py 生成经验操作 (题目+ground truth+rollout摘要+本题检索使用的旧经验):
   exp_intra_prompt.txt
   exp_intra_resp.txt
   exp_items.json
   ↓
experience_manager.py 合并进experiences.json:
   add: 新增经验、计算新经验embedding、计算相似度查找旧经验
        相似LLM合并新旧经验，不相似加入experiences.json，相似度阈值SIMILARITY_THRESHOLD 0.7
   modify: 修改已有经验候选、修改对应旧经验
   重新编号E0/E1/E2...
   更新 memory_bank/test/experiences.json
   更新 ExperienceRetriever embedding缓存
   ↓
skill_builder.py:
   skill_raw.md
   更新 memory_bank/test/SKILL.md
   ↓
snapshots/batch_000:
   保存该 batch 后的 memory 副本
```

如果经验库比较大，还会做全局 refine：
```bash
refine_experience_library:
    把整个经验库给 LLM
      ↓
    让 LLM 判断哪些可以 merge / delete
      ↓
    应用 merge/delete
      ↓
    重新编号
```

## 训练测试数据集划分
论文主实验不是把 4 个 benchmark 的 train 全部混在一起训一个统一 memory 后再测，而是 每个 benchmark 分别积累自己的 skill/experience，再在该 benchmark 的 test split 上测试。


下载好benchmark数据后运行scripts/prepare_xskill_benchmarks.py脚本，按照论文中的数据划分标准进行划分和预处理：

工具使用：Code是执行代码、Search-W是搜索网页（输入query）获取内容、Search-I是输入文字或图片搜索获取内容、visit是访问网页（输入URL）提取文本内容。

(1) 视觉智能体工具使用:
- VisualToolBench: Code、Search-W、visit
- TIR-Bench: Code

(2) 多模态搜索：
- MMSearch-Plus: Code、Search-W、Search-I、visit
- MMBrowseComp: Code、Search-W、Search-I、visit

(3) 综合：
- AgentVista: Code、Search-W、Search-I、visit

## 关键参数理解
- BENCHMARK：测试benchmark的名称
- MAX_SAMPLES_TEST：
- NUM_WORKERS：并行跑的worker数量
- ROLLOUTS_PER_SAMPLE：每个样本生成的rollout数量（4），这是为了计算 average@4 和pass@4；如果设成 2，就只能得到 average@2/pass@2。测试结果会写到：
```bash
output/full_benchmark/TIR-Bench/test/metrics_at_k.json
output/full_benchmark/TIR-Bench/test/average_pass_at_k.json
```
- EXPERIENCE_LARGE_BATCH：累计多少个 rollout 后做一次经验/技能更新
- 

## 存在的问题
1. 问题：检索到的经验是选topk吗，如果检索到的经验相似度很低呢？
  * 流程是：当前 query --> 生成 query embedding --> 和所有 experience embedding 算 cosine similarity --> 过滤掉 similarity < min_similarity 的经验 --> 按 similarity 降序排序 --> 取前 top_k 条。目前top_k=3，如果开了--experience-retrieval-decomposition，会执行子任务分解，每个任务检索top-k。如果相似度很低怎么办？关键在这个参数：--experience-retrieval-min-similarity，代码默认是：default=0.0，也就是说：只要 cosine similarity >= 0.0，就可能被选中，所以当前默认逻辑下，即使 top-k 的相似度很低，只要它大于等于 0，也会被检索出来。这就是你之前指出的问题：相似不代表有用，低相似度经验也可能误导推理。

  当前代码的实际问题：当前 tt_decomposition_info.txt 里只记录了检索到的 experience ID，但没有把相似度分数完整写出来。所以你不太容易从输出文件判断：E3 是 0.82 检索到的，还是 0.05 检索到的。如果你想分析检索质量，建议后面把 similarity score 也写进 retrieval_info。当前代码里只在 debug logger 里有类似信息，不一定会落盘。

2. 问题：大模型喜欢走捷径，我给了它skill和经验，他很自信觉得能够答对，直接不用工具进行图文推理导致回答错误。

## 修改
1. 0517: 增加了断点续跑，中途断了之后如果样本目录已有指标值，会自动跳过，不会跑已完成样本。