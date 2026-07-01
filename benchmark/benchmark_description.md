# Multimodal Agent Benchmark Notes

This file is a compact reference for the benchmarks currently used by MMAgent/XSkill and several newer 2026 benchmarks that are close enough to consider adapting. The reported scores below are headline numbers from the original papers, project pages, or dataset cards; protocols differ, so they should be read as evidence of benchmark difficulty rather than directly comparable leaderboard values.

## Quick Setup

For Hugging Face datasets:

```powershell
pip install -U "huggingface_hub[cli,hf_transfer]" hf_transfer hf_xet
hf auth login

$env:HF_HUB_ENABLE_HF_TRANSFER="1"
$env:HF_XET_HIGH_PERFORMANCE="1"
```

Mainland China users can optionally use:

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
```

`hf download --local-dir` preserves the repository structure and supports incremental downloads. `hf_transfer` can speed up large-file downloads through parallel chunk transfer.

## Benchmarks Already Supported

| Benchmark | First public version | Main capability tested | Public headline result | MMAgent adapter status |
|---|---:|---|---|---|
| VisualToolBench | 2025-10-14 | Tool-enabled image perception, transformation, and reasoning | Best reported model GPT-5-think reaches only **18.68% pass rate** | Supported |
| TIR-Bench | 2025-11-03 | Agentic thinking-with-images with image-processing/code tools | Best reported performance is about **46%**; best non-agentic model Gemini-2.5-Pro is **28.9%** | Supported |
| MMSearch-Plus | 2025-08-29 | Provenance-aware multimodal browsing and visual search | Strongest reported system reaches **36.0% accuracy**; Set-of-Mark gives gains up to **+3.9 points** | Supported |
| MM-BrowseComp | 2025-08-14 | Multimodal deep browsing over text, image, and video web evidence | OpenAI o3 with tools reaches only **29.02% accuracy** | Supported |
| AgentVista | 2026-02-26 | Realistic long-horizon multimodal tool use | Gemini-3-Pro with tools reaches **27.3% overall accuracy** | Supported |

### What These Scores Suggest

The current public results show a consistent pattern: generic multimodal models do not solve these tasks by static image understanding alone. Stronger results usually require tool execution, web search, image search, browsing, code-based image operations, or process-level planning. Even then, the best reported scores remain low: roughly 18.68% on VisualToolBench, 27.3% on AgentVista, 29.02% on MM-BrowseComp, 36.0% on MMSearch-Plus, and about 46% on TIR-Bench.

This is why these benchmarks are suitable for evaluating memory, experience reuse, skill induction, recovery, and tool-plan selection rather than only final-answer prompting.

## Current Benchmark Downloads

### VisualToolBench

Links:
- Dataset: https://huggingface.co/datasets/ScaleAI/VisualToolBench
- Paper: https://arxiv.org/abs/2510.12712

```powershell
hf download ScaleAI/VisualToolBench `
  --repo-type dataset `
  --local-dir benchmark/raw/VisualToolBench
```

The full dataset has 1,204 samples: 603 single-turn and 601 multi-turn. The current MMAgent split uses the single-turn hybrid subset.

### TIR-Bench

Links:
- Dataset: https://huggingface.co/datasets/Agents-X/TIR-Bench
- Paper: https://arxiv.org/abs/2511.01833
- EvalScope note: https://evalscope.readthedocs.io/en/v1.6.1/benchmarks/tir_bench.html

```powershell
hf download Agents-X/TIR-Bench `
  --repo-type dataset `
  --local-dir benchmark/raw/TIR-Bench
```

The full benchmark contains 1,215 examples across 13 tasks. The current split focuses on the XSkill paper subset: `refcoco`, `maze`, `instrument`, `ocr`, and `contrast`.

### MMSearch-Plus

Links:
- Dataset: https://huggingface.co/datasets/Cie1/MMSearch-Plus
- Project: https://mmsearch-plus.github.io/
- Paper: https://arxiv.org/abs/2508.21475

```powershell
hf download Cie1/MMSearch-Plus `
  --repo-type dataset `
  --local-dir benchmark/raw/MMSearch-Plus
```

The dataset is encrypted to reduce contamination. Use the official `decrypt_after_load.py` after download.

### MM-BrowseComp

Links:
- GitHub: https://github.com/MMBrowseComp/MM-BrowseComp
- HF dataset: https://huggingface.co/datasets/mmbrowsecomp/MMBrowseComp
- XSkill-friendly mirror: https://huggingface.co/datasets/DjangoJungle/MMBrowseComp
- Paper: https://arxiv.org/abs/2508.13186

```powershell
hf download mmbrowsecomp/MMBrowseComp `
  --repo-type dataset `
  --local-dir benchmark/raw/MMBrowseComp
```

For direct MMAgent-style processing, prefer the XSkill-friendly mirror:

```powershell
hf download DjangoJungle/MMBrowseComp `
  --repo-type dataset `
  --local-dir benchmark/raw/MMBrowseComp_XSkillFormat
```

The current MMAgent setup treats MM-BrowseComp as OOD test-only data.

### AgentVista

Links:
- Dataset: https://huggingface.co/datasets/Warrieryes/AgentVista
- Project: https://agentvista-bench.github.io/
- GitHub: https://github.com/hkust-nlp/AgentVista
- Paper: https://arxiv.org/abs/2602.23166

```powershell
hf download Warrieryes/AgentVista `
  --repo-type dataset `
  --local-dir benchmark/raw/AgentVista
```

AgentVista has 209 samples across 7 domains and 25 subdomains. Each sample contains one or more images, a problem, an answer, and domain metadata.

## Stratified Train/Val/Test Subsets

`build_benchmark_subset.py` generates non-overlapping `train.json`, `val.json`, and `test.json` files. The current stratification keys are:

- `TIR-Bench`: `task`
- `AgentVista`: `domain`, `subdomain`
- `MMSearch-Plus`: `category`, `subtask`, `difficulty`
- `MMBrowseComp`: `category`, `subtask`, `level`
- `VisualToolBench`: `prompt_category`, `eval_focus`, `turncase`

For benchmarks with official train/test files, validation is held out from the train split rather than the test split.

Example:

```bash
python benchmark/build_benchmark_subset.py \
  --processed-root benchmark/processed \
  --output-root benchmark/smoke_val \
  --benchmarks TIR-Bench \
  --train-per-benchmark 4 \
  --val-per-benchmark 2 \
  --test-per-benchmark 4 \
  --seed 42
```

Use `--train-per-benchmark -1` or `--test-per-benchmark -1` to keep all remaining train/test candidates after validation holdout.

## 2026 Benchmarks Worth Considering

| Benchmark | First public version | Size / scope | Public headline result | Fit for MMAgent |
|---|---:|---|---|---|
| BrowseComp-V3 | 2026-02-13 | 300 multimodal browsing questions | SOTA models reach only about **36% accuracy** | Very high |
| VisBrowse-Bench | 2026-03-17 | 169 visual-native browsing VQA tasks | Claude-4.6-Opus **47.6%**, o3-deep-research **41.1%** | Very high |
| Agentic-MME | 2026-04-03 | 418 tasks, 6 domains, 3 difficulty levels, 2,000+ checkpoints | Gemini3-pro **56.3% overall**, **23.0%** on Level-3 | Very high |
| MC-Search | 2026-03-01 | 3,333 multimodal agentic search examples, avg. 3.7 hops | Paper emphasizes process failures more than a single headline score | Medium-high |
| InterLV-Search | 2026-05-08 | 2,061 interleaved language-vision search examples | Best model below **50% overall accuracy** | Medium-high |
| MERRIN | 2026-04-15 | Noisy open-web evidence retrieval across text/image/video/audio | Average agent accuracy **22.3%**, best **40.1%** | Medium |
| EpiBench | 2026-04-07 | Multi-turn scientific research workflows over papers, figures, and tables | Leading model **29.23%** on the hard split | Medium |

### Recommended Adaptation Priority

1. **BrowseComp-V3**  
   Closest to MM-BrowseComp and MMSearch-Plus. It keeps the same core idea of multimodal deep browsing, but adds stronger process evaluation and 300 curated tasks. This is the cleanest next benchmark for showing generalization.

2. **VisBrowse-Bench**  
   Strong fit for the existing `web_search + image_search + visit + code_interpreter` tool stack. It emphasizes visual-native web evidence rather than treating visual evidence as only the input image.

3. **Agentic-MME**  
   A good broad generalization benchmark because it explicitly evaluates visual expansion, knowledge expansion, tool use, web search, and process checkpoints. It is less tied to one browsing style than BrowseComp-like datasets.

4. **InterLV-Search / MC-Search**  
   Useful if the goal is to test structured multi-hop retrieval and planning. They may need more adapter work because they involve interleaved evidence chains or local knowledge bases.

5. **MERRIN / EpiBench**  
   Good for stress testing noisy evidence selection and long research workflows, but they are less format-aligned with the current benchmark converters.

## 2026 Candidate Downloads

### BrowseComp-V3

Links:
- GitHub: https://github.com/Halcyon-Zhang/BrowseComp-V3
- Dataset: https://huggingface.co/datasets/Halcyon-Zhang/BrowseComp-V3
- Paper: https://arxiv.org/abs/2602.12876

```powershell
hf download Halcyon-Zhang/BrowseComp-V3 `
  --repo-type dataset `
  --local-dir benchmark/raw/BrowseComp-V3
```

### VisBrowse-Bench

Links:
- GitHub: https://github.com/ZhengboZhang/VisBrowse-Bench
- Paper: https://arxiv.org/abs/2603.16289

```powershell
git clone https://github.com/ZhengboZhang/VisBrowse-Bench benchmark/raw/VisBrowse-Bench
```

### Agentic-MME

Links:
- Dataset: https://huggingface.co/datasets/Agentic-MME/Agentic-MME
- GitHub: https://github.com/ChoS3nE11ven/Agentic-MME
- Paper: https://arxiv.org/abs/2604.03016

```powershell
hf download Agentic-MME/Agentic-MME `
  --repo-type dataset `
  --local-dir benchmark/raw/Agentic-MME
```

### MC-Search

Links:
- GitHub: https://github.com/YennNing/MC-Search
- Project: https://mc-search-project.github.io/
- Paper: https://arxiv.org/abs/2603.00873

```powershell
git clone https://github.com/YennNing/MC-Search benchmark/raw/MC-Search
```

### InterLV-Search

Links:
- GitHub: https://github.com/hbhalpha/InterLV-Search-Bench
- Paper: https://arxiv.org/abs/2605.07510

```powershell
git clone https://github.com/hbhalpha/InterLV-Search-Bench benchmark/raw/InterLV-Search-Bench
```

The repository stores encrypted JSON files. Follow its README to decrypt `level1.json.enc`, `level2.json.enc`, and `level3.json.enc`.

### MERRIN

Links:
- GitHub: https://github.com/HanNight/MERRIN
- Dataset: https://huggingface.co/datasets/HanNight/MERRIN
- Project: https://merrin-benchmark.github.io/
- Paper: https://arxiv.org/abs/2604.13418

```powershell
git clone https://github.com/HanNight/MERRIN benchmark/raw/MERRIN
cd benchmark/raw/MERRIN
python load_dataset.py --output data/questions/MERRIN.jsonl
```

### EpiBench

Links:
- Paper: https://arxiv.org/abs/2604.05557

I did not find a stable official dataset repository in the quick survey. Keep it as a watch-list benchmark unless the authors release data/code.

## Notes For Future Adapters

The easiest new adapters are likely BrowseComp-V3, VisBrowse-Bench, and Agentic-MME because they already expose question/image/answer-style fields and align with the current tool interface. For each new benchmark, try to normalize into the existing processed schema:

```json
{
  "question_id": "...",
  "problem": "...",
  "answer": "...",
  "images": ["relative/path.png"],
  "data_source": "...",
  "metadata": {}
}
```

For BrowseComp-style and search-style datasets, keep source metadata such as `category`, `sub_category`, `difficulty`, `level`, `domain`, and `evidence_type` because `build_benchmark_subset.py` can use them for stratified sampling.

## Source Links

- VisualToolBench: https://arxiv.org/abs/2510.12712, https://huggingface.co/datasets/ScaleAI/VisualToolBench
- TIR-Bench: https://arxiv.org/abs/2511.01833, https://huggingface.co/datasets/Agents-X/TIR-Bench
- MMSearch-Plus: https://arxiv.org/abs/2508.21475, https://huggingface.co/datasets/Cie1/MMSearch-Plus, https://mmsearch-plus.github.io/
- MM-BrowseComp: https://arxiv.org/abs/2508.13186, https://github.com/MMBrowseComp/MM-BrowseComp, https://huggingface.co/datasets/mmbrowsecomp/MMBrowseComp
- AgentVista: https://arxiv.org/abs/2602.23166, https://github.com/hkust-nlp/AgentVista, https://huggingface.co/datasets/Warrieryes/AgentVista
- BrowseComp-V3: https://arxiv.org/abs/2602.12876, https://github.com/Halcyon-Zhang/BrowseComp-V3, https://huggingface.co/datasets/Halcyon-Zhang/BrowseComp-V3
- VisBrowse-Bench: https://arxiv.org/abs/2603.16289, https://github.com/ZhengboZhang/VisBrowse-Bench
- Agentic-MME: https://arxiv.org/abs/2604.03016, https://github.com/ChoS3nE11ven/Agentic-MME, https://huggingface.co/datasets/Agentic-MME/Agentic-MME
- MC-Search: https://arxiv.org/abs/2603.00873, https://github.com/YennNing/MC-Search, https://mc-search-project.github.io/
- InterLV-Search: https://arxiv.org/abs/2605.07510, https://github.com/hbhalpha/InterLV-Search-Bench
- MERRIN: https://arxiv.org/abs/2604.13418, https://github.com/HanNight/MERRIN, https://huggingface.co/datasets/HanNight/MERRIN
- EpiBench: https://arxiv.org/abs/2604.05557
