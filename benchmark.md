**通用加速设置**
PowerShell：

```powershell
pip install -U "huggingface_hub[cli,hf_transfer]" hf_transfer hf_xet
hf auth login   # 可选，但建议登录，速率限制更少

$env:HF_HUB_ENABLE_HF_TRANSFER="1"
$env:HF_XET_HIGH_PERFORMANCE="1"
```

大陆网络可尝试镜像：

```powershell
$env:HF_ENDPOINT="https://hf-mirror.com"
```

官方文档说明 `hf download --local-dir` 会保留目录结构并支持增量下载；`hf_transfer` 可通过并行分块加速大文件下载。

**各 Benchmark 下载**

1. VisualToolBench  
链接：https://huggingface.co/datasets/ScaleAI/VisualToolBench

```powershell
hf download ScaleAI/VisualToolBench `
  --repo-type dataset `
  --local-dir benchmark_full/raw/VisualToolBench
```

约 7.8GB。论文只用其中 single-turn hybrid 子集：train 100 / test 214，需要下载后再按论文 split 过滤。

2. TIR-Bench  
链接：https://huggingface.co/datasets/Agents-X/TIR-Bench  
备用说明页：https://evalscope.readthedocs.io/en/v1.6.1/benchmarks/tir_bench.html

```powershell
hf download Agents-X/TIR-Bench `
  --repo-type dataset `
  --local-dir benchmark_full/raw/TIR-Bench
```

约 1.2GB。论文只用 5 类：`refcoco`、`maze`、`instrument`、`ocr`、`contrast`，再抽 train 100 / test 200。

3. MMSearch-Plus  
链接：https://huggingface.co/datasets/Cie1/MMSearch-Plus  
项目页：https://mmsearch-plus.github.io/

```powershell
hf download Cie1/MMSearch-Plus `
  --repo-type dataset `
  --local-dir benchmark_full/raw/MMSearch-Plus
```

约 924MB。该仓库含 `decrypt_after_load.py`，下载后按仓库说明解密/加载。论文 split：train 100 / test 211。

4. MMBrowseComp  
官方 GitHub：https://github.com/MMBrowseComp/MM-BrowseComp  
HF 数据：https://huggingface.co/datasets/mmbrowsecomp/MMBrowseComp  
XSkill 友好格式镜像：https://huggingface.co/datasets/DjangoJungle/MMBrowseComp

```powershell
hf download mmbrowsecomp/MMBrowseComp `
  --repo-type dataset `
  --local-dir benchmark_full/raw/MMBrowseComp
```

如果你是为了直接喂给 XSkill，建议用 `DjangoJungle/MMBrowseComp`，它已经有图片和类似 XSkill 的字段：

```powershell
hf download DjangoJungle/MMBrowseComp `
  --repo-type dataset `
  --local-dir benchmark_full/raw/MMBrowseComp_XSkillFormat
```

论文用 130 条，全部作为 OOD test。

5. AgentVista  
链接：https://huggingface.co/datasets/Warrieryes/AgentVista  
论文页：https://arxiv.org/abs/2602.23166

```powershell
hf download Warrieryes/AgentVista `
  --repo-type dataset `
  --local-dir benchmark_full/raw/AgentVista
```

约 538MB。论文 split：train 100 / test 109。