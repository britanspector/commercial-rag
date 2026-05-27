# Milvus 与 FAISS 索引类型对比实验方案

## 背景

原实验设计在 FAISS 上对比 **Flat（精确）**、**IVF（倒排聚类近似）**、**HNSW（图索引近似）** 的召回速度与精度权衡。当前项目使用 **Milvus Lite**（`pymilvus` + 本地 `milvus.db`）存储 BGE 向量。

## 结论摘要

| 能力 | Milvus Lite（当前） | Milvus Standalone / Cloud |
|------|---------------------|---------------------------|
| Flat / 暴力精确检索 | 默认行为（小规模） | 支持，索引类型 `FLAT` |
| IVF 近似索引 | **不支持**（Lite 能力受限） | 支持 `IVF_FLAT` / `IVF_PQ` / `IVF_SQ8` |
| HNSW 近似索引 | **不支持** | 支持 `HNSW` |
| 多索引 A/B 同库对比 | Lite 下基本不可行 | 可为同一 collection 建多种索引（需服务端） |

**可行方案：**

1. **小规模（当前 ~1k 向量）**  
   Milvus Lite 的向量检索等价于 **Flat + COSINE**，与 FAISS `IndexFlatIP/COSINE` 在同一量级。  
   实验重点可放在 **召回路线**（向量 / BM25 / 混合），而非 ANN 索引结构。

2. **若必须复现 FAISS 三索引对比**  
   - **方案 A（推荐）**：单独脚本用 `faiss-cpu` 对同一 `embeddings.npy` 建 Flat / IVF / HNSW，固定 query 集测 **Recall@K vs QPS**，Milvus 仅作线上存储。  
   - **方案 B**：部署 **Milvus Standalone**（Docker），创建 collection 时指定 `index_type` 为 `HNSW` 或 `IVF_FLAT`，通过 `search_params`（如 `nprobe`、`ef`）调参；需重新 `embed_chunks` 写入服务端。  
   - **方案 C**：Milvus 2.5+ **多索引**（同一字段多个 index）在 **Cluster 模式** 下可做并行检索对比；Lite 不适用。

## Milvus 索引类型与 FAISS 对照

| Milvus 索引 | 类似 FAISS | 特点 | 主要参数 |
|-------------|------------|------|----------|
| `FLAT` | `IndexFlatL2` / 内积版 | 100% 召回（暴力），适合 < 百万级 | 无 |
| `IVF_FLAT` | `IndexIVFFlat` | 聚类后搜索桶，可调速度/召回 | `nlist`, `nprobe` |
| `IVF_PQ` | `IndexIVFPQ` | 更高压缩，召回略降 | `m`, `nbits`, `nprobe` |
| `HNSW` | `IndexHNSWFlat` | 图索引，低延迟高召回 | `M`, `efConstruction`, `ef` |
| `AUTOINDEX` | — | 托管版自动选择 | 服务端 |

当前 `milvus_store.py` 使用 `MilvusClient.create_collection(..., metric_type="COSINE")`，**未显式指定 index_type**，在 Lite 上为默认平面检索。

## 在 Milvus Standalone 上做对比实验的步骤（不执行，仅方案）

1. Docker 启动 Milvus：`docker compose -f standalone/docker-compose.yml up`  
2. 修改 `embed_chunks.py` / `milvus_store.py`：连接 `http://localhost:19530`，创建 collection schema。  
3. 对同一批向量分别建三个 collection（或同一 collection 换索引需 rebuild）：  
   - `rag_chunks_flat` → `index_type=FLAT`  
   - `rag_chunks_ivf` → `index_type=IVF_FLAT`, `params={"nlist": 128}`  
   - `rag_chunks_hnsw` → `index_type=HNSW`, `params={"M": 16, "efConstruction": 200}`  
4. 用 `eval_retrieval.py` 的 90 题 query 向量，对每库跑 Recall@10，并记录 `search` 延迟 P50/P99。  
5. IVF/HNSW 扫描 `nprobe` / `ef` 曲线，绘制 **速度-召回** 折中图（对标 FAISS 实验报告）。

## Milvus Lite 的限制说明

- 官方定位：本地开发/原型，**不支持**完整索引管理与多索引。  
- Windows 上通过本地 `milvus.db` 文件持久化；`flush` 可能失败（项目已做容错）。  
- 若索引实验是论文/课程硬性要求，建议 **FAISS 离线对比 + Milvus Lite 工程集成** 双线并行，而不是强行在 Lite 上模拟 IVF/HNSW。

## 与当前 hybrid 实验的关系

| 实验维度 | 工具 | 状态 |
|----------|------|------|
| 稀疏 vs 稠密 vs 混合 | BM25 + Milvus + `eval_route_comparison.csv` | 已实现（feature-hybridRecall） |
| ANN 索引结构 | FAISS 或 Milvus Standalone | 待选方案，Lite 不适用 |

## 参考链接

- [Milvus Index Types](https://milvus.io/docs/index.md)  
- [Milvus Lite Limitations](https://milvus.io/docs/milvus_lite.md)  
- [FAISS Wiki — Index types](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
