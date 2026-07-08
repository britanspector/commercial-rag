# MinerU 解析说明（历史入口）

本文件的内容已经整合进 [data-pipeline.md](data-pipeline.md)。

现在建议统一从下面文档阅读：

- [project-overview.md](project-overview.md)：项目总览
- [data-pipeline.md](data-pipeline.md)：PDF 解析、分块、向量化、BM25 索引全流程

如果你只是想运行解析阶段，当前入口仍然是：

```bash
python src/parse_pdf_mineru.py
```

历史说明：

- 这里原先记录了 MinerU 解析阶段的安装、GPU 配置和旧分支背景
- 其中分支切换和“方案 A / 方案 B”对比属于项目演进历史，不再作为主文档维护
