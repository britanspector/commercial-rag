# commercial-rag 前端

React + Vite + TypeScript + Ant Design 控制台，通过 HTTP 调用 FastAPI 后端（`src/api/main.py`）。

## 开发

```bash
cd frontend
npm install
npm run dev
```

默认地址：http://localhost:5173  
开发环境通过 Vite 将 `/api` 代理到 `http://localhost:8000`。

另开终端启动后端：

```bash
uvicorn rag_api:app --host 0.0.0.0 --port 8000 --app-dir src
```

## 目录结构

```
src/
├── api/          # HTTP 客户端与类型定义
├── components/   # 通用组件
├── layouts/      # 页面布局
├── pages/        # 功能页面（含占位页）
└── routes/       # 路由配置
```

## 环境变量

复制 `.env.example` 为 `.env` 并按需修改：

- `VITE_API_BASE_URL`：API 根路径，默认 `/api`（开发代理）

## 功能页面

| 路径 | 说明 |
|------|------|
| `/search`、`/chat` | 结果区展示响应体 `cache` 字段（命中/来源/延迟） |
| `/cache` | 累计统计，对接 `GET /cache/stats` |

生产启用语义缓存时，后端需设置 `RAG_SEMANTIC_CACHE_ENABLED=1` 等环境变量，见项目根 `.env.example` 与 [semantic-cache-scheme.md](../docs/semantic-cache-scheme.md)。

