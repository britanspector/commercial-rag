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

## 构建

```bash
npm run build
npm run preview
```
