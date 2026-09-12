# 部署手册

> 面向部署执行人。故障处置见 [runbook.md](runbook.md)。

## 1. 部署拓扑

```
浏览器 ──HTTPS──> Nginx (web)  ──/api 等路径──> FastAPI (app:8000)
                    │静态资源(Spa)                 │
                    └ /usr/share/nginx/html        └ 持久卷: data / database / knowledge_files / uploads / logs
```

- `web`：Nginx，托管前端构建产物并反向代理 API（含 SSE 免缓冲）。
- `app`：FastAPI + 工作流，非 root 运行，带健康检查。
- 数据全部落在持久卷中；容器重建不丢数据。

## 2. 前置条件

| 项 | 要求 |
|---|---|
| Docker | 24+（含 compose v2） |
| 域名/证书 | HTTPS 需 `deploy/nginx/certs/fullchain.pem` 与 `privkey.pem`（获取方式见 `deploy/nginx/certs/README.md`） |
| 密钥 | `DEEPSEEK_API_KEY`（必需）、`EMBEDDING_API_KEY`（RAG 必需） |
| 可选密钥 | `TAVILY_API_KEY`（结构化搜索）、`COMPOSIO_API_KEY`（外部集成） |

## 3. 环境变量

必填与关键项（完整清单见 [.env.example](../.env.example)）：

| 变量 | 作用 | 缺失后果 |
|---|---|---|
| `DEEPSEEK_API_KEY` | 核心 LLM | 生产模式启动失败（`APP_ENV=production`） |
| `EMBEDDING_API_KEY` | 向量检索 / 意图快路径 | 生产模式启动失败；开发模式降级为词法检索 |
| `APP_ENV` / `ENFORCE_REQUIRED_ENV` | 生产校验开关 | 默认 development，缺失密钥仅告警 |
| `AUTH_TOKEN` | 接口鉴权（除 `/health`） | 为空则**不鉴权**，仅适合内网/本机 |
| `CORS_ORIGINS` | 允许的跨域来源 | 为空时仅允许本机来源 |
| `QDRANT_URL` | 远程向量库地址 | 为空则用本地磁盘索引 |
| `LOG_ROTATION_*` | 日志轮转 | 默认按天、保留 14 份 |

## 4. Docker Compose 部署（推荐）

```bash
# 1) 准备配置与证书
cp .env.example .env         # 填写密钥；生产设置 APP_ENV=production
mkdir -p deploy/nginx/certs  # 放入 fullchain.pem / privkey.pem

# 2) 构建并启动
docker compose up --build -d

# 3) 验证
curl -k https://localhost/health
docker compose ps
```

无证书的本地验证：以 HTTP 配置构建 web 镜像。

```bash
docker compose build --build-arg NGINX_CONF=nginx-http.conf web
docker compose up -d
```

## 5. Kubernetes 部署

清单位于 `deploy/k8s/`，按顺序应用：

```bash
kubectl apply -f deploy/k8s/configmap.yaml
kubectl apply -f deploy/k8s/pvc.yaml
kubectl create secret generic agentflow-secrets \
  --from-literal=deepseek_api_key="$DEEPSEEK_API_KEY" \
  --from-literal=embedding_api_key="$EMBEDDING_API_KEY"
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml
kubectl apply -f deploy/k8s/web-nginx-configmap.yaml
kubectl apply -f deploy/k8s/web-deployment.yaml
kubectl apply -f deploy/k8s/web-service.yaml
kubectl apply -f deploy/k8s/ingress.yaml
```

注意事项：

1. **副本数固定为 1**：SQLite 单写 + `ReadWriteOnce` PVC 决定。需要扩容必须先迁移到共享数据库与 RWX 存储。
2. **密钥必须齐全**：`APP_ENV=production` + 缺少 `EMBEDDING_API_KEY` 会导致容器反复重启（CrashLoopBackOff）。
3. **Ingress 终止 TLS**：证书放在 Ingress 的 Secret 中；容器内 Nginx 只跑 HTTP。
4. **数据卷**：`agentflow-data` PVC 通过 subPath 分别挂载 `data/`、`database/`、`knowledge_files/`、`uploads/`、`logs/`。

## 6. 升级与回滚

```bash
# 升级：拉取新代码 → 构建带版本 tag 的镜像 → 滚动更新
docker compose build
docker compose up -d

# 回滚：切回上一个镜像 tag（建议每次发布保留上一版本镜像）
docker compose down
docker tag omniforge-api:<上一个版本> omniforge-api:latest
docker compose up -d
```

数据库 schema 由应用启动时自动迁移（`PRAGMA user_version` 记录版本）。**升级前务必备份**（见 runbook）。

## 7. 发布前验收（冒烟清单）

| # | 检查 | 通过标准 |
|---|---|---|
| 1 | `/health` | 返回 200，`status=ok`，数据库检查通过 |
| 2 | 对话 | 发一句"你好"，收到正常回答；再发任务类请求（如"写一个待办清单项目"）能产出文件 |
| 3 | 知识库 | 上传一个 txt/md，检索能命中；文档出现在列表并可预览 |
| 4 | 前端 | 打开站点可加载，消息可流式显示，刷新后历史保留 |
| 5 | 持久化 | 重启容器后，会话与知识库数据仍在 |
| 6 | 鉴权（如启用） | 未带 Token 的请求返回 401，`/health` 例外 |
| 7 | 日志 | 有 JSON 日志且含 `trace_id`；`logs/` 持久化 |

## 8. 回滚判据

出现下列任一情况即回滚：`/health` 持续非 200；对话接口 5xx 比例显著上升；
数据迁移失败；鉴权/跨域异常导致前端不可用。
