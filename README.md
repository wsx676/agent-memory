# 金融长文档智能阅读理解系统 MVP

本项目基于 MVP 设计文档实现，面向金融长文档竞赛场景，提供从文档预处理、规则召回、证据筛选、逐选项推理到结果交付的完整控制台与后端 API 骨架。

## 项目结构

```text
.
├── api/                 # FastAPI 后端
├── src/                 # React 控制台前端
├── tests/               # 后端测试
├── docs/                # 部署、交付、测试等文档
├── .trae/documents/     # PRD 与技术架构文档
├── data/                # 运行期结果文件
└── .github/workflows/   # CI 配置
```

## 快速开始

### 1. 安装前端依赖

```bash
npm install
```

### 2. 安装后端依赖

```bash
python -m pip install -r requirements.txt
```

### 3. 启动后端

```bash
npm run backend:dev
```

### 4. 启动前端

```bash
npm run dev
```

## 常用命令

```bash
npm run lint
npm run check
npm run test:run
npm run backend:test
npm run backend:lint
```

## 当前已实现内容
- React 多页控制台
- FastAPI MVP API
- 文档登记、预处理、任务执行、结果展示
- 结构化证据与 Token 统计输出
- Git Flow 与 GitHub Actions CI 基线
- 部署手册、开发文档、交付报告骨架

## 后续迭代建议
- 接入真实 Qwen API 客户端
- 加强 PDF/OCR 结构恢复与表格处理
- 提升 B 榜文档级召回与多选题策略
- 增强质量面板和交付打包脚本
