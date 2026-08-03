# 智扫通：基于 Agent 与 RAG 的机器人智能客服

[![CI](https://github.com/Wan1ngLunar/smart-cleaning-agent-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Wan1ngLunar/smart-cleaning-agent-rag/actions/workflows/ci.yml)

智扫通是一个面向扫地机器人的智能客服项目，基于 LangChain、LangGraph、Chroma 和 Streamlit 构建。系统能够调用知识库检索、演示天气、用户信息和设备使用记录等工具，并支持多轮对话与个性化使用报告生成。

> 本项目用于技术学习和功能演示。用户身份、地理位置、天气及设备记录均为本地演示数据，不代表真实业务系统或实时接口。

## 核心能力

- **RAG 知识库问答**：检索本地 TXT、PDF 文档，生成带行内编号和文件来源的回答。
- **混合拒答机制**：先过滤明显低分片段，再由 System 级知识边界判断资料是否真正足以回答，避免把相似知识迁移到错误设备或时效场景。
- **可量化检索评估**：使用固定正负例计算 Hit@1、Hit@3、MRR 和分数重叠，并提供真实模型冒烟与全量验收脚本。
- **Agent 工具调用**：根据问题自动选择知识检索、天气、用户信息和使用记录工具。
- **持久化多轮记忆**：使用 LangGraph SQLite Checkpointer 保存会话状态，以 URL 中的 UUID 隔离会话并支持应用重启恢复。
- **个性化报告生成**：根据本地 CSV 中的设备使用记录生成月度报告和保养建议。
- **确定性演示数据**：固定演示用户和城市，避免随机数据导致结果不可复现。
- **容器化交付**：使用非 root Docker 镜像和 Docker Compose 一键启动，提供健康检查、运行时密钥注入及数据库与日志持久化挂载。
- **请求级可观测性**：使用问题编号关联Agent、模型和工具调用，记录成功、失败及耗时，同时避免日志保存用户问题、工具参数和完整会话标识。
- **友好错误处理**：服务端保留带问题编号的异常堆栈，Streamlit页面只展示安全提示，不向用户暴露SDK、文件路径和内部调用栈。
- **工程化保障**：提供环境变量管理、自动化测试、覆盖率统计和 Ruff 静态检查。

## 技术栈

| 分类 | 技术 |
| --- | --- |
| Web 界面 | Streamlit |
| Agent 编排 | LangChain、LangGraph |
| 大语言模型 | 通义千问 |
| Embedding 模型 | DashScope Text Embedding |
| 向量数据库 | Chroma |
| 会话持久化 | LangGraph SQLite Checkpointer |
| 数据与配置 | CSV、YAML、python-dotenv |
| 容器化 | Docker、Docker Compose |
| 测试与质量 | pytest、pytest-cov、Ruff |

## 系统架构

```mermaid
flowchart TD
    U["用户"] --> UI["Streamlit 聊天界面"]
    UI --> A["ReactAgent / LangGraph"]
    A <--> M["通义千问聊天模型"]
    A --> C["SQLite Checkpointer 持久化会话"]
    A --> MW["请求追踪、模型与工具监控、动态 Prompt 中间件"]
    A --> T{"Agent 选择工具"}

    T --> R["RAG 知识检索"]
    R --> V["Chroma 向量数据库"]
    V --> K["本地 TXT / PDF 知识文档"]

    T --> D["演示用户、城市与天气"]
    D --> Y["config/agent.yml"]

    T --> E["月度使用记录查询"]
    E --> CSV["data/external/records.csv"]

    MW --> P["普通客服 Prompt / 报告 Prompt"]
```

## 请求处理流程

1. Streamlit 校验 URL 中的 UUID `thread_id`；参数缺失或非法时创建新会话并写回地址栏。
2. LangGraph 根据 `thread_id` 从 SQLite 恢复模型状态，页面同时过滤内部工具消息并重建可展示的聊天历史。
3. 每次提问生成12位问题编号；Agent、模型和工具日志使用同一编号记录调用结果与毫秒耗时，不记录问题正文、工具参数或完整会话ID。
4. Agent 将用户问题交给模型，由模型判断是否需要调用工具。
5. 知识问答场景会检索 Chroma 中的本地文档，并在生成前过滤低于最低相关性分数的片段。
6. 通过低分过滤后，System 消息会约束模型核对问题对象、范围和时效；资料不足时返回统一拒答，资料充分时生成带编号引用的回答，再由服务层追加去重后的文件名和 PDF 页码。
7. 报告场景会读取本地 CSV 演示记录，并通过中间件切换到报告专用 Prompt。
8. Agent的最终结果以流式方式返回Streamlit；执行失败时服务端日志保留异常因果链，页面只展示可用于排查的问题编号。

## 项目结构

```text
.
├─ agent/                 # Agent 编排、工具和中间件
├─ config/                # Agent、RAG、Chroma 等非敏感配置
├─ data/                  # 本地知识文档和演示使用记录
├─ evaluation/            # 检索用例、离线指标和真实模型可回答性评估
├─ model/                 # 聊天模型与 Embedding 模型工厂
├─ prompts/               # 普通问答、RAG 总结和报告 Prompt
├─ rag/                   # 文档切分、向量入库和检索服务
├─ tests/                 # 自动化测试与文本编码守卫
├─ utils/                 # 配置、文件、日志、路径和 Prompt 工具
├─ app.py                 # Streamlit 应用入口
├─ Dockerfile             # 非 root 用户、健康检查和Streamlit启动配置
├─ compose.yml            # 端口、密钥和持久化目录编排
├─ .dockerignore          # 排除密钥、虚拟环境和本地运行数据
├─ requirements.txt       # 固定版本的运行依赖
├─ requirements-dev.txt   # 测试与静态检查依赖
├─ pyproject.toml         # pytest、coverage 和 Ruff 配置
├─ .env.example           # 环境变量示例，不包含真实密钥
└─ change.md              # 从原始版本开始的增量改造记录
```

首次运行后生成的 `storage/` 保存 Chroma 数据库、文档 MD5 状态和 SQLite 会话检查点。该目录属于可能包含本地对话的运行产物，已被 Git 忽略，不应提交或公开分享。

## 快速开始

### 1. 准备环境

当前版本在 Python 3.13 环境中完成验证。以下命令在项目根目录执行。

```powershell
# 创建项目专用虚拟环境，避免依赖污染系统 Python。
python -m venv .venv

# 在 PowerShell 中激活虚拟环境。
.\.venv\Scripts\Activate.ps1

# 安装固定版本的项目运行依赖。
python -m pip install -r requirements.txt
```

### 2. 配置模型密钥

```powershell
# 根据安全示例创建只在本机使用的环境变量文件。
Copy-Item -LiteralPath .\.env.example -Destination .\.env
```

打开 `.env`，将占位符替换为自己的 DashScope API Key：

```dotenv
DASHSCOPE_API_KEY=你的真实密钥
```

`.env` 已被 Git 忽略，不应把真实密钥写入 `.env.example`、源码、日志或提交记录。

### 3. 初始化本地知识库

```powershell
# 读取 data 目录中的 TXT 和 PDF，切分文本并写入 Chroma。
python -m rag.vector_store
```

文档内容未变化时，程序会根据 MD5 跳过重复导入。数据库和导入状态统一保存在 `storage/`。

### 4. 启动应用

```powershell
# 从项目根目录启动 Streamlit 聊天页面。
python -m streamlit run app.py
```

启动后访问终端显示的本地地址。页面会把合法的 `thread_id` 写入 URL；使用同一完整 URL 可以在应用重启后恢复模型状态和页面聊天历史。点击侧边栏的“新建对话”会生成新 UUID、清空页面消息并切换到隔离会话。

URL 中的 UUID 相当于本地会话访问标识。当前演示没有登录和权限校验，不应公开分享包含真实对话的完整 URL。

### 5. 使用 Docker Compose 启动

首次启动前仍需按照第2步在项目根目录创建本地`.env`。Docker构建上下文会通过`.dockerignore`排除真实密钥、虚拟环境、数据库、日志和本地备份；`.env`只在容器运行时注入，不会写入镜像。

```powershell
# 构建镜像并在后台启动应用。
docker compose up --build --detach

# 首次运行或知识文档发生变化时，将data中的文档导入宿主机挂载的Chroma目录。
docker compose run --rm app python -m rag.vector_store

# 查看容器是否进入healthy状态。
docker compose ps

# 查看最近的应用日志，不输出本地.env文件内容。
docker compose logs --tail 100 app
```

启动后访问`http://localhost:8501`。`storage/`和`logs/`分别挂载到容器内的`/app/storage`和`/app/logs`，因此重建或重启容器不会删除向量库、SQLite会话和宿主机日志。

```powershell
# 停止并删除Compose容器和项目网络；宿主机挂载的数据与本地镜像会保留。
docker compose down
```

镜像使用普通用户`appuser`运行，并通过Streamlit内置健康接口检查服务。Docker Desktop的镜像下载代理与容器访问模型API的网络路径可能不同；代理应在Docker Desktop或本机网络工具中按环境配置，不应把代理地址、端口或凭据固化到Dockerfile、Compose文件或代码仓库。

### DashScope 连接被重置

如果页面出现 `ConnectionResetError 10054` 或 `Connection aborted`，说明 Python 在建立 DashScope HTTPS 连接时被本地网络或代理中断，并不代表 Chroma、Agent 或 API Key 一定存在问题。

- 检查代理软件的系统代理、TUN 和分流规则；
- 国内网络直连 DashScope 可能比强制经过代理更稳定，具体以本机网络测试为准；
- 修改代理状态后需要停止并重新启动 Streamlit 进程；
- 不要通过关闭 SSL 验证来绕过连接问题；
- 不要把本机代理地址、端口或凭据写入仓库配置。

### 使用问题编号排查失败请求

模型、工具或网络异常时，页面只显示安全提示和12位问题编号，例如：

```text
请求处理暂时失败，请稍后重试。问题编号：a1b2c3d4e5f6
```

开发者可以在本地或容器日志中按编号定位同一次请求：

```powershell
# 在Compose日志中查找指定问题编号关联的Agent、模型和工具记录。
docker compose logs app |
  Select-String "request_id=a1b2c3d4e5f6"
```

正常日志会记录请求开始、成功或失败、模型与工具名称、消息数量和毫秒耗时。日志不会主动保存用户完整问题、工具参数、完整`thread_id`或API Key。服务端失败日志包含开发排查所需的异常类型和调用栈，但这些内部信息不会显示在Streamlit页面。

## 质量检查

开发环境安装：

```powershell
# 在运行依赖基础上安装 pytest、pytest-cov 和 Ruff。
python -m pip install -r requirements-dev.txt
```

提交代码前建议依次执行：

```powershell
# 检查明显语法错误、未定义名称和导入顺序。
python -m ruff check .

# 运行全部自动化测试。
python -m pytest -q

# 运行核心模块覆盖率并执行与CI相同的60%门禁。
python -m pytest --cov=agent --cov=model --cov=rag --cov=utils --cov-report=term-missing --cov-fail-under=60 -q
```

当前基线为39项测试通过，核心模块测试覆盖率为76.85%。测试覆盖来源去重、PDF页码、混合拒答、检索指标、SQLite表与安全序列化、跨Agent实例恢复、会话隔离、历史消息过滤、UUID校验、请求编号、模型与工具耗时日志、异常包装和日志参数脱敏。现有一条`langchain-community`弃用警告，已记录为后续依赖迁移事项，没有通过过滤规则隐藏。

## RAG 评估

检索评估用例保存在 `evaluation/retrieval_cases.yml`，当前包含 6 条正例和 8 条负例。负例既包含炒菜、股票等简单跨领域问题，也包含实时价格、具体型号、手持吸尘器和工业机器人等语义相近但知识范围不匹配的困难问题。

```powershell
# 查询真实Chroma并计算Hit@1、Hit@3、MRR和正负例分数边界。
python -m evaluation.evaluate_retrieval

# 使用真实模型快速验证正常回答、低分拒答和高分范围拒答。
python -m evaluation.evaluate_answerability

# 使用真实模型运行YAML中的全部14条可回答性用例。
python -m evaluation.evaluate_answerability --all
```

真实检索基线为 Hit@1 100%、Hit@3 100%、MRR 1.0。困难负例最高 Top-1 分数为 0.6617，高于正例最低分 0.4798，证明单一相似度阈值无法可靠判断是否可回答。因此系统采用两层机制：`0.20` 仅作为明显无关片段的粗过滤门槛，高分结果继续由 System 级知识范围和资料充分性规则判断。

真实模型冒烟评估为 3/3 通过，全量评估为 14/14 通过。上述命令会调用真实 Chroma 和 DashScope，不属于默认 CI，运行前需要配置有效 API Key，并会产生少量 API 用量。

## 持续集成

仓库提供 [`.github/workflows/ci.yml`](.github/workflows/ci.yml)。推送到 `master` 或 `main`、创建或更新 Pull Request 时，GitHub Actions 会自动：

1. 在 Ubuntu 环境中安装 Python 3.13；
2. 根据依赖文件缓存并安装开发依赖；
3. 运行 Ruff 静态检查；
4. 运行全部测试和覆盖率统计；
5. 在覆盖率低于 60% 时阻止检查通过。

CI 使用测试专用的 DashScope 占位值，不保存真实 API Key，也不会在现有测试中调用真实聊天或 Embedding 服务。工作流已在 GitHub 的 Ubuntu Runner 中完成首次在线验证，状态为 Success。

## 可复现的演示场景

可以在页面中依次尝试：

1. `小户型应该如何选择扫地机器人？请给出知识库来源。`
2. `深圳现在的演示天气是什么？请说明数据是否实时。`
3. `我家面积是68平方米，并且养了两只猫。`
4. `请复述我刚才提供的家庭信息。`
5. `生成用户1001在2025-06的使用报告。`
6. 点击“新建对话”，再询问第4个问题，验证新旧会话隔离。
7. 复制当前完整URL，停止并重新启动Streamlit，再打开该URL验证聊天历史和模型记忆恢复。
8. 点击“新建对话”确认URL中的UUID变化，再打开旧URL验证旧会话仍可恢复。

这些场景分别覆盖带来源的 RAG 检索、演示工具、持久化多轮记忆、报告生成、会话隔离和跨应用重启恢复。RAG 回答末尾应显示去重后的文件名，PDF 来源还应显示页码。

## 关键工程设计

- **路径稳定性**：Chroma 和 MD5 路径先转换为项目绝对路径，避免从不同目录启动时生成多份数据库。
- **幂等导入**：文档写入 Chroma 成功后才保存 MD5，失败任务可以在下次启动时重试。
- **持久化会话**：模型状态按`thread_id`写入SQLite；URL保留UUID，应用重启后可重新定位并恢复同一会话。
- **状态一致性**：页面历史由SQLite中的LangGraph状态重建，只展示用户消息和最终AI文本；新建对话同步更新URL、模型会话键和页面消息。
- **连接生命周期**：每个浏览器会话复用独立Agent和SQLite连接，资源释放时幂等关闭；WAL和等待超时改善本地并发访问。
- **安全反序列化**：Checkpoint只允许LangGraph内置安全类型白名单，白名单外对象不会被重新实例化。
- **数据一致性**：CSV 按表头读取并校验必要列、空记录和重复用户月份，再一次性发布到进程缓存。
- **来源可追溯**：模型上下文使用稳定编号，服务层根据文档元数据追加文件名和 PDF 页码，不依赖模型自行生成来源。
- **空检索保护**：没有有效知识片段时直接返回明确说明，不调用模型编造答案或产生额外费用。
- **混合拒答**：低相关性粗过滤减少无效模型调用；高分结果由 System 级设备范围、问题对象和时效规则继续判断，资料不足时不附加误导性来源。
- **评估可回归**：固定 YAML 用例同时衡量正确来源排名和无答案行为；离线指标计算不依赖真实模型，真实可回答性评估可按冒烟或全量模式运行。
- **安全配置**：模型密钥从环境变量加载；系统环境变量优先于本地 `.env`，缺失时快速失败并给出修复提示。
- **容器安全**：构建上下文排除密钥与本地产物，镜像以无登录权限的普通用户运行；密钥只在容器启动时注入，健康检查不调用模型API。
- **容器持久化**：Compose将Chroma、SQLite和日志映射到宿主机目录，容器重建后仍可恢复知识索引与历史会话。
- **请求关联**：每次提问生成独立问题编号，Agent、模型和工具日志共享该编号，并分别记录调用结果和耗时。
- **日志最小化**：日志只保存排障所需的编号、阶段、数量、异常类型和耗时，不主动记录问题正文、工具参数、完整会话ID或密钥。
- **错误边界**：Agent边界记录真实异常并转换为安全错误；页面只显示问题编号，避免将SDK调用栈、内部路径和网络细节暴露给用户。
- **演示边界**：固定用户、城市和天气均明确标注为演示数据，不冒充真实业务接口。
- **自动化守卫**：测试覆盖关键配置、路径、工具、CSV、文件边界和 UTF-8 文本规范。

## 当前限制

- SQLite持久化适合本地演示和单机工作流；多实例生产部署仍应迁移到PostgreSQL等服务化存储。
- URL中的UUID尚未绑定登录用户，知道完整URL的人可能访问对应本地会话；公开部署前必须增加身份认证、授权和会话过期机制。
- 天气、用户身份和设备记录来自本地演示配置，不是实时外部服务。
- Chroma 使用本地持久化目录，暂未提供独立向量数据库服务。
- 当前评估集为 14 条人工设计用例，能够建立回归基线，但不能代表所有真实用户问题；真实模型判断还可能随模型版本变化。
- 当前容器配置面向本地单机演示，尚未提供用户登录、权限隔离、限流、监控告警、TLS入口和多实例生产编排。
- 当前可观测性基于本地文本日志和问题编号，尚未接入集中式日志、指标系统、分布式追踪和自动告警。
- `langchain-community` 中的通义模型集成已提示弃用，后续需要迁移到独立维护的集成包。

## 后续计划

- 增加会话列表、标题、删除和过期清理，并为公开部署加入登录与访问控制。
- 将单机SQLite Checkpointer迁移为适合多实例部署的PostgreSQL后端。
- 扩充检索评估集，引入更多真实问法、困难负例和持续指标对比。
- 评估重排序或结构化可回答性分类，进一步降低对生成模型边界判断的依赖。
- 为 Agent 流程、中间件和 Streamlit 交互补充自动化测试。
- 将本地请求日志接入集中式日志与指标平台，并建立延迟、错误率和工具失败告警。
- 增加公开部署所需的身份认证、TLS入口、监控告警和多实例编排。
- 迁移已弃用的模型集成依赖，并验证版本兼容性。

## 改造记录

项目从原始版本开始的每项问题、修改内容、设计意义和验证结果均记录在 [`change.md`](change.md)。
