# 项目改造记录

本文档记录项目在原始版本基础上的增量改造，帮助后续开发者了解每项修改的背景、实现方式和验证结果。

## 记录规则

- 每次新增或修改代码时，同步更新本文档。
- 每项变更应说明：原问题、修改内容、修改意义、验证方式和相关提交。
- 只记录实际完成并验证过的结果；未完成的验证明确标记为“待验证”。
- 密钥、用户隐私数据和完整日志不得写入本文档。
- 新增或修改代码时，为会话状态、缓存、数据边界和失败处理等非显然逻辑补充解释性注释。
- 项目文本统一使用 UTF-8，中文直接写为可读字符串，不使用 Unicode 转义序列。

## 2026-07-27：建立 Git 安全基线

### 原问题

- 项目目录不是有效的 Git 仓库，缺少可追踪的版本历史。
- 本地包含 API 密钥、虚拟环境、日志、Python 缓存和 Chroma 运行数据，直接执行 `git add .` 有误提交敏感信息和运行产物的风险。
- Windows 与其他系统的默认换行符不同，可能产生无意义的整文件差异。

### 修改内容

1. 初始化 Git 仓库，并创建原始项目基线提交。
2. 扩充 `.gitignore`，忽略以下本地内容：
   - `.venv/`、IDE 配置和 Python 缓存；
   - `config/secret.yml`、`.env` 等密钥文件；
   - `logs/` 和 `*.log`；
   - 所有 `chroma_db/`、新的 `storage/` 和旧的 `md5.text`；
   - `local_backup/` 本地备份目录。
3. 新增 `.gitattributes`：
   - Python、YAML、TXT、CSV 和 Markdown 文件在仓库内统一使用 LF；
   - PDF 和 SQLite 文件按二进制文件处理。

### 修改意义

- 防止 API 密钥、日志及本地数据库被提交到代码仓库。
- 为后续改造建立可回退、可审查的版本基线。
- 减少 Windows CRLF 与跨平台 LF 差异造成的无效 Git 变更。

### 验证结果

- `git check-ignore` 确认 `config/secret.yml` 和 `rag/chroma_db/chroma.sqlite3` 已被忽略。
- 基线提交后 `git status` 显示工作区干净。

### 相关提交

- `410d5d8 chore: initialize project baseline`

## 2026-07-27：统一 Chroma 数据库与导入状态路径

### 原问题

`config/chroma.yml` 原先使用相对路径 `chroma_db`。Chroma 会相对于进程的当前工作目录解析该路径，因此项目中产生了三份数据库：

- `chroma_db/`：0 条 embedding；
- `agent/chroma_db/`：0 条 embedding；
- `rag/chroma_db/`：326 条 embedding。

与此同时，文档 MD5 导入状态保存在项目根目录的 `md5.text`。数据库和导入状态可能互相不匹配，导致程序连接空数据库后仍将知识文档误判为“已经导入”。

### 修改内容

1. 将三份旧 Chroma 数据库和旧 `md5.text` 移动到本地备份目录：
   - `local_backup/chroma_before_fix/root_chroma_db`
   - `local_backup/chroma_before_fix/agent_chroma_db`
   - `local_backup/chroma_before_fix/rag_chroma_db`
   - `local_backup/chroma_before_fix/md5.text`
2. 修改 `config/chroma.yml`：
   - `persist_directory` 从 `chroma_db` 改为 `storage/chroma`；
   - `md5_hex_store` 从 `md5.text` 改为 `storage/ingested_md5.txt`。
3. 修改 `rag/vector_store.py`：
   - 使用 `get_abs_path` 将数据库和 MD5 配置转换为基于项目根目录的绝对路径；
   - 初始化时自动创建所需目录；
   - 使用 `self.md5_hex_store` 统一完成 MD5 文件的检查和写入；
   - 保留原有 `RecursiveCharacterTextSplitter` 文本切分逻辑。
4. 重新导入 `data/` 中的 6 份知识文档，生成唯一的新数据库。

### 修改意义

- 无论程序从哪个工作目录启动，都使用 `D:\agent+RAG\storage\chroma`。
- Chroma 数据库和文档导入状态被放在同一个 `storage/` 运行目录下。
- 避免连接空数据库、重复生成数据库或因旧 MD5 状态跳过文档的问题。
- `storage/` 被 Git 忽略，运行数据不会进入代码仓库。

### 验证结果

- `rag/vector_store.py` 通过 Python 语法检查。
- `git diff --check` 未发现补丁格式或空白错误。
- `storage/chroma/chroma.sqlite3` 已成功生成。
- `storage/ingested_md5.txt` 包含 6 条非空 MD5 记录。
- 新 Chroma 数据库包含 326 条 embedding。

### 待补充验证

- 再次执行文档导入后，确认 6 个文件均被跳过且 embedding 数量仍为 326。

### 相关提交

- `6ed6d73 fix: use a single absolute Chroma storage path`

## 2026-07-28：增加基于 thread_id 的多轮对话记忆

### 原问题

Streamlit 使用 `session_state` 显示历史消息，但 `ReactAgent.execute_stream` 每次只向模型传入当前用户问题。界面虽然显示多轮对话，模型实际上无法读取此前内容。

### 修改内容

1. 在 `ReactAgent` 中创建 LangGraph `InMemorySaver`。
2. 创建 Agent 时通过 `checkpointer` 参数启用消息状态保存。
3. `execute_stream` 新增 `thread_id` 参数，并通过 `configurable.thread_id` 标识会话。
4. Streamlit 为每个页面会话生成 UUID。
5. 新增“新建对话”按钮，同时更换 thread ID 并清空页面消息。
6. 同步修改 `react_agent.py` 底部的本地调试调用。

### 修改意义

- 同一个 thread ID 下，模型可以读取此前的用户消息和回答。
- 不同会话之间的历史互相隔离。
- 用户可以主动创建一段没有旧上下文的新对话。
- 当前使用内存存储，适合单机演示；应用重启后历史会清空。

### 验证结果

- `agent/react_agent.py` 和 `app.py` 通过 Python 语法检查。
- `ReactAgent` 初始化成功，输出 `agent_init_ok`，确认当前 LangChain 版本接受 `checkpointer` 参数。
- 在同一会话中先提供“房间面积 68 平方米、家里有两只猫”，下一轮能够准确复述这两项信息。
- 点击“新建对话”后再次询问上述信息，Agent 不再记得旧会话内容，确认新 thread ID 与旧历史隔离。

### 相关提交

- 与本条记录同一提交：`feat: add thread-scoped conversation memory`

## 2026-07-28：使演示工具数据确定且可追溯

### 原问题

用户ID、城市和月份通过随机函数返回，同一会话可能出现身份和报告月份变化。天气工具返回固定数据，却被描述为实时天气。CSV 使用字符串分割，无法正确处理带逗号的字段；查询工具类型标注为字符串，实际返回字典。

### 修改内容

1. 在 `config/agent.yml` 中集中配置演示用户、城市和天气数据。
2. 删除随机用户、随机城市和随机月份逻辑。
3. 使用系统日期返回真实当前月份。
4. 天气返回值和工具描述明确标注为“演示数据、非实时查询”。
5. 使用 `csv.DictReader` 按表头读取本地演示记录。
6. 增加必要列、空记录和重复用户月份校验。
7. 先在局部变量中完成数据加载，成功后再更新全局缓存。
8. 使用JSON字符串作为使用记录的统一返回格式。
9. 修改主Prompt和报告Prompt，明确演示数据边界和可用范围。
10. 将报告标题统一为“智扫通”。

### 修改意义

- 相同输入能够得到稳定、可复现的工具结果。
- Demo 数据不会再被误导性地描述为真实用户或实时天气。
- CSV 中包含逗号或引号时仍能正确解析。
- 工具类型声明、描述和实际返回值保持一致。
- 查询不到记录时进入明确的空结果分支，不再随机选择其他数据代替。

### 验证结果

- 确定性工具测试输出 `tool_context_ok`。
- 当前演示用户固定返回 `1001`。
- 当前月份正确返回 `2026-07`。
- CSV 成功加载10个用户、120条月度记录。
- 使用记录工具返回类型为字符串，测试输出 `return_type: str`。
- 查询不存在的月份时返回空字符串并记录WARNING。
- Prompt 边界扫描未发现将演示工具描述为真实能力的肯定性语句。
- 天气场景明确标注数据为演示数据或非实时天气。
- 用户1001在2025-06的报告生成成功，标题为“智扫通扫地机器人使用情况报告与保养建议”。
- 查询当前月份2026-07时，Agent 明确说明没有记录，未编造指标或改查随机月份。

### 相关提交

- 与本条记录同一提交：`fix: make demo tools deterministic and explicit`

## 2026-07-28：迁移环境变量并固定项目依赖

### 原问题

项目通过 `config/secret.yml` 加载 DashScope API Key，普通配置模块与敏感信息耦合。克隆仓库后缺少该本地文件会导致项目启动失败，同时项目没有依赖清单，无法在新环境中复现当前已验证的运行版本。

### 修改内容

1. 新增 `requirements.txt`，固定当前运行环境中的11个直接依赖及版本。
2. 新增 `.env.example`，公开声明项目需要 `DASHSCOPE_API_KEY`，但只保留安全占位符。
3. 创建被 Git 忽略的本地 `.env`，用于保存真实 DashScope API Key。
4. 从 `utils/config_handler.py` 中删除 `load_secret_config` 和 `secret_conf`。
5. 将普通 YAML 配置加载方式从 `yaml.load(..., FullLoader)` 改为 `yaml.safe_load`。
6. 在 `model/factory.py` 中使用 `python-dotenv` 从项目根目录加载 `.env`。
7. 系统环境变量优先于本地 `.env`，便于后续容器和服务器部署。
8. 新增 `get_required_env`，缺少必要变量时返回包含修复方法的明确错误。
9. 聊天模型和 Embedding 模型统一从 `DASHSCOPE_API_KEY` 环境变量获得凭据。
10. 在迁移验证通过后删除旧 `config/secret.yml` 及其临时备份。

### 修改意义

- 真实密钥不再保存在项目 YAML 配置中，也不会进入 Git。
- 新开发者可以根据 `.env.example` 创建自己的本地配置。
- 缺少环境变量时能够快速失败，避免进入模型 SDK 后才出现难以定位的错误。
- 精确依赖版本使项目更容易在其他电脑复现。
- `yaml.safe_load` 限制 YAML 只能构造安全的基础数据类型。

### 验证结果

- `pip check` 输出 `No broken requirements found.`。
- `.env` 命中 `.gitignore` 规则，不会出现在 Git 状态中。
- `.env.example` 仅包含 `DASHSCOPE_API_KEY` 安全占位符。
- `requirements.txt` 包含11个预期依赖，名称和版本与当前环境一致。
- 项目源码中不再存在 `secret_conf` 或 `load_secret_config` 引用。
- `model.factory` 通过 `.env` 成功创建 `ChatTongyi` 和 `DashScopeEmbeddings`。
- 移走旧 `secret.yml` 后，Embedding API 调用成功并返回非空向量。
- 缺失测试环境变量时成功触发清晰的 `RuntimeError`，输出 `missing_env_fail_fast_ok`。
- 旧 `config/secret.yml` 和临时备份均已删除，本地 `.env` 保留且被 Git 忽略。

### 相关提交

- 与本条记录同一提交：`chore: add reproducible environment configuration`

## 2026-07-29：建立自动化测试与代码质量基线

### 原问题

项目没有自动化测试和覆盖率统计，路径计算、演示工具、CSV 加载及文件处理只能依赖手工验证。`listdir_with_allowed_type` 在目录不存在时错误返回允许的文件后缀元组，调用方可能继续把 `"txt"` 和 `"pdf"` 当作文件路径处理。

### 修改内容

1. 新增 `requirements-dev.txt`，在运行依赖基础上固定 pytest、pytest-cov 和 Ruff 版本。
2. 新增 `pyproject.toml`，统一配置测试发现、覆盖率统计和 Ruff 检查规则。
3. 新增 `tests/conftest.py`，使用测试专用占位 Key，避免测试依赖真实 DashScope 凭据。
4. 新增配置与路径测试，验证项目根目录、统一 Chroma 路径和确定性演示配置。
5. 新增工具测试，验证固定用户上下文、真实当前月份、天气免责声明、CSV 幂等加载、JSON字符串返回和缺失记录分支。
6. 新增文件工具测试，验证 MD5、文件类型过滤和目录不存在的边界行为。
7. 修复 `listdir_with_allowed_type`：目录不存在时返回空元组，不再返回文件后缀。

### 修改意义

- 核心工具和配置变更可以通过自动化测试回归验证。
- 测试不调用真实聊天模型或 Embedding API，也不要求开发者提供真实测试凭据。
- 覆盖率报告为后续补测提供可量化基线。
- 文件扫描失败时返回与函数语义一致的空集合，避免产生虚假路径和级联错误。

### 验证结果

- 首轮测试为9项通过、1项失败，失败项准确定位目录不存在时的错误返回值。
- 修复后单项回归测试输出 `1 passed`。
- 完整测试输出 `10 passed, 1 warning`。
- `agent`、`model`、`rag` 和 `utils` 共356条可执行语句，136条未覆盖，总覆盖率为62%。
- 当前仍存在1条 `langchain-community` 停止维护的弃用警告，已作为依赖迁移技术债保留，未通过过滤规则隐藏。
- Ruff 首轮发现13个可安全修复问题，包括10个导入顺序问题和3个无占位符 f-string。
- Ruff 自动修复13项后再次检查输出 `All checks passed!`。
- Ruff 修复后完整测试仍为10项通过，覆盖率保持62%。
- README 编写待完成。

### 相关提交

- 与本条记录同一提交：`test: add automated quality baseline`

## 2026-07-29：为核心改造补充解释性注释

### 原问题

前序改造已经实现统一向量库、多轮记忆、确定性演示工具、环境变量和自动化测试，但部分新增代码只体现“做了什么”，没有说明“为什么这样做”。会话隔离、缓存发布、CSV 编码、环境变量优先级等设计容易被后来者误删或误改。

### 修改内容

1. 为 Agent 检查点、thread ID、运行时报告标记和 Streamlit 会话状态补充注释。
2. 为 Chroma 绝对路径、MD5 状态、分片重叠和“先入库后记录 MD5”的顺序补充说明。
3. 为演示数据边界、CSV 局部加载、表头校验、重复记录和 JSON 返回约定补充说明。
4. 为 `.env` 优先级、必要变量快速失败和模型客户端复用补充说明。
5. 为安全 YAML 加载、分块 MD5 和目录不存在时返回空元组补充说明。
6. 为 Demo 与 Chroma YAML 配置、测试配置和开发依赖增加行内说明。
7. 为自动化测试增加测试意图 docstring，并解释 Unicode 转义、缓存隔离和边界测试。
8. 清理过时的大段注释代码，并统一少量相关格式与类型标注。

### 修改意义

- 后续开发者能理解关键实现背后的约束，而不仅是阅读语法。
- 注释重点覆盖容易引发数据错位、会话串线、密钥覆盖和缓存污染的风险点。
- 测试失败时可以从测试 docstring 快速理解被保护的业务行为。
- 配置和依赖文件可以直接说明各配置组的用途。

### 验证状态

- Ruff 静态检查输出 `All checks passed!`。
- 完整自动化测试输出 `10 passed, 1 warning`。
- 覆盖率回归统计354条可执行语句、136条未覆盖，总覆盖率保持62%。
- 唯一警告仍为已记录的 `langchain-community` 弃用技术债。

### 相关提交

- 与本条记录同一提交：`docs: explain core implementation with comments`

## 2026-07-29：统一使用可读的UTF-8中文

### 原问题

部分测试为了绕过终端编码问题使用了Unicode转义序列，降低了代码可读性。终端编码问题不应该通过降低源码可读性解决。

### 修改内容

1. 新增 `.editorconfig`，统一文本文件编码为UTF-8、换行为LF。
2. 将测试中的Unicode转义替换为直接中文字符串。
3. 新增文本编码测试，验证项目文本能够按严格UTF-8解码。
4. 新增转义扫描测试，禁止源码和数据重新引入Unicode转义序列。

### 修改意义

- 中文断言和数据字段可以直接阅读。
- IDE编码设置在仓库层面统一。
- 编码损坏和Unicode转义会被自动化测试发现。
- 终端显示问题与源码存储规则相互分离。

### 验证结果

- 全仓库转义扫描无输出，源码和数据中未发现反斜杠加字母u的四位十六进制转义形式。
- Ruff静态检查通过：`All checks passed!`。
- 文本编码专项测试通过：`2 passed`。
- 完整自动化测试通过：`12 passed, 1 warning`。
- 当前警告是已知的`langchain-community`弃用提醒，不影响本次编码规范修改。

### 相关提交

- 与本条记录同一提交：`style: use readable UTF-8 Chinese literals`
