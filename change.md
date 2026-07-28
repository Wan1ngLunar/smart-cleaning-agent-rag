# 项目改造记录

本文档记录项目在原始版本基础上的增量改造，帮助后续开发者了解每项修改的背景、实现方式和验证结果。

## 记录规则

- 每次新增或修改代码时，同步更新本文档。
- 每项变更应说明：原问题、修改内容、修改意义、验证方式和相关提交。
- 只记录实际完成并验证过的结果；未完成的验证明确标记为“待验证”。
- 密钥、用户隐私数据和完整日志不得写入本文档。

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
