# 使用官方Python 3.13精简镜像，与项目当前开发版本保持一致。
FROM python:3.13-slim

# 禁止生成.pyc缓存文件，让容器文件更干净。
ENV PYTHONDONTWRITEBYTECODE=1

# 让Python日志立即输出，方便通过Docker查看实时日志。
ENV PYTHONUNBUFFERED=1

# 禁止pip保存下载缓存，减少镜像体积。
ENV PIP_NO_CACHE_DIR=1

# 禁止pip每次安装时检查自身更新，减少无关网络请求。
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# 后续命令都以容器内的/app目录作为工作目录。
WORKDIR /app

# 先单独复制依赖清单，以便Docker复用依赖安装缓存。
COPY requirements.txt ./

# 安装项目运行所需的固定版本依赖。
RUN python -m pip install --requirement requirements.txt

# 创建没有登录权限的普通用户，避免应用以root身份运行。
RUN groupadd --system appuser \
    && useradd \
        --system \
        --gid appuser \
        --create-home \
        --shell /usr/sbin/nologin \
        appuser

# 把经过.dockerignore过滤的项目文件复制到容器中。
COPY --chown=appuser:appuser . .

# 创建数据库和日志目录，并允许普通用户写入运行数据。
RUN mkdir -p /app/storage /app/logs \
    && chown -R appuser:appuser /app/storage /app/logs

# 后续启动命令使用普通用户执行，提高容器安全性。
USER appuser

# 声明Streamlit使用8501端口。
EXPOSE 8501

# 定期访问Streamlit健康检查接口，判断应用是否正常运行。
HEALTHCHECK \
    --interval=30s \
    --timeout=5s \
    --start-period=30s \
    --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=3)"

# 容器启动时运行Streamlit，并允许宿主机浏览器访问。
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
