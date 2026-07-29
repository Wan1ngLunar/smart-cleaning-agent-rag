import os

# 测试只初始化模型对象，不发送 API 请求；占位值可避免读取开发者的真实密钥。
os.environ.setdefault(
    "DASHSCOPE_API_KEY",
    "test-only-placeholder",
)
