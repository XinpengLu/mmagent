"""
工具函数模块
提供 token 估算、数据处理等辅助功能
"""


def estimate_tokens(text: str) -> int:
    """
    估算文本的 token 数量

    使用启发式方法：
    - 中文字符：约 2 字符/token
    - 英文字符：约 4 字符/token
    - 混合文本：使用加权平均

    Args:
        text: 输入文本

    Returns:
        估算的 token 数量
    """
    if not text:
        return 0

    # 统计中文字符数
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    # 其他字符数
    other_chars = len(text) - chinese_chars

    # 中文约 2 字符/token，英文约 4 字符/token
    estimated_tokens = (chinese_chars / 2) + (other_chars / 4)

    return int(estimated_tokens)


def format_token_count(count: int) -> str:
    """
    格式化 token 数量显示

    Args:
        count: token 数量

    Returns:
        格式化的字符串，如 "1.2K" 或 "1.5M"
    """
    if count < 1000:
        return str(count)
    elif count < 1_000_000:
        return f"{count / 1000:.1f}K"
    else:
        return f"{count / 1_000_000:.1f}M"


def calculate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    input_price_per_million: float,
    output_price_per_million: float
) -> float:
    """
    计算调用成本

    Args:
        prompt_tokens: 输入 token 数
        completion_tokens: 输出 token 数
        input_price_per_million: 输入价格（每百万 token）
        output_price_per_million: 输出价格（每百万 token）

    Returns:
        成本（美元）
    """
    input_cost = (prompt_tokens / 1_000_000) * input_price_per_million
    output_cost = (completion_tokens / 1_000_000) * output_price_per_million
    return input_cost + output_cost


def truncate_string(s: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    截断字符串

    Args:
        s: 输入字符串
        max_length: 最大长度
        suffix: 截断后的后缀

    Returns:
        截断后的字符串
    """
    if len(s) <= max_length:
        return s
    return s[:max_length - len(suffix)] + suffix


def extract_model_name(model: str) -> str:
    """
    从完整模型名称中提取简短名称

    例如: "gpt-4-turbo-2024-04-09" -> "gpt-4-turbo"

    Args:
        model: 完整模型名称

    Returns:
        简短模型名称
    """
    # 移除日期后缀
    parts = model.split('-')
    if len(parts) > 2 and parts[-1].isdigit():
        # 可能是日期格式，移除最后的日期部分
        return '-'.join(parts[:-3]) if len(parts) > 3 else model
    return model


def safe_get(d: dict, *keys, default=None):
    """
    安全地从嵌套字典中获取值

    Args:
        d: 字典
        *keys: 键路径
        default: 默认值

    Returns:
        获取的值或默认值

    Example:
        safe_get({"a": {"b": {"c": 1}}}, "a", "b", "c")  # 返回 1
        safe_get({"a": {"b": {}}}, "a", "b", "c", default=0)  # 返回 0
    """
    result = d
    for key in keys:
        if isinstance(result, dict) and key in result:
            result = result[key]
        else:
            return default
    return result
