# 极简安全计算器：仅允许 0-9 + - * / ( ) . 和空格
import re

SAFE_PATTERN = re.compile(r"^[0-9\s\+\-\*\/\(\)\.]+$")

def safe_eval(expr: str):
    expr = expr.strip()
    if not expr:
        return "[calc] 请输入表达式"
    if not SAFE_PATTERN.match(expr):
        return "[calc] 表达式包含非法字符"
    try:
        return str(eval(expr))
    except Exception as e:
        return f"[calc] 计算错误：{e}"
