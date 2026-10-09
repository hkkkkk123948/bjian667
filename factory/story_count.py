"""One count convention throughout the app: zero means all eligible stories."""
import math


def clip_limit(params):
    value = params.get('max_clips',0)
    if value is None or value == '' or value in ('auto','all'):
        return 0
    if isinstance(value,bool):
        raise ValueError('成片数量须为0（自动）或正整数')
    try:
        number = float(value)
    except (TypeError,ValueError) as exc:
        raise ValueError('成片数量须为0（自动）或正整数') from exc
    if not math.isfinite(number) or number < 0 or not number.is_integer():
        raise ValueError('成片数量须为0（自动）或正整数')
    return int(number)


def automatic_count(params):
    return clip_limit(params) == 0


def limit_items(items,params):
    limit = clip_limit(params)
    return items[:limit] if limit else items
