

def debug_msg(is_debug_mode: bool, *args, **kwargs):
    if is_debug_mode:
        print(*args, **kwargs)
