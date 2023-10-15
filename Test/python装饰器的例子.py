def log_decorator(func):
    def wrapper(*args, **kwargs):
        print(f"Calling function: {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Function {func.__name__} finished")
        print(result)
        return result
    return wrapper

@log_decorator
def add_numbers(a, b):
    return a + b

result = add_numbers(2, 3)
print(result)  # 输出: 5