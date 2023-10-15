def isValid(s: str):
    if len(s) % 2 == 1:
        return False
    pairs = {
        ")": "(",
        "]": "[",
        "}": "{",
    }
    stack = list()
    for ch in s:
        if ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
        else:
            stack.append(ch)
    return not stack


if __name__ == '__main__':
    s = isValid("([{}])")
    print(s)
