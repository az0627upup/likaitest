def get_common_str(lis_a):
    if len(lis_a) == 0:
        return ''
    common_str = ''
    min_str = min(lis_a, key=lambda x: len(x))
    for i in range(len(min_str)):
        flag = False
        for j in lis_a:
            if min_str[i] != j[i]:
                common_str = min_str[:i]
                flag = True
                break
        if flag:
            break
    else:
        return min_str
    return common_str


if __name__ == '__main__':
    a = ["flower", "flow", "flight"]
    print(get_common_str(a))
