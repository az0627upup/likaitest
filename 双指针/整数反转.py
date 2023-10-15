def reverse(x):
    if -10 < x < 10:
        return x
    x = str(x)
    if x[0] != '-':
        x = x[::-1]
        x = int(x)
    else:
        x = x[:0:-1]
        x = int(x)
        x = -x
    return x if -2147483648 < x < 2147483647 else 0


if __name__ == '__main__':
    m = reverse(-321)
    print(m)


