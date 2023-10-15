def args(length, *flag, **entropy):
    len = length
    fla = flag
    entro = entropy
    print(len)
    print(fla)
    print(fla[0][0])
    print(entro)


if __name__ == '__main__':
    args(2, (41, 'ar', 000000), 42, 0000, *(23, 45), a=8, b="hello", **{"c": 24})
    m = (10, 15, 'ac', 23, 'jdk')
    print("m的值是", m[2])
