# 我们用 P(i,j) 表示字符串s 的第 i 到 j 个字母组成的串(s[i:j]）是否为回文串.也就是说，
# 只有 s[i+1:j−1] 是回文串，并且 s 的第 i 和 j 个字母相同时，s[i:j] 才会是回文串。
def longestPalindrome(s):
    n = len(s)
    d = [[False] * n for _ in range(n)]
    if n <= 1:
        return s
    maxlen = 1
    for i in range(n):
        d[i][i] = True
    begin = 0
    for L in range(2, n + 1):
        for i in range(0, n):
            j = i + L - 1
            if j >= n:
                break
            if s[i] != s[j]:
                d[i][j] = False
            else:
                if j - i < 3:
                    d[i][j] = True
                else:
                    d[i][j] = d[i + 1][j - 1]
            if d[i][j] and j - i + 1 > maxlen:
                maxlen = j - i + 1
                begin = i
    return s[begin:begin + maxlen]

# 滑动窗口
def longestPalindrom(s):
    s = s
    mlen = len(s)
    while True:
        i = 0
        while i + mlen <= len(s):
            sl = s[i:i + mlen]
            sr = sl[::-1]
            if sl == sr:
                return sl
            i = i + 1
        mlen = mlen - 1
        if mlen == 0:
            return ""
if __name__ == '__main__':
    m = longestPalindrom('fbabad')    # 推荐使用滑动窗口去做
    print(m)











