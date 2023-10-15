class Solution:
    def countSubstrings(self, s: str) -> int:
        def speard(l, r):
            cnt = 0
            while l >= 0 and r < len(s):
                if s[l] == s[r]:
                    # print(s[l: r + 1]) # 左闭右开
                    l -= 1
                    r += 1
                    cnt += 1
                else:
                    break
            return cnt
        ans = 0
        for i in range(len(s)):
            ans += speard(i, i) # 中心是一个字符；
            ans += speard(i, i + 1) # 中心是2个字符；
        return ans

# 中心扩展思想，如果一个字符它两边的字符相等那么就是回文子串，或者两个字符的话如果这两个字符它两边的字符相等的话那么也是回文子串