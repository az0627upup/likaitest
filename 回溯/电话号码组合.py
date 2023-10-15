class Solution:
    def letterCombinations(self, digits: str):
        if not digits:
            return []
        phone = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl", "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
        def backtrack(conbination, nextdigit):
            if len(nextdigit) == 0:
                res.append(conbination)
            else:
                for letter in phone[nextdigit[0]]:
                    backtrack(conbination + letter, nextdigit[1:])
        res = []
        backtrack('', digits)
        return res


if __name__ == '__main__':
    s = Solution()
    re = s.letterCombinations("234")
    print(re)
