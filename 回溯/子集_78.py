class Solution:
    def subsets(self, nums):
        ans = []
        item = []

        def helper(i, item, ans):
            if i == len(nums):
                return
            item.append(nums[i])
            ans.append(item[:])
            helper(i + 1, item, ans)
            item.pop()
            helper(i + 1, item, ans)

        ans.append(item[:])
        helper(0, item, ans)
        return ans

if __name__ == '__main__':
    solution = Solution()
    nums = [1, 2, 3]
    su = solution.subsets(nums)
    print(su)