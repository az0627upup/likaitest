class Solution:
    def lengthOfLIS(self, nums) -> int:
        if not nums:
            return 0
        dp = []
        for i in range(len(nums)):
            dp.append(1)
            for j in range(i):
                if nums[i] > nums[j]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)


if __name__ == '__main__':
    solution = Solution()
    nums = [10, 9, 2, 5, 3, 7, 101, 18]
    x = solution.lengthOfLIS(nums)
    print(x)
