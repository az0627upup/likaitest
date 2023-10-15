def permute(nums):
    def trackback(first=0):
        if first == n:
            return res.append(nums[:])
        else:
            for i in range(first, n):
                nums[first], nums[i] = nums[i], nums[first]
                trackback(first + 1)
                nums[first], nums[i] = nums[i], nums[first]

    n = len(nums)
    res = []
    trackback()
    return res

if __name__ == '__main__':
    r = permute(['a', 'b', 'c'])
    print(r)
