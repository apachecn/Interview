class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        lookup = {}
        for i, num in enumerate(nums):
            if target - num in lookup:
                return [lookup[target - num],i]
            lookup[num] = i
        return []


nums = [2, 7, 11, 15]
target = 9
s = Solution()
print(s.twoSum(nums, target))
