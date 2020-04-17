
class Solution(object):
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            print(">>> %s: %s[%s], %s[%s], %s[%s]" % (target, nums[left], left, nums[right], right, nums[mid], mid))
            if nums[mid] > target:
                right = mid - 1
            elif nums[mid] < target:
                left = mid + 1
            else:
                break

        mid = mid+1 if left > mid else mid
        print("结果: ", mid, left)


if __name__ == "__main__":

    nums = [1, 2, 4, 5, 6, 7, 8, 9, 11, 15]
    target = -2
    s = Solution()
    s.searchInsert(nums, target)
