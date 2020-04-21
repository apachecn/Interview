"""
1. 二分查找是有条件的，首先是有序，其次因为二分查找操作的是下标，所以要求是顺序表
2. 最优时间复杂度：O(1)
3. 最坏时间复杂度：O(logn)
"""

# def binary_search(nums, data):
#     """
#     非递归解决二分查找
#     :param nums:
#     :return:
#     """
#     n = len(nums)
#     first = 0
#     last = n - 1
#     while first <= last:
#         mid = (last + first) // 2
#         if nums[mid] > data:
#             last = mid - 1
#         elif nums[mid] < data:
#             first = mid + 1
#         else:
#             return True
#     return False


def binary_search(nums, data):
    """
    递归解决二分查找: nums 是一个有序数组
    :param nums:
    :return:
    """
    n = len(nums)
    if n < 1:
        return False
    mid = n // 2
    if nums[mid] > data:
        return binary_search(nums[:mid], data)
    elif nums[mid] < data:
        return binary_search(nums[mid+1:], data)
    else:
        return True

if __name__ == '__main__':
    nums = [3, 6, 8, 5, 2, 4, 9, 1, 7]
    if binary_search(nums, 14):
        print('ok')
