# coding:utf-8
"""
# 冒泡排序
# 1. 外层循环负责循环的次数，依次递减到1就停止(1个数不存在下一个值)
# 2. 内层循环负责前后两两比较, 判断是否需要交换位置，然后移动判断

5个数
5  0，1，2，3，4
4  0，1，2，3
3  0，1，2
2  0，1
"""

def bubble_sort(nums):
    # 判断外出循环的次数
    index = len(nums) - 1
    while index:
        print(index)
        # 第一个数字，和后面每一个数字进行对比，找出最大值，放到最后！！
        for i in range(index):
            if nums[i] > nums[i+1]:
                    nums[i], nums[i+1] = nums[i+1], nums[i]
        index -= 1


if __name__ == "__main__":
    nums = [3, 6, 8, 5, 2, 4, 9, 1, 7]
    bubble_sort(nums)
    print('result:', nums)
