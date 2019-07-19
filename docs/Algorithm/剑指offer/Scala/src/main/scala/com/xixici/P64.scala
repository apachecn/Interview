package com.xixici

import java.{util => ju}

import scala.collection.mutable.ListBuffer

/**
  * Created by xixici
  * Date: 2019/6/25 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P64 {

  def maxInWindows(nums: Array[Int], size: Int): List[Int] = {
    if (nums == null || nums.length <= 0 || size <= 0) return null

    val list = new ListBuffer[Int]
    // deque中保存索引 以支持滑动
    val deque = new ju.LinkedList[Int]()
    // 第一个窗口前
    for (i <- 0 until size - 1) {
      while (!deque.isEmpty && nums(i) > nums(deque.getLast)) {
        deque.removeLast()
      }
      deque.addLast(i)
    }
    // 窗口计算和滑动
    for (i <- size - 1 until nums.length) {
      while (!deque.isEmpty && nums(i) > nums(deque.getLast)) {
        deque.removeLast()
      }
      deque.addLast(i)
      // 滑动
      if (i - deque.getFirst + 1 > size) {
        deque.removeFirst()
      }
      list.append(nums(deque.getFirst))
    }

    list.toList
  }
}
