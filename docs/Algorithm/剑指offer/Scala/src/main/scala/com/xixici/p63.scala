package com.xixici

import java.util.Comparator
import java.{util => ju}

/**
  * Created by xixici
  * Date: 2019/5/13 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P63 {

  val minHeap = new ju.PriorityQueue[Int](15)
  val maxHeap = new ju.PriorityQueue[Int](15, new Comparator[Int] {
    override def compare(o1: Int, o2: Int): Int = o2 - o1
  })
  var count = 0

  def insert(num: Int) = {
    if (count % 2 == 0) {
      maxHeap.offer(num)
      val filterNum = maxHeap.poll()
      minHeap.offer(filterNum)
    } else {
      minHeap.offer(num)
      val filterNum = minHeap.poll()
      maxHeap.offer(filterNum)
    }
    count += 1
  }

  // 最大堆最小堆法
  def getMedian(): Double = {
    if (count % 2 == 0) {
      (maxHeap.peek() + minHeap.peek()) / 2.0
    } else {
      minHeap.peek().toDouble
    }
  }

}
