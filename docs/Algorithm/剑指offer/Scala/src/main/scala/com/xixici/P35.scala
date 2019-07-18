package com.xixici

/**
  * Created by xixici
  * Date: 2019/3/22 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P35 {
  def InversePairs(arr: Array[Int]): Int = {
    if (arr == null || arr.length == 0) {
      return 0
    }
    val copy = new Array[Int](arr.length)
    for (i <- arr.indices) {
      copy(i) = arr(i)
    }
    val count = InversePairsCore(arr, copy, 0, arr.length - 1)
    return count
    0
  }

  def InversePairsCore(arr: Array[Int], copy: Array[Int], low: Int, high: Int): Int = {
    if (low == high) {
      return 0
    }
    val mid = (low + high) >> 1
    val leftCount = InversePairsCore(arr, copy, low, mid) % 1000000007
    val rightCount = InversePairsCore(arr, copy, mid + 1, high) % 1000000007
    var count = 0
    var i = mid
    var j = high
    var locCopy = high
    while (i >= low && j > mid) {
      if (arr(i) > arr(j)) {
        count += (j - mid)
        copy(locCopy) = arr(i)
        locCopy -= 1
        i -= 1
        if (count >= 1000000007) {
          count %= 1000000007
        }
      }
      else {
        copy(locCopy) = arr(j)
        locCopy -= 1
        j -= 1
      }
    }
    for (i <- i until low by -1) {
      copy(locCopy) = arr(i)
      locCopy -= 1

    }
    for (j <- j until mid by -1) {
      copy(locCopy) = arr(j)
      locCopy -= 1

    }
    for (s <- low to high) {
      arr(s) = copy(s)
    }
    (leftCount + rightCount + count) % 1000000007

  }

}
