package com.xixici

/**
  * Created by xixici
  * Date: 2019/3/25 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P37 {
  def GetNumberOfK(array: Array[Int], k: Int): Int = {
    val length = array.length
    if (length <= 0) return 0
    val firstK = getFirstK(array, k, 0, length - 1)
    val lastK = getLastK(array, k, 0, length - 1)
    if (firstK != -1 && lastK != -1) {
      return lastK - firstK + 1
    }
    0
  }

  def getFirstK(arr: Array[Int], k: Int, start: Int, end: Int): Int = {
    val mid = (start + end) >> 1
    if (arr(mid) > k) {
      getFirstK(arr, k, start, mid - 1)
    }
    else if (arr(mid) < k) {
      getFirstK(arr, k, mid + 1, end)
    }
    else if (mid - 1 >= 0 && arr(mid - 1) == k) {
      getFirstK(arr, k, start, mid - 1)
    } else {
      mid
    }
  }

  def getLastK(arr: Array[Int], k: Int, start: Int, end: Int): Int = {
    val mid = (start + end) >> 1
    if (arr(mid) > k) {
      getLastK(arr, k, start, mid - 1)
    }
    else if (arr(mid) < k) {
      getLastK(arr, k, mid + 1, end)
    }
    else if (mid + 1 <= end && arr(mid + 1) == k) {
      getLastK(arr, k, mid + 1, end)
    } else {
      mid
    }
  }
}
