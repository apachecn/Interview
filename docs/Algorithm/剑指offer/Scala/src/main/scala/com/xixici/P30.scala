package com.xixici

/**
  * Created by xixici
  * Date: 2019/3/21 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P30 {
  def FindGreatestSumOfSubarr(arr: Array[Int]): Int = {
    if (arr.length == 0 || arr == null) {
      return 0
    }
    var curSum = 0
    var greatestSum = 0
    for (i <- 0 until arr.length) {
      if (curSum <= 0) {
        curSum = arr(i)
      } else {
        curSum += arr(i)
      }
      if (curSum > greatestSum) {
        greatestSum = curSum
      }
    }
    greatestSum
  }
}
