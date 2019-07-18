package com.xixici

/**
  * Created by xixici
  * Date: 2019/3/13 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P06 {

  def minNumberInRotateArray(array: Array[Int]) = {
    var res = 0
    if (array.length < 1) {

    }
    else {
      var left = 0
      var right = array.length - 1
      var middle = -1
      while (array(left) >= array(right) && middle != right) {
        if ((right - left) == 1) {
          middle = right
        } else {
          middle = left + (right - left) / 2
          if (array(middle) >= array(left)) {
            left = middle
          }
          if (array(middle) <= array(right)) {
            right = middle
          }
        }
      }
      res = array(middle)
    }
    res
    //array.min
  }
}
