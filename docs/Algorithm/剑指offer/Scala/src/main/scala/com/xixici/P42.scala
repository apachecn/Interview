package com.xixici

import scala.collection.mutable

/**
  * Created by xixici
  * Date: 2019/3/30 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P42 {
  def FindNumbersWithSum(array: Array[Int], sum: Int): List[Int] = {
    var i = 0
    var j = array.length - 1
    val list = mutable.Buffer[Int]()
    while (i < j) {
      if (array(i) + array(j) == sum) {
        list.append(array(i))
        list.append(array(j))
        return list.toList
      } else if (array(i) + array(j) > sum) {
        j -= 1
      } else {
        i += 1
      }
    }
    list.toList
  }
}
