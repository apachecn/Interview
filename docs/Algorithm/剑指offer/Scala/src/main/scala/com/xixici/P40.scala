package com.xixici

/**
  * Created by xixici
  * Date: 2019/3/25 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P40 {

  def findNumsAppearOnce(array: Array[Int]): Array[Int] = {
    //    get(array)

    var num1 = 0
    var num2 = 0
    if (array == null || array.length <= 1) {
      num1 = 0
      num2 = 0
    }
    val len = array.length
    var sum = 0
    var index = 0
    for (i <- 0 until len) {
      sum ^= array(i)
    }
    var flag = true
    for (index2 <- 0 until 32 if flag) {
      if ((sum & (1 << index2)) != 0) {
        flag = false
        index = index2
      }
    }
    for (i <- 0 until len) {
      if ((array(i) & (1 << index)) != 0) {
        num2 ^= array(i)
      } else {
        num1 ^= array(i)
      }
    }
    Array(num1, num2)
  }


  def get(a: Array[Int]): Array[Int] = {
    if (a == null || a.length < 2) return null

    var number = 0
    for (i <- 0 until a.length) {
      number ^= a(i)
    }
    val index = firstBitis1(number)
    var num1 = 0
    var num2 = 0
    for (i <- 0 until a.length) {
      if (isBit1(a(i), index)) {
        num1 ^= a(i)
      } else num2 ^= a(i)
    }
    Array(num1, num2)
  }

  def firstBitis1(num: Int): Int = {
    var index = 0
    var _num = num
    while ((_num & 1) == 0) {
      _num >>= 1
      index += 1
    }
    index
  }

  def isBit1(num: Int, index: Int): Boolean = {
    var _num = num
    for (i <- 0 until index) {
      _num >>= index
    }
    (_num & 1) == 1
  }
}
