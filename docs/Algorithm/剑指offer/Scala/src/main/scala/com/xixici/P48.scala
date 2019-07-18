package com.xixici

/**
  * Created by xixici
  * Date: 2019/3/16 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P48 {
  def Add(num1: Int, num2: Int) = {
    var numA = num1
    var numB = num2
    while (numB != 0) {
      val sum = numA ^ numB
      val carray = (numA & numB) << 1
      numA = sum
      numB = carray
    }
    numA
  }
}
