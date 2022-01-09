package com.xixici

/**
  * Created by xixici
  * Date: 2019/3/18 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P51 {
  def multiply(A: Array[Int]): Array[Int] = {
    val len = A.length
    val B = new Array[Int](len)
    B(0) = 1
    for (i <- 1 until len) {
      B(i) = B(i - 1) * A(i - 1)
    }
    var tmp = 1
    for (j <- (len - 2) to 0 by -1) {
      tmp *= A(j + 1)
      B(j) *= tmp
    }
    B
  }

}
