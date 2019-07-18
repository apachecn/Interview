package com.xixici

/**
  * Created by xixici
  * Date: 2019/3/13 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P11 {
  def NumberOf1(n: Int) = {
    var count = 0
    var nNew = n
    while (nNew != 0) {
      count += 1
      nNew = nNew & (nNew - 1)
    }
    count
  }
}
