package com.xixiciTest

import com.xixici.P21
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/15 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P21Test extends FunSuite {
  val pushA = Array(1, 2, 3, 4, 5)
  val popA = Array(4, 5, 3, 2, 1)
  test("P21Test") {
    val Expected = true
    val Actual = P21.IsPopOrder(pushA, popA)
    assert(Actual === Expected)
  }
}
