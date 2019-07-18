package com.xixiciTest

import com.xixici.P11
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/13 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P11Test extends FunSuite {
  test("P11Test") {
    val Expected = 1
    val Actual = P11.NumberOf1(4)
    assert(Expected === Actual)
  }
}
