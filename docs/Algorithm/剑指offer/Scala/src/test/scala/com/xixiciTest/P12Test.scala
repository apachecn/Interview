package com.xixiciTest

import com.xixici.P12
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/13 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P12Test extends FunSuite {
  test("P12Test") {
    val Expected = 4
    val Actual = P12.Power(2, 2)
    val predict1 = 1 / 4
    val real1 = P12.Power(2, -2)
    assert(Actual === Expected, predict1 === real1)
  }
}
