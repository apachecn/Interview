package com.xixiciTest

import com.xixici.P13
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/13 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P13Test extends FunSuite {
  test("P13Test") {
    val Expected = Array(1, 3, 5, 2, 4, 8)
    val Actual = P13.reOrderArray(Array(1, 2, 3, 4, 5, 8))
    assert(Actual === Expected)
  }
}
