package com.xixiciTest

import com.xixici.P50
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/18 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P50Test extends FunSuite {
  test("P50Test") {
    val Expected = true
    val Actual = P50.duplicate(Array(2, 3, 1, 0, 2, 5, 3), 7, Array(2))
    assert(Actual === Expected)
  }
}
