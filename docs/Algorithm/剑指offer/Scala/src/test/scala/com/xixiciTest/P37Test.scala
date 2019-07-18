package com.xixiciTest

import com.xixici.P37
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/25 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P37Test extends FunSuite {
  test("P37Test") {
    val Expected = 4
    val Actual = P37.GetNumberOfK(Array(1, 2, 3, 3, 3, 3, 4, 5, 6, 7), 3)
    assert(Actual === Expected)
  }
}
