package com.xixiciTest

import com.xixici.P35
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/22 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P35Test extends FunSuite {
  test("P35Test") {
    val Expected = 7
    val Actual = P35.InversePairs(Array(1, 2, 3, 4, 5, 6, 7, 0))
    assert(Actual === Expected)
  }
}
