package com.xixiciTest

import com.xixici.P42
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/4/25 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P42Test extends FunSuite {
  test("P42Test") {
    val Expected = List(1, 3)
    val Actual = P42.FindNumbersWithSum(Array(1, 2, 2, 3), 4)
    assert(Actual.sorted === Expected.sorted)
  }
}
