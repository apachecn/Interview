package com.xixiciTest

import com.xixici.P41
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/4/25 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P41Test extends FunSuite {
  test("P41Test") {
    val Expected = List(9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22)
    val Actual = P41.FindContinuousSequence(100)
    assert(Actual.sorted === Expected.sorted)
  }
}
