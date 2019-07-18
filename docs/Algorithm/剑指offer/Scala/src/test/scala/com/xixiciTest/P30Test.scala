package com.xixiciTest

import com.xixici.P30
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/21 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P30Test extends FunSuite {
  val A = Array(2, 2, 2, -9, 2, 5, 3)
  test("P30Test") {
    val Expected = 10
    val Actual = P30.FindGreatestSumOfSubarr(A)
    assert(Actual === Expected)
  }
}