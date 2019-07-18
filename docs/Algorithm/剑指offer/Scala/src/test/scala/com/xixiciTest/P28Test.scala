package com.xixiciTest

import com.xixici.P28
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/18 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P28Test extends FunSuite {
  val A = Array(2, 2, 2, 0, 2, 5, 3)
  test("P28Test") {
    val Expected = 2
    val Actual = P28.MoreThanHalfNum_Solution(A)
    assert(Actual === Expected)
  }
}
