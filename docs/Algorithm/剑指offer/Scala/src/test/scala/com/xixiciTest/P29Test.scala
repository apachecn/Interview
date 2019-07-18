package com.xixiciTest

import com.xixici.P29
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/20 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P29Test extends FunSuite {
  val A = Array(2, 2, 2, 0, 2, 5, 3)
  test("P29Test") {
    val Expected = List(0, 2, 2, 2)
    val Actual = P29.GetLeastNumbers_Solution(A, 4)
    assert(Actual === Expected)
  }

}
