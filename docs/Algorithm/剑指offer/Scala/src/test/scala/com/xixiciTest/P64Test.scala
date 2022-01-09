package com.xixiciTest

import com.xixici.P64
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/5/9 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P64Test extends FunSuite {
  val A = Array[Int](2, 3, 4, 2, 6, 2, 5, 1)
  test("P64Test") {
    val Expected = Array[Int](4, 4, 6, 6, 6, 5)
    val Actual = P64.maxInWindows(A, 3)
    assert(Actual === Expected)
  }
}
