package com.xixiciTest

import com.xixici.P19
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/14 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P19Test extends FunSuite {
  val matrix = Array(Array(1, 2, 3, 4), Array(5, 6, 7, 8), Array(9, 10, 11, 12), Array(13, 14, 15, 16))
  test("P19Test") {
    val Expected = Array(1, 2, 3, 4, 8, 12, 16, 15, 14, 13, 9, 5, 6, 7, 11, 10)
    val Actual = P19.printMatrix(matrix)
    assert(Actual === Expected)
  }
}