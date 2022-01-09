package com.xixiciTest

import com.xixici.P51
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/18 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P51Test extends FunSuite {
  val A = Array(2, 3, 1, 0, 2, 5, 3)
  test("P51Test") {
    val Expected = Array(0, 0, 0, 180, 0, 0, 0)
    val Actual = P51.multiply(A)
    assert(Actual === Expected)
  }

}
