package com.xixiciTest

import com.xixici.P23
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/15 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P23Test extends FunSuite {
  val seq = Array(5, 7, 6, 9, 11, 10, 8)
  test("P23Test") {
    val Expected = true
    val Actual = P23.VerifySquenceOfBST(seq)
    assert(Actual === Expected)
  }
}
