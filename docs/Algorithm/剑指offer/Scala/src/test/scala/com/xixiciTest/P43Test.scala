package com.xixiciTest

import com.xixici.P43
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/4/25 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P43Test extends FunSuite {
  test("P43Test") {
    val Expected = "XYZdefabc"
    val Actual = P43.LeftRotateString("abcXYZdef", 3)
    assert(Actual === Expected)
  }
}
