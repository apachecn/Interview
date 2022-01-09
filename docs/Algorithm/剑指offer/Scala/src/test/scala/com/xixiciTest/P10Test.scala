package com.xixiciTest

import com.xixici.P10
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/13 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P10Test extends FunSuite {
  test("P10Test") {
    val Expected = 5
    val Actual = P10.RectCover(4)
    assert(Expected === Actual)
  }
}