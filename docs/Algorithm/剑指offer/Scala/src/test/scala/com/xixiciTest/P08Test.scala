package com.xixiciTest

import com.xixici.P08
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/11 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P08Test extends FunSuite {
  test("P8Test") {
    val Expected = 5
    val Actual = P08.Jump(4)
    assert(Expected === Actual)
  }
}
