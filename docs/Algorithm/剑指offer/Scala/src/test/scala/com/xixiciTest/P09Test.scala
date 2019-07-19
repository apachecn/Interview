package com.xixiciTest

import com.xixici.P09
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/11 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P09Test extends FunSuite {
  test("P9Test") {
    val Expected = 8
    val Actual = P09.Jump2(4)
    assert(Expected === Actual)
  }
}
