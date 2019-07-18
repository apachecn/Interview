package com.xixiciTest

import com.xixici.P46
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/4/26 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P46Test extends FunSuite {
  test("P46Test") {
    val Expected = 2
    val Actual = P46.LastRemaining_Solution(5, 2)
    assert(Actual === Expected)
  }
}

