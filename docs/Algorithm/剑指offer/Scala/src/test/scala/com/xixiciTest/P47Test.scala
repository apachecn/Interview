package com.xixiciTest

import com.xixici.P47
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/16 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P47Test extends FunSuite {

  test("P47Test") {
    val Expected = 15
    val Actual = P47.Sum_Solution(5)
    assert(Actual === Expected)
  }
}
