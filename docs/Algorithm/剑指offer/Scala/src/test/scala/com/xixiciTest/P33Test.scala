package com.xixiciTest

import com.xixici.P33
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/21 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P33Test extends FunSuite {
  test("P33Test") {
    val Expected = 12
    val Actual = P33.GetUglyNumber_Solution(10)
    assert(Actual === Expected)
  }
}