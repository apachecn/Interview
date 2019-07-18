package com.xixiciTest

import com.xixici.P31
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/22 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P31Test extends FunSuite {
  test("P33Test") {
    val Expected = 8
    val Actual = P31.NumberOf1Between1AndN_Solution(15)
    assert(Actual === Expected)
  }
}