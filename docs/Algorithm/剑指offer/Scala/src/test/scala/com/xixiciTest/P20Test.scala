package com.xixiciTest

import com.xixici.P20
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/14 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P20Test extends FunSuite {
  val stack = new P20
  stack.push(2)
  stack.push(3)
  stack.push(5)
  test("P20Test") {
    val Expected = 2
    val Actual = stack.min()
    assert(Actual === Expected)
  }
}
