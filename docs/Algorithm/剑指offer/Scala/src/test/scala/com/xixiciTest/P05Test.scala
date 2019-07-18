package com.xixiciTest

import com.xixici.P05
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/11 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P05Test extends FunSuite {

  test("testPush") {
    P05.push(1)
    P05.push(2)
    P05.push(3)
    assert(List(P05.pop(), P05.pop(), P05.pop()) === List(1, 2, 3))
  }

  test("testPop") {
    assertThrows[Exception](P05.pop())
  }
}
