package com.xixiciTest

import com.xixici.P63
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/5/9 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P63Test extends FunSuite {

  test("P63Test") {
    P63.insert(2)
    P63.insert(3)
    P63.insert(4)
    P63.insert(2)
    P63.insert(6)
    P63.insert(2)
    P63.insert(5)
    P63.insert(1)
    println(P63.getMedian())
    val Expected = 2.5
    val Actual = P63.getMedian()
    assert(Actual === Expected)
  }
}
