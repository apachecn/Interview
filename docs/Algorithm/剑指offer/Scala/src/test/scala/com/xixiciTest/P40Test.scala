package com.xixiciTest

import com.xixici.P40
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/25 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P40Test extends FunSuite {
  test("P40Test") {
    val ttt = Array(2, 2, 3, 3, 4, 7, 5, 5, 9, 9)
    val Expected = Array(4, 7)
    val Actual = P40.findNumsAppearOnce(ttt)
    assert(Actual.sorted === Expected.sorted)
  }
}