package com.xixiciTest

import com.xixici.P54
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/4/29 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P54Test extends FunSuite {
  test("P54Test") {
    val google = "google".toCharArray
    val s = new StringBuffer()
    for (c: Char <- google) {
      P54.Insert(c)
      s.append(P54.FirstAppearingOnce())
    }
    val Expected = "ggg#ll"
    val Actual = s.toString
    assert(Actual === Expected)

  }
}