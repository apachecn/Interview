package com.xixiciTest

import com.xixici.P25
import com.xixici.util.RandomListNode
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/19 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P25Test extends FunSuite {
  val rln = new RandomListNode(1)
  val a = new RandomListNode(2)
  val b = new RandomListNode(3)
  val c = new RandomListNode(4)
  val d = new RandomListNode(5)
  rln.next = a
  rln.random = d
  a.next = b
  a.random = c
  b.next = c
  b.random = a
  c.next = null
  c.random = null
  test("P25Test") {
    val Expected = rln
    val Actual = P25.Clone(rln)
    assert(Actual.data === Expected.data)
  }

}
