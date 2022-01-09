package com.xixiciTest

import com.xixici.P55
import com.xixici.util.ListNode
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/4/29 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P55Test extends FunSuite {
  val a = new ListNode(2)
  val b = new ListNode(3)
  val c = new ListNode(4)
  var head = new ListNode(1)
  head.next = a
  a.next = b
  b.next = c
  c.next = b
  test("P55Test") {
    val Expected = b
    val Actual = P55.EntryNodeOfLoop(head)
    assert(Actual === Expected)
  }

}