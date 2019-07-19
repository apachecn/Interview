package com.xixiciTest

import com.xixici.P36
import com.xixici.util.ListNode
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/22 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P36Test extends FunSuite {
  val list1 = new ListNode(1)
  val a = new ListNode(2)
  val b = new ListNode(3)
  val c = new ListNode(4)
  list1.next = a
  a.next = b
  b.next = c
  c.next = null
  val list2 = new ListNode(1)
  list2.next = b
  b.next = c
  c.next = a
  a.next = null
  test("P36Test") {
    val Expected = b
    val Actual = P36.FindFirstCommonNode(list1, list2)
    assert(Actual.data === Expected.data)
  }
}