package com.xixiciTest

import com.xixici.P16
import com.xixici.util.ListNode
import org.scalatest.FunSuite

import scala.collection.mutable.ArrayBuffer

/**
  * Created by xixici
  * Date: 2019/3/14 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P16Test extends FunSuite {
  val list1 = new ListNode(1)
  val a = new ListNode(2)
  val b = new ListNode(5)
  val c = new ListNode(7)
  val d = new ListNode(8)
  list1.next = a
  a.next = b
  b.next = c
  c.next = d
  d.next = null
  val list2 = new ListNode(2)
  val a2 = new ListNode(4)
  val b2 = new ListNode(6)
  val c2 = new ListNode(8)
  val d2 = new ListNode(9)
  val e2 = new ListNode(10)
  list2.next = a2
  a2.next = b2
  b2.next = c2
  c2.next = d2
  d2.next = e2
  e2.next = null
  val list3 = new ListNode(1)
  val a3 = new ListNode(1)
  list3.next = a3
  a3.next = null
  val list4 = new ListNode(2)
  val a4 = new ListNode(3)
  list4.next = a4
  a4.next = null
  test("P16Test") {
    val Expected1 = Array(1, 1, 2, 4)
    var Actual1 = P16.Merge(list3, list4)
    val Expected = Array(1, 2, 2, 4, 5, 6, 7, 8, 8, 9, 10)
    var Actual = P16.Merge(list1, list2)
    val res = new ArrayBuffer[Int]
    while (Actual != null) {
      res.append(Actual.data)
      Actual = Actual.next
    }
    assert(res.toArray === Expected)
  }
}
