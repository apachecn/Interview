package com.xixiciTest

import com.xixici.P56
import com.xixici.util.ListNode
import org.scalatest.FunSuite

import scala.collection.mutable.ArrayBuffer

/**
  * Created by xixici
  * Date: 2019/4/29 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P56Test extends FunSuite {
  val a = new ListNode(2)
  val b = new ListNode(3)
  val c = new ListNode(4)
  val d = new ListNode(4)
  val e = new ListNode(5)
  var head = new ListNode(1)
  head.next = a
  a.next = b
  b.next = c
  c.next = d
  d.next = e
  test("P56Test") {
    var Expected = Array(1, 2, 3, 5)
    var Actual = P56.deleteDuplication(head)
    val res = new ArrayBuffer[Int]
    while (Actual != null) {
      res.append(Actual.data)
      Actual = Actual.next
    }
    assert(res.toArray === Expected)
  }

}