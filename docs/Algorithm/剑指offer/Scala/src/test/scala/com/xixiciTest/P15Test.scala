package com.xixiciTest

import com.xixici.P15
import com.xixici.util.ListNode
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/3/13 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P15Test extends FunSuite {
  val list1 = new ListNode(1)
  val a = new ListNode(2)
  val b = new ListNode(3)
  val c = new ListNode(4)
  list1.next = a
  a.next = b
  b.next = c
  c.next = null
  test("P15Test") {
    val Expected = c.data
    val Actual = P15.ReverseList(list1).data
    assert(Actual === Expected)
  }
}
