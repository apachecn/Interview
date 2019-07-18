package com.xixiciTest

import com.xixici.P57
import com.xixici.util.TreeLinkNode
import org.scalatest.FunSuite


/**
  * Created by xixici
  * Date: 2019/5/7 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
/* d - b - h - e - i - a - f - c - g
*        a
*      /   \
*     b     c
*    / \   / \
*   d  e  f  g
*     / \
*    h  i
*    */
class P57Test extends FunSuite {
  val a = new TreeLinkNode(1)
  val b = new TreeLinkNode(2)
  val c = new TreeLinkNode(3)
  val d = new TreeLinkNode(4)
  val e = new TreeLinkNode(5)
  val f = new TreeLinkNode(6)
  val g = new TreeLinkNode(7)
  val h = new TreeLinkNode(8)
  val i = new TreeLinkNode(9)
  a.left = b
  a.right = c
  b.left = d
  b.right = e
  b.next = a
  e.left = h
  e.right = i
  e.next = b
  c.left = f
  c.right = g
  c.next = a
  d.next = b
  h.next = e
  i.next = e
  f.next = c
  g.next = c
  test("P57Test") {
    val Expected = f.data
    val Actual = P57.TreeLinkNode(a).data

    val Expected2 = b.data
    val Actual2 = P57.TreeLinkNode(d).data

    val Expected3 = h.data
    val Actual3 = P57.TreeLinkNode(b).data
    assert(Actual === Expected)
    assert(Actual2 === Expected2)
    assert(Actual3 === Expected3)
  }
}

