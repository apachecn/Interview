package com.xixiciTest

import com.xixici.P61
import com.xixici.util.TreeNode
import org.scalatest.FunSuite

/**
  * Created by xixici
  * Date: 2019/5/9 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P61Test extends FunSuite {
  val a = new TreeNode(1)
  val b = new TreeNode(2)
  val bb = new TreeNode(3)
  val c = new TreeNode(4)
  val d = new TreeNode(5)
  a.left = b
  b.left = c
  b.right = d
  a.right = bb
  bb.left = d
  bb.right = c
  test("P61Test") {
    val Expected1 = "1,2,4,$,$,5,$,$,3,5,$,$,4,$,$"
    val Actual1 = P61.deserialize("1,2,4,$,$,5,$,$,3,5,$,$,4,$,$")
    val Actual0 = P61.serialize(Actual1)
    assert(Actual0 === Expected1)
  }
}