package com.xixici

import com.xixici.util.TreeNode

import scala.collection.mutable

/**
  * Created by xixici
  * Date: 2019/5/9 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P60 {
  def Print(pNode: TreeNode) = {
    var layer = 1
    val list = mutable.ArrayBuffer[mutable.ArrayBuffer[Int]]()
    val s1 = mutable.ArrayStack[TreeNode]()
    val s2 = mutable.ArrayStack[TreeNode]()
    s1.push(pNode)
    while (s1.nonEmpty || s2.nonEmpty) {
      if (layer % 2 != 0) {
        val temp = mutable.ArrayBuffer[Int]()
        while (s1.nonEmpty) {
          val node = s1.pop()
          if (node != null) {
            temp.append(node.data)
            s2.push(node.right)
            s2.push(node.left)
          }
        }
        layer += 1
        if (temp.nonEmpty) {
          list.append(temp)
        }
      } else {
        val temp = mutable.ArrayBuffer[Int]()
        while (s2.nonEmpty) {
          val node = s2.pop()
          if (node != null) {
            temp.append(node.data)
            s1.push(node.left)
            s1.push(node.right)
          }
        }
        layer += 1
        if (temp.nonEmpty) {
          list.append(temp)
        }


      }
    }
    list
  }

}
