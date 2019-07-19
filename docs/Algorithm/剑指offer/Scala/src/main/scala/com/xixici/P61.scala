package com.xixici

import com.xixici.util.TreeNode

/**
  * Created by xixici
  * Date: 2019/5/9 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P61 {

  var index = -1

  def serialize(root: TreeNode): String = {
    if (root == null) return ""
    val sb = new StringBuilder
    serialize(root, sb)
    sb.substring(0, sb.length - 1)
  }

  def deserialize(str: String): TreeNode = {
    if (str == null || str.length <= 0) return null
    val strs = str.split(",")
    deserialize(strs)
  }

  private def serialize(root: TreeNode, sb: StringBuilder): Unit = {
    if (root == null) {
      sb.append("$,")
      return
    }
    sb.append(root.data).append(",")
    serialize(root.left, sb)
    serialize(root.right, sb)

  }

  private def deserialize(strs: Array[String]): TreeNode = {
    index += 1
    if (!(strs(index) == "$")) {
      val root = new TreeNode(strs(index).toInt)
      root.left = deserialize(strs)
      root.right = deserialize(strs)
      return root
    }
    null
  }
}
