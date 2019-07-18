package com.xixici

import com.xixici.util.TreeNode

/**
  * Created by xixici
  * Date: 2019/3/5 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P04 {
  def reConstructBinaryTree(pre: Array[Int], in: Array[Int]): TreeNode = {
    if (pre == null || in == null || pre.length <= 0 || in.length <= 0) {

    }
    if (pre.length != in.length) {
      throw new Exception("error, array length is not equal.")
    }

    val root: TreeNode = new TreeNode(-1)
    var flag: Boolean = false
    for (i <- 0 until pre.length) {
      if (pre(0) == in(i)) {
        flag = true
        root.data = pre(0)
        root.left = reConstructBinaryTree(java.util.Arrays.copyOfRange(pre, 1, i + 1),
          java.util.Arrays.copyOfRange(in, 0, i))
        root.right = reConstructBinaryTree(java.util.Arrays.copyOfRange(pre, i + 1, pre.length),
          java.util.Arrays.copyOfRange(in, i + 1, in.length))
      }
    }

    if (!flag) {
      throw new Exception("error, no root node.")
    }

    root
  }
}
