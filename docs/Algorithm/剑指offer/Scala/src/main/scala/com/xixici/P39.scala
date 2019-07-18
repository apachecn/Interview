package com.xixici

import com.xixici.util.TreeNode

/**
  * Created by xixici
  * Date: 2019/3/25 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P39 {
  var isBalanced = true

  def IsBalanced_Solution(root: TreeNode): Boolean = {

    getDepth(root)
    isBalanced

  }

  def getDepth(root: TreeNode): Int = {

    if (root == null)
      return 0
    val left = getDepth(root.left)
    val right = getDepth(root.right)

    if (Math.abs(left - right) > 1) {
      isBalanced = false
    }
    if (right > left) {
      right + 1
    } else {
      left + 1
    }

  }

}
