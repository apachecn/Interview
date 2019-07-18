package com.xixici

import com.xixici.util.TreeNode

/**
  * Created by xixici
  * Date: 2019/5/8 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P58 {
  def comRoot(left: TreeNode, right: TreeNode): Boolean = {
    if (left == null) return right == null
    if (right == null) return false
    if (left.data != right.data) return false
    comRoot(left.right, right.left) && comRoot(left.left, right.right)
  }

  def isSymmetrical(pRoot: TreeNode): Boolean = {
    if (pRoot == null) {
      return true
    }
    comRoot(pRoot.left, pRoot.right)
  }
}
