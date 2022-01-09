package com.xixici

import com.xixici.util.TreeNode

/**
  * Created by xixici
  * Date: 2019/3/19 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P26 {
  var preNode: TreeNode = _

  def Convert(root: TreeNode): TreeNode = {
    if (root == null) return null
    convertNode(root)
    if (preNode == null) return null
    while (preNode.left != null) {
      preNode = preNode.left
    }
    preNode
  }

  def convertNode(current: TreeNode): Unit = {
    if (current == null) return
    if (current.left != null) {
      convertNode(current.left)
    }
    current.left = preNode
    if (preNode != null) {
      preNode.right = current
    }
    preNode = current
    if (current.right != null) {
      convertNode(current.right)
    }
  }
}
