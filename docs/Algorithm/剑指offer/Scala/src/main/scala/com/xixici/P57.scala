package com.xixici

import com.xixici.util.TreeLinkNode

/**
  * Created by xixici
  * Date: 2019/5/7 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P57 {
  def TreeLinkNode(pNodeO: TreeLinkNode): TreeLinkNode = {
    var pNode = pNodeO
    if (pNode == null)
      return pNode
    if (pNode.right != null) { // 节点有右子树
      pNode = pNode.right
      while (pNode.left != null) {
        pNode = pNode.left
      }
      return pNode
    }
    while (pNode.next != null) { //没右子树，则找第一个当前节点是父节点左孩子的节点
      if (pNode.next.left == pNode)
        return pNode.next
      pNode = pNode.next
    }
    null //退到了根节点仍没找到，则返回null
  }
}
