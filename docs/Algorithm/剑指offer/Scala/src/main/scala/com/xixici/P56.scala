package com.xixici

import com.xixici.util.ListNode

/**
  * Created by xixici
  * Date: 2019/4/29 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P56 {
  def deleteDuplication(pHead: ListNode): ListNode = {
    if (pHead == null || pHead.next == null) { // 只有0个或1个结点，则返回
      return pHead
    }
    if (pHead.data == pHead.next.data) {
      // 当前结点是重复结点
      var pNode = pHead.next
      while (pNode != null && pNode.data == pHead.data) {
        // 跳过值与当前结点相同的全部结点,找到第一个与当前结点不同的结点
        pNode = pNode.next
      }
      deleteDuplication(pNode) // 从第一个与当前结点不同的结点开始递归
    } else {
      // 当前结点不是重复结点
      pHead.next = deleteDuplication(pHead.next) // 保留当前结点，从下一个结点开始递归
      pHead
    }
  }
}
