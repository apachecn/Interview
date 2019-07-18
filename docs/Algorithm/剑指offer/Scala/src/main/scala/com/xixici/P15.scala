package com.xixici

import com.xixici.util.ListNode

/**
  * Created by xixici
  * Date: 2019/3/13 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P15 {
  def ReverseList(listNode: ListNode): ListNode = {
    if (listNode == null)
      return null
    var head: ListNode = listNode
    var pre: ListNode = null
    //前一结点
    var next: ListNode = null //后一结点
    while (head != null) {
      next = head.next
      head.next = pre
      pre = head
      head = next
    }
    pre
  }
}
