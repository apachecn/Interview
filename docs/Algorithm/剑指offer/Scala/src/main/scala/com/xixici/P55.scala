package com.xixici

import com.xixici.util.ListNode

/**
  * Created by xixici
  * Date: 2019/4/29 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P55 {
  def EntryNodeOfLoop(pHead: ListNode): ListNode = {
    if (pHead == null || pHead.next == null || pHead.next.next == null) return null
    var fast = pHead.next.next
    var slow = pHead.next
    /////先判断有没有环
    while (fast != slow) {
      if (fast.next != null && fast.next.next != null) {
        fast = fast.next.next
        slow = slow.next
      } else { //没有环,返回
        return null
      }
    }
    //循环出来的话就是有环，且此时  fast == slow.
    fast = pHead
    while (fast != slow) {
      fast = fast.next
      slow = slow.next
    }
    slow
  }
}
