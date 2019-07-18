package com.xixici

import scala.collection.mutable

/**
  * Created by xixici
  * Date: 2019/3/14 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P21 {
  def IsPopOrder(pushA: Array[Int], popA: Array[Int]): Boolean = {
    val stackA = mutable.ArrayStack[Int]()
    var j = 0
    for (i <- 0 until pushA.length) {
      stackA.push(pushA(i))
      while (!stackA.isEmpty && stackA.top == popA(j)) {
        stackA.pop()
        j += 1
      }
    }
    stackA.isEmpty
  }
}
