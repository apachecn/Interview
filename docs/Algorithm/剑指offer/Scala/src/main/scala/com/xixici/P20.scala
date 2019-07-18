package com.xixici

/**
  * Created by xixici
  * Date: 2019/3/14 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
class P20 {
  private val dataStack = new collection.mutable.ArrayStack[Int]
  private val minStack = new collection.mutable.ArrayStack[Int]

  def push(e: Int): Unit = {
    dataStack.push(e)
    if (minStack.isEmpty || e < minStack.top) {
      minStack.push(e)
    } else {
      minStack.push(minStack.top)
    }
  }

  def pop(): Int = {
    minStack.pop
    dataStack.pop
  }

  def min(): Int = {
    minStack.top
  }
}
