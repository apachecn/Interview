package com.xixici

import scala.collection.mutable
import scala.collection.mutable.ListBuffer

/**
  * Created by xixici
  * Date: 2019/3/20 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P27 {
  val list = new collection.mutable.ListBuffer[String]()

  def Permutation(str: String): List[String] = {
    if (str != null && str.length() > 0) {
      PermutationHelper(str.toCharArray, 0, list)
      list.sorted
    }
    list.toList

  }


  def PermutationHelper(chars: Array[Char], i: Int, list: ListBuffer[String]): Unit = {
    if (i == chars.length - 1) {
      list.append(chars.mkString(""))
    } else {
      val charSet = new mutable.HashSet[Char]()
      for (j <- i until chars.length) {
        if (j == i || !charSet.contains(chars(j))) {
          charSet.add(chars(j))
          SwapTwo(chars, i, j)
          PermutationHelper(chars, i + 1, list)
          SwapTwo(chars, j, i)
        }
      }
    }
  }

  def SwapTwo(chars: Array[Char], i: Int, j:
  Int) = {
    val tmp = chars(i)
    chars(i) = chars(j)
    chars(j) = tmp
  }
}
