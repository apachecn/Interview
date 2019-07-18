package com.xixici

/**
  * Created by xixici
  * Date: 2019/3/18 
  * Project Name: sword-offer-scala-sbt
  * Project URL: https://github.com/xixici/sword-offer-scala
  **/
object P52 {
  def matchPattern(str: Array[Char], pattern: Array[Char]): Boolean = {
    if (str == null || pattern == null) {
      return false
    }
    val strIndex = 0
    val patternIndex = 0
    matchCore(str, strIndex, pattern, patternIndex)
  }

  def matchCore(str: Array[Char], strIndex: Int, pattern: Array[Char], patternIndex: Int): Boolean = {
    //有效性检验：str到尾，pattern到尾，匹配成功
    if (strIndex == str.length && patternIndex == pattern.length) {
      return true
    }
    //pattern先到尾，匹配失败
    if (strIndex != str.length && patternIndex == pattern.length) {
      return false
    }
    //模式第2个是*，且字符串第1个跟模式第1个匹配,分3种匹配模式；如不匹配，模式后移2位
    if (patternIndex + 1 < pattern.length && pattern(patternIndex + 1) == '*') {
      if ((strIndex != str.length && pattern(patternIndex) == str(strIndex)) || (pattern(patternIndex) == '.' && strIndex != str.length)) {
        //模式后移2，视为x*匹配0个字符
        //视为模式匹配1个字符
        //*匹配1个，再匹配str中的下一个
        return matchCore(str, strIndex, pattern, patternIndex + 2) || matchCore(str, strIndex + 1, pattern, patternIndex + 2) || matchCore(str, strIndex + 1, pattern, patternIndex)
      } else {
        return matchCore(str, strIndex, pattern, patternIndex + 2)
      }
    }
    //模式第2个不是*，且字符串第1个跟模式第1个匹配，则都后移1位，否则直接返回false
    if ((strIndex != str.length && pattern(patternIndex) == str(strIndex)) || (pattern(patternIndex) == '.' && strIndex != str.length)) {
      return matchCore(str, strIndex + 1, pattern, patternIndex + 1)
    }
    false
  }
}
