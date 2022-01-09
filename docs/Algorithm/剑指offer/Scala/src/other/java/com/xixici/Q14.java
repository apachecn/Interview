package com.xixici;

public class Q14 {
    public ListNode FindKthToTail(ListNode head, int k) {
        ListNode pre = null, p = null;
        p = head;
        pre = head;
        int a = k;
        int count = 0;

        while (p != null) {
            p = p.next;
            count++;
            if (k < 1) {
                pre = pre.next;
            }
            k--;
        }
        if (count < a)
            return null;
        return pre;

    }

    public class ListNode {
        int val;
        ListNode next = null;

        ListNode(int val) {
            this.val = val;
        }
    }
}
