package com.xixici;

public class Q04 {
    public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
        return reConBTree(pre, 0, pre.length - 1, in, 0, in.length - 1);
    }

    public TreeNode reConBTree(int[] pre, int preleft, int preright, int[] in, int inleft, int inright) {
        if (preleft > preright || inleft > inright)// ????????????????null
            return null;
        // ??????TreeNode
        TreeNode root = new TreeNode(pre[preleft]);
        // ???????????????????????
        for (int i = inleft; i <= inright; i++) {
            if (pre[preleft] == in[i]) {
                // ????????????????????
                root.left = reConBTree(pre, preleft + 1, preleft + i - inleft, in, inleft, i - 1);
                // ????????????????????
                root.right = reConBTree(pre, preleft + i + 1 - inleft, preright, in, i + 1, inright);
            }
        }
        return root;
    }

    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }
}
