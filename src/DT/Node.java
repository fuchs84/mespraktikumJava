package DT;

/**
 * Created by MatthiasFuchs on 13.11.15.
 */
public class Node {
    private int decisionAttribute = Integer.MIN_VALUE;
    private double decisionValue = Double.NEGATIVE_INFINITY;
    private boolean leaf = false;
    private double classLabel = Double.NEGATIVE_INFINITY;

    public Node left;
    public Node right;
    public Node parent;

    public boolean getLeaf() {
        return leaf;
    }

    public double getClassLabel() {
        return classLabel;
    }

    public void setClassLabel(double classLabel) {
        this.classLabel = classLabel;
    }

    public void setLeaf(boolean leaf) {
        this.leaf = leaf;
        if(leaf == true) {
            right = null;
            left = null;
        }
    }

    public double getDecisionValue() {
        return decisionValue;
    }

    public void setDecisionValue(double decisionValue) {
        this.decisionValue = decisionValue;
    }

    public int getDecisionAttribute() {
        return decisionAttribute;
    }

    public void setDecisionAttribute(int decisionAttribute) {
        this.decisionAttribute = decisionAttribute;
    }
}
