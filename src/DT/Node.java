package DT;

/**
 * Created by MatthiasFuchs on 13.11.15.
 */
public class Node {
    private int decisionAttribute = Integer.MIN_VALUE;
    private double decisionValue = Double.NEGATIVE_INFINITY;
    private boolean leaf = false;

    public Node previousNode = null;

    public boolean getLeaf() {
        return leaf;
    }

    public void setLeaf(boolean leaf) {
        this.leaf = leaf;
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
