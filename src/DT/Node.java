package DT;

/**
 * Generally class for the nodes
 */
public class Node {

    /**
     * nodes attributes
     */
    protected int decisionAttribute = Integer.MIN_VALUE;
    protected boolean leaf = false;
    protected double classLabel = Double.NEGATIVE_INFINITY;

    //node depth
    public int deep;

    //parent node
    public Node parent;

    /**
     * Getter- and setter-methods of nodes attributes
     */
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
    }
    public int getDecisionAttribute() {
        return decisionAttribute;
    }
    public void setDecisionAttribute(int decisionAttribute) {
        this.decisionAttribute = decisionAttribute;
    }
}
