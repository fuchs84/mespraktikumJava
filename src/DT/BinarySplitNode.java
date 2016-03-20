package DT;


/**
 * Binary-split node
 */
public class BinarySplitNode extends Node {

    /**
     * nodes attributes
     */
    private double decisionValueBound;

    /**
     * children nodes
     */
    public BinarySplitNode left;
    public BinarySplitNode right;

    /**
     * Getter- and setter-methods of the nodes
     */
    public double getDecisionValueBound() {
        return decisionValueBound;
    }
    public void setDecisionValueBound(double decisionValueBound) {
        this.decisionValueBound = decisionValueBound;
    }
}
