package DT;

/**
 * Multi-split node
 */
public class MultiSplitNode extends DT.Node implements java.io.Serializable {

    /**
     * nodes attributes
     */
    private double[] decisionValues;

    /**
     * children nodes
     */
    public MultiSplitNode[] children;

    /**
     * Getter- and setter-methods of the nodes
     */
    public double[] getDecisionValues() {
        return decisionValues;
    }
    public void setDecisionValues(double[] decisionValues) {
        this.decisionValues = decisionValues;
    }
}
