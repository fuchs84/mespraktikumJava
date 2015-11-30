package DT.MSDT;

/**
 * Created by MatthiasFuchs on 13.11.15.
 */
public class MultiSplitNode extends DT.Node{

    /**
     * Attribute des Kontens
     */
    private double[] decisionValues;


    /**
     * Verknüpfungen der Knoten untereinander
     */
    public MultiSplitNode[] children;

    /**
     * Getter- und Setter-Methoden des Knotens
     */
    public double[] getDecisionValues() {
        return decisionValues;
    }
    public void setDecisionValues(double[] decisionValues) {
        this.decisionValues = decisionValues;
    }
}
