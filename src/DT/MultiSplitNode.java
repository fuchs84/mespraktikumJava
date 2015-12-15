package DT;

/**
 * Created by MatthiasFuchs on 13.11.15.
 */
public class MultiSplitNode extends DT.Node implements java.io.Serializable {

    /**
     * Attribute des Kontens
     */
    private double[] decisionValues;


    /**
     * Verknuepfungen der Knoten untereinander
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
