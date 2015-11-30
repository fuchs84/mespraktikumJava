package DT.BSDT;

/**
 * Created by MatthiasFuchs on 13.11.15.
 */
public class BinarySplitNode extends DT.Node{

    /**
     * Attribute des Kontens
     */
    private double decisionValueBound;

    /**
     * Verknüpfungen der Knoten untereinander
     */
    public BinarySplitNode left;
    public BinarySplitNode right;

    /**
     * Getter- und Setter-Methoden des Knotens
     */
    public double getDecisionValueBound() {
        return decisionValueBound;
    }
    public void setDecisionValueBound(double decisionValueBound) {
        this.decisionValueBound = decisionValueBound;
    }
}
