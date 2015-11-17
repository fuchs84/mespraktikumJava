package DT;

/**
 * Created by MatthiasFuchs on 13.11.15.
 */
public class Node {

    /**
     * Attribute des Kontens
     */
    private int decisionAttribute = Integer.MIN_VALUE;
    private double decisionValueUpperBound;
    private double decisionValueLowerBound;
    private boolean leaf = false;
    private double classLabel = Double.NEGATIVE_INFINITY;

    /**
     * Verknüpfungen der Knoten untereinander
     */
    public Node left;
    public Node right;
    public Node parent;

    /**
     * Getter- und Setter-Methoden des Knotens
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
    public double getDecisionValueUpperBound() {
        return decisionValueUpperBound;
    }
    public void setDecisionValueUpperBound(double decisionValueUpperBound) {
        this.decisionValueUpperBound = decisionValueUpperBound;
    }
    public double getDecisionValueLowerBound() {
        return decisionValueLowerBound;
    }
    public void setDecisionValueLowerBound(double decisionValueLowerBound) {
        this.decisionValueLowerBound = decisionValueLowerBound;
    }
    public int getDecisionAttribute() {
        return decisionAttribute;
    }
    public void setDecisionAttribute(int decisionAttribute) {
        this.decisionAttribute = decisionAttribute;
    }
}
