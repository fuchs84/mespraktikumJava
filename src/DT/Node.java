package DT;

/**
 * Created by MatthiasFuchs on 13.11.15.
 */
public class Node {

    /**
     * Attribute des Kontens
     */
    protected int decisionAttribute = Integer.MIN_VALUE;
    protected boolean leaf = false;
    protected double classLabel = Double.NEGATIVE_INFINITY;
    public int deep;
    /**
     * Verknuepfungen der Knoten untereinander
     */

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
    public int getDecisionAttribute() {
        return decisionAttribute;
    }
    public void setDecisionAttribute(int decisionAttribute) {
        this.decisionAttribute = decisionAttribute;
    }
}
