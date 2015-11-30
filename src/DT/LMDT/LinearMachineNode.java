package DT.LMDT;

/**
 * Created by MatthiasFuchs on 26.11.15.
 */
public class LinearMachineNode extends DT.Node {

    /**
     * Attribute des Knotens
     */
    private double[] weights;

    /**
     * Getter- und Setter-Methoden des Knotens
     */
    public double[] getWeights(){
        return weights;
    }
    public void setWeights(double[] weights){
        this.weights = weights;
    }

    /**
     * Methode initialisiert die Gewichte
     * @param numberOfFeatures Anzahl der Gewichte (Features + 1)
     */
    public void initWeights(int numberOfFeatures) {
        weights = new double[numberOfFeatures];
        for(int i = 0; i < weights.length; i++) {
            weights[i] = Math.random() - 0.5;
        }
    }
}
