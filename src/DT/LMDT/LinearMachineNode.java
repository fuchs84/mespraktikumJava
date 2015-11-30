package DT.LMDT;

/**
 * Created by MatthiasFuchs on 26.11.15.
 */
public class LinearMachineNode extends DT.Node {
    private double[] weights;

    public double[] getWeights(){
        return weights;
    }

    public void setWeights(double[] weights){
        this.weights = weights;
    }

    public void initWeights(int numberOfFeatures) {
        weights = new double[numberOfFeatures];
        for(int i = 0; i < weights.length; i++) {
            weights[i] = Math.random() - 0.5;
        }
    }
}
