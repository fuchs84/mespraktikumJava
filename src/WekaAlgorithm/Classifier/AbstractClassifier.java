package WekaAlgorithm.Classifier;

import weka.classifiers.Classifier;
import weka.core.Instances;

/**
 * Abstract class specifies all methods for the classifiers
 */
public abstract class AbstractClassifier implements Runnable {

    /**
     * instances for train
     */
    protected Instances instances;

    /**
     * classified output
     */
    protected double[] classified;

    /**
     * Setter-method for the instances
     * @param instances Train-instances
     */
    public void setInstances(Instances instances) {
        this.instances = instances;
    }

    /**
     * Getter-method for the classified output
     * @return classified output
     */
    public double[] getClassified() {
        return classified;
    }

    /**
     * Abstract method trains the classifier
     * @throws Exception to invoking method
     */
    public abstract void train() throws Exception;

    /**
     * Abstract method classifies the instances
     * @throws Exception to invoking method
     */
    public abstract double[] classify() throws Exception;

    /**
     * Abstract Getter-method for classifier
     * @return used classifier
     */
    public abstract Classifier getClassifier();

    /**
     * Abstract train-method for threading
     */
    @Override
    public abstract void run();
}
